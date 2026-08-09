"""JSON-lines worker for the isolated NVIDIA Parakeet virtual environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import wave
from pathlib import Path
from typing import Any


PROTOCOL_MARKER = "__PARAKEET_RESULT__"
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
# Long recordings are split into 20-second clips, then submitted to Parakeet
# in bounded groups.  Keeping a whole two-hour lecture in one NeMo
# ``transcribe`` invocation can leave allocator state resident on older CUDA
# cards; a group boundary gives the CUDA cache an explicit reset point.
MAX_CLIPS_PER_INFERENCE = 48


def send(value: dict[str, Any]) -> None:
    print(PROTOCOL_MARKER + json.dumps(value, ensure_ascii=False), flush=True)


def prepare_mono_audio(source: Path, destination: Path) -> None:
    bundled = Path(__file__).with_name("ffmpeg.exe")
    executable = str(bundled) if bundled.is_file() else "ffmpeg"
    completed = subprocess.run(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or "unknown FFmpeg error"
        raise RuntimeError(f"Could not prepare mono audio for Parakeet: {detail}")


def split_wav(source: Path, output_dir: Path, *, seconds: int = 20) -> tuple[list[Path], float]:
    clips: list[Path] = []
    with wave.open(str(source), "rb") as reader:
        if reader.getnchannels() != 1:
            raise RuntimeError("Parakeet preparation did not produce mono audio")
        sample_rate = reader.getframerate()
        duration = reader.getnframes() / sample_rate
        frames_per_clip = sample_rate * seconds
        parameters = reader.getparams()
        index = 1
        while frames := reader.readframes(frames_per_clip):
            clip = output_dir / f"clip-{index:05d}.wav"
            with wave.open(str(clip), "wb") as writer:
                writer.setparams(parameters)
                writer.writeframes(frames)
            clips.append(clip)
            index += 1
    if not clips:
        raise RuntimeError("Prepared audio contains no samples")
    return clips, duration


def _clip_batches(clip_count: int, max_size: int) -> list[tuple[int, int]]:
    """Return (start, length) pairs covering every clip, chunked by max_size.

    Every native CUDA crash observed in production landed on a final batch
    of exactly one clip (e.g. 145 clips = 48+48+48+1) -- model.transcribe()
    aborted the whole process with a fatal, non-Python-catchable exception
    each time. Rather than submit a lone trailing clip on its own, fold it
    into the previous batch so no batch this function returns is ever size 1
    (except when the source has only one clip in total, which has nothing
    to fold into).
    """

    boundaries: list[tuple[int, int]] = []
    start = 0
    while start < clip_count:
        remaining = clip_count - start
        size = min(max_size, remaining)
        if remaining - size == 1:
            size += 1
        boundaries.append((start, size))
        start += size
    return boundaries


def transcribe_one(model: Any, source: Path, *, model_name: str) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="parakeet-audio-") as temporary:
        temporary_path = Path(temporary)
        prepared = temporary_path / "input.wav"
        prepare_mono_audio(source, prepared)
        clips, duration = split_wav(prepared, temporary_path)
        device = next(model.parameters()).device
        hypotheses: list[Any] = []
        batches = _clip_batches(len(clips), MAX_CLIPS_PER_INFERENCE)
        total_batches = len(batches)
        for batch_index, (start, size) in enumerate(batches, 1):
            batch = clips[start : start + size]
            print(
                f"Transcribing clip batch {batch_index}/{total_batches} "
                f"({start + 1}-{start + len(batch)} of {len(clips)})",
                file=sys.stderr,
                flush=True,
            )
            with torch.inference_mode():
                batch_hypotheses = model.transcribe(
                    [str(clip) for clip in batch],
                    batch_size=1,
                    num_workers=0,
                    use_lhotse=False,
                    verbose=False,
                )
            # Every downstream segment timestamp is derived purely from its
            # position in `hypotheses` (`start = index * 20.0`), on the
            # assumption that result N always corresponds to clip N. NeMo is
            # not guaranteed to preserve that 1:1 correspondence -- a single
            # silently dropped or reordered clip anywhere in a batch shifts
            # every later timestamp, and because the audio's true tail then
            # has no slot left in the (now-short) list, real trailing speech
            # is silently discarded rather than merely mislabelled. This was
            # confirmed happening in production: several multi-hour lectures
            # had 5-30+ minutes of real, non-silent speech missing from the
            # end of their transcripts with no error raised anywhere. Treat
            # any count mismatch as fatal rather than let it corrupt the
            # timeline silently.
            if len(batch_hypotheses) != len(batch):
                raise RuntimeError(
                    f"Parakeet returned {len(batch_hypotheses)} result(s) for "
                    f"clip batch {batch_index}/{total_batches} but {len(batch)} "
                    "clip(s) were submitted; refusing to build a transcript "
                    "with drifted timestamps"
                )
            hypotheses.extend(batch_hypotheses)
            if device.type == "cuda":
                # The result list is now CPU-side Python objects. Synchronise
                # the completed group, then release cached intermediate
                # tensors before the next group begins.
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    texts = [str(getattr(item, "text", item)).strip() for item in hypotheses]
    if not any(texts):
        raise RuntimeError("Parakeet returned no speech text")
    clip_seconds = duration / len(clips) if len(clips) == 1 else 20.0
    segments = [
        {
            "id": index,
            "start": round(index * 20.0, 3),
            "end": round(min(duration, (index + 1) * 20.0), 3),
            "text": text,
            "timing": "clip-aligned",
        }
        for index, text in enumerate(texts)
        if text
    ]
    # Clip seams are implementation details, not prose paragraphs. The GLM
    # editor receives continuous text and chooses its own coherent paragraphs.
    raw_text = " ".join(text for text in texts if text)
    return {
        "raw_text": raw_text,
        "text": raw_text,
        "segments": segments,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "metadata": {
            "model": model_name,
            "engine": "NVIDIA Parakeet TDT 0.6B v3",
            "backend": "nvidia-parakeet",
            "device": str(next(model.parameters()).device),
            "audio_duration_seconds": round(duration, 3),
            "timing_evidence": "20-second clip-aligned segments (not word timestamps)",
            "clip_count": len(clips),
        },
    }


def serve(model_name: str, device_name: str) -> int:
    import torch
    from nemo.collections.asr.models import ASRModel

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Parakeet but no CUDA GPU is available")
    device = torch.device("cuda" if device_name == "cuda" else "cpu")
    print(f"Loading {model_name} on {device}", file=sys.stderr, flush=True)
    model = ASRModel.from_pretrained(model_name=model_name, map_location=device)
    model.eval()
    send({"id": "ready", "status": "ready", "model": model_name, "device": str(device)})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = str(request.get("id") or "")
            operation = request.get("op")
            if operation == "shutdown":
                send({"id": request_id, "status": "stopped"})
                return 0
            if operation != "transcribe":
                raise ValueError("unsupported Parakeet worker operation")
            source = Path(str(request.get("source") or "")).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"audio source is unavailable: {source}")
            print(f"Transcribing {source.name}", file=sys.stderr, flush=True)
            send(
                {
                    "id": request_id,
                    "status": "ok",
                    "details": transcribe_one(model, source, model_name=model_name),
                }
            )
        except Exception as exc:
            send(
                {
                    "id": str(locals().get("request_id") or "unknown"),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=12),
                }
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent local Parakeet worker")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not args.serve:
        parser.error("--serve is required")
    return serve(args.model, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
