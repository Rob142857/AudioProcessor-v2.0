"""Run a local NVIDIA Parakeet transcription as a controlled comparison.

This deliberately does not enter the production publication pipeline.  It
creates only a plain-text transcript and a small provenance JSON file beside
the selected test recording, so a human can compare it with Faster-Whisper
before enabling Parakeet for archive work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path


AUDIO_EXTENSIONS = frozenset(
    {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".wav", ".wma"}
)
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".parakeet-", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_audio(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
        return path
    if path.is_dir():
        files = sorted(
            candidate for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS
        )
        if len(files) == 1:
            return files[0]
        if not files:
            raise ValueError(f"No supported audio file found in {path}")
        raise ValueError(
            f"Expected exactly one audio file in {path}; found {len(files)}. "
            "Choose a file explicitly for a controlled comparison."
        )
    raise ValueError(f"Audio input does not exist: {path}")


def prepare_mono_audio(source: Path, destination: Path) -> None:
    """Decode one archive recording to Parakeet's expected mono 16 kHz WAV."""

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
        message = completed.stderr.strip() or "unknown FFmpeg error"
        raise RuntimeError(f"Could not prepare mono audio for Parakeet: {message}")


def split_wav_for_parakeet(source: Path, output_dir: Path, *, seconds: int = 20) -> list[Path]:
    """Split a mono PCM WAV into GPU-safe sequential ASR clips."""

    clips: list[Path] = []
    with wave.open(str(source), "rb") as reader:
        if reader.getnchannels() != 1:
            raise RuntimeError("Parakeet preparation did not produce mono audio.")
        frames_per_clip = reader.getframerate() * seconds
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
        raise RuntimeError("Prepared audio contains no samples.")
    return clips


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a local Parakeet v3 comparison transcript."
    )
    parser.add_argument("input", help="One audio file, or a folder containing exactly one")
    parser.add_argument("--output", help="Output transcript path (defaults beside the audio)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--reuse-transcript",
        action="store_true",
        help="Reuse an existing Parakeet text output instead of loading the speech model.",
    )
    parser.add_argument(
        "--glm-cleanup",
        action="store_true",
        help="Run the protected glossary-backed GLM review after Parakeet transcription.",
    )
    args = parser.parse_args()

    audio = resolve_audio(Path(args.input))
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else audio.with_name(f"{audio.stem} - Parakeet v3.txt")
    )
    metadata_path = output.with_suffix(".parakeet.json")

    started = time.perf_counter()
    device = "reused"
    if args.reuse_transcript:
        if not output.is_file():
            raise FileNotFoundError(f"No existing Parakeet transcript to reuse: {output}")
        text = output.read_text(encoding="utf-8").strip()
        print(f"Reusing {output.name}; speech model not loaded.", flush=True)
    else:
        import torch
        from nemo.collections.asr.models import ASRModel

        use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f"Loading {args.model} on {device}…", flush=True)
        model = ASRModel.from_pretrained(model_name=args.model, map_location=device)
        model.eval()
        print(f"Preparing mono 16 kHz audio for {audio.name}…", flush=True)
        with tempfile.TemporaryDirectory(prefix="parakeet-audio-") as temporary:
            prepared_audio = Path(temporary) / "input.wav"
            prepare_mono_audio(audio, prepared_audio)
            clips = split_wav_for_parakeet(prepared_audio, Path(temporary))
            print(f"Transcribing {audio.name} in {len(clips)} sequential clips…", flush=True)
            transcription = model.transcribe(
                [str(clip) for clip in clips],
                batch_size=1,
                num_workers=0,
                use_lhotse=False,
            )
        if not transcription:
            raise RuntimeError("Parakeet returned no transcription.")
        text = "\n\n".join(
            str(getattr(item, "text", item)).strip()
            for item in transcription
            if str(getattr(item, "text", item)).strip()
        )
    if not text:
        raise RuntimeError("Parakeet returned an empty transcription.")

    elapsed = round(time.perf_counter() - started, 3)
    atomic_write_text(output, text + "\n")
    atomic_write_text(
        metadata_path,
        json.dumps(
            {
                "engine": "NVIDIA Parakeet",
                "model": args.model,
                "audio": str(audio),
                "output": str(output),
                "device": str(device),
                "elapsed_seconds": elapsed,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    print(f"COMPLETE: {output}")
    print(f"Elapsed: {elapsed:.1f}s")

    if args.glm_cleanup:
        from cleanup_client import CleanupClient
        from txt_to_docx import convert_txt_to_docx_from_text

        print("Running protected glossary-backed GLM review…", flush=True)
        client = CleanupClient.from_environment()
        cleanup = client.cleanup_text(
            text,
            checkpoint_dir=output.parent / ".parakeet-cleanup-checkpoints",
        )
        review_text = cleanup.text.strip()
        if not review_text:
            raise RuntimeError("GLM cleanup returned an empty transcript.")
        review_txt = output.with_name(f"{audio.stem} - Parakeet v3 - GLM Review.txt")
        review_docx = review_txt.with_suffix(".docx")
        atomic_write_text(review_txt, review_text + "\n")
        convert_txt_to_docx_from_text(
            review_text,
            audio,
            output_path=review_docx,
            needs_human_review=True,
        )
        atomic_write_text(
            review_txt.with_suffix(".glm.json"),
            json.dumps(
                {
                    "engine": "NVIDIA Parakeet -> GLM review",
                    "parakeet_model": args.model,
                    "glm_model": cleanup.model,
                    "glossary_count": cleanup.glossary_count,
                    "glossary_sha256": cleanup.glossary_sha256,
                    "chunk_count": len(cleanup.chunks),
                    "needs_review": cleanup.needs_review,
                    "warnings": cleanup.warnings,
                    "source_transcript": str(output),
                    "review_text": str(review_txt),
                    "review_docx": str(review_docx),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        print(f"GLM REVIEW COMPLETE: {review_docx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
