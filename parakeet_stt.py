"""Persistent local NVIDIA Parakeet speech-to-text session.

The archive orchestration runs in the normal application virtual environment,
whereas NeMo and its CUDA build live in ``.parakeet-venv``.  This module keeps
one small JSON-lines worker alive in that environment so the Parakeet model is
loaded once for a batch, rather than once per recording.

It deliberately performs *only* local speech-to-text.  The protected GLM
review remains in the archive pipeline and may run concurrently in separate
workers after each raw transcript has been made durable.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from stt_coverage import finite_seconds


DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
PROTOCOL_MARKER = "__PARAKEET_RESULT__"
WORKER_READY_TIMEOUT_SECONDS = 300
TRANSCRIBE_TIMEOUT_SECONDS = 60 * 60
# A native CUDA/PyTorch abort terminates the isolated worker but does not harm
# durable pipeline artifacts. Restart once and retry the exact source before
# treating that recording as failed.
WORKER_CRASH_RETRIES = 1
# NeMo's mel-spectrogram front end (AudioToMelSpectrogramPreprocessor) needs
# at least one hop's worth of audio to produce a single mel frame. The
# nvidia/parakeet-tdt-0.6b-v3 checkpoint's own model_config.yaml pins
# sample_rate=16000 and window_stride=0.01s, i.e. hop_length = 160 samples =
# 0.01s -- and right up to two hops of audio, NeMo's `normalize_batch` divides
# by a torch.std() of nan and raises a raw ValueError ("received a tensor of
# length 1 ... make sure your audio length has enough samples for a single
# feature (at least hop_length for Mel Spectrograms)") instead of a usable
# message. 0.1s is a ~10x margin over that boundary -- enough to comfortably
# clear the 0/1-frame crash zone plus resampling rounding, without hard-coding
# a specific downloaded checkpoint's exact internal hop length.
MIN_TRANSCRIBABLE_SECONDS = 0.1


class ParakeetError(RuntimeError):
    """The isolated Parakeet worker could not complete a request."""


class ParakeetCancelledError(ParakeetError):
    """A caller requested that the local Parakeet worker stop."""


def parakeet_python_path() -> Path:
    """Return the reviewed isolated interpreter used for NVIDIA Parakeet."""

    candidate = Path(__file__).with_name(".parakeet-venv") / "Scripts" / "python.exe"
    if not candidate.is_file():
        raise ParakeetError(
            "NVIDIA Parakeet is not installed. Expected the isolated interpreter at "
            f"{candidate}. Run the reviewed Parakeet setup before selecting it."
        )
    return candidate


def _probe_duration_seconds(source: Path) -> Optional[float]:
    """Best-effort media duration probe for the pre-flight length check.

    Reuses ``transcribe.get_media_duration`` -- the same primitive
    ``archive_pipeline.py``'s ``probe_audio_duration_seconds`` calls to backfill
    ``audio_duration_seconds`` for the STT coverage check in ``stt_coverage.py``
    -- so this module does not grow a second duration probe. Any failure
    (missing ffprobe, unreadable container, ...) is treated as "unknown" rather
    than fatal; the pre-flight check simply does not run in that case.
    """

    try:
        from transcribe import get_media_duration

        return finite_seconds(get_media_duration(str(source)), positive=True)
    except Exception:
        return None


class ParakeetSession:
    """One serial GPU session that exposes a safe request/response interface."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_PARAKEET_MODEL,
        device: str = "cuda",
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.log = log or (lambda _message: None)
        self.process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._sequence = 0
        self._output: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None

    def __enter__(self) -> "ParakeetSession":
        self.start()
        return self

    def __exit__(self, *_unused: Any) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        interpreter = parakeet_python_path()
        worker = Path(__file__).with_name("parakeet_worker.py")
        if not worker.is_file():
            raise ParakeetError(f"Parakeet worker module is missing: {worker}")
        command = [str(interpreter), "-u", str(worker), "--serve", "--model", self.model, "--device", self.device]
        self.log("Starting local NVIDIA Parakeet worker (model loads once for this batch)…\n")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(worker.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise ParakeetError(f"Could not start NVIDIA Parakeet: {exc}") from exc
        self._output = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_worker_output,
            args=(self.process,),
            daemon=True,
            name="parakeet-worker-output",
        )
        self._reader.start()
        response = self._read_response(
            expected_id="ready", timeout=WORKER_READY_TIMEOUT_SECONDS, cancel_check=None
        )
        if response.get("status") != "ready":
            self.close()
            raise ParakeetError(str(response.get("error") or "Parakeet did not become ready"))
        self.log(
            f"Parakeet ready: {response.get('model', self.model)} on "
            f"{response.get('device', self.device)}.\n"
        )

    def _read_worker_output(self, process: subprocess.Popen[str]) -> None:
        """Move blocking pipe reads off the cancellation-aware request thread."""

        stream = process.stdout
        if stream is None:
            self._output.put(None)
            return
        try:
            for line in stream:
                self._output.put(line)
        finally:
            self._output.put(None)

    def transcribe(
        self,
        source: Path,
        *,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        """Transcribe one source and return the archive STT details contract."""

        source = Path(source)
        duration = _probe_duration_seconds(source)
        if duration is not None and duration < MIN_TRANSCRIBABLE_SECONDS:
            raise ParakeetError(f"audio too short to transcribe: {duration:.2f}s")

        with self._lock:
            for attempt in range(WORKER_CRASH_RETRIES + 1):
                try:
                    self.start()
                    assert self.process is not None and self.process.stdin is not None
                    self._sequence += 1
                    request_id = f"job-{self._sequence}"
                    request = {
                        "id": request_id,
                        "op": "transcribe",
                        "source": str(Path(source).resolve()),
                    }
                    try:
                        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                        self.process.stdin.flush()
                    except (BrokenPipeError, OSError) as exc:
                        self.close()
                        raise ParakeetError(
                            f"Parakeet worker stopped before accepting {source.name}: {exc}"
                        ) from exc
                    response = self._read_response(
                        expected_id=request_id,
                        timeout=TRANSCRIBE_TIMEOUT_SECONDS,
                        cancel_check=cancel_check,
                    )
                    if response.get("status") != "ok":
                        raise ParakeetError(
                            str(response.get("error") or "Parakeet transcription failed")
                        )
                    details = response.get("details")
                    if not isinstance(details, dict):
                        raise ParakeetError("Parakeet worker returned an invalid transcript payload")
                    return details
                except ParakeetCancelledError:
                    raise
                except ParakeetError as exc:
                    if attempt >= WORKER_CRASH_RETRIES or not self._retryable_worker_error(exc):
                        raise
                    self.log(
                        "Parakeet worker stopped unexpectedly; restarting once and retrying "
                        f"{source.name} from the beginning.\n"
                    )
                    self.close(force=True)

        raise AssertionError("Parakeet retry loop ended unexpectedly")

    @staticmethod
    def _retryable_worker_error(error: ParakeetError) -> bool:
        message = str(error).casefold()
        return any(
            marker in message
            for marker in (
                "worker exited unexpectedly",
                "stopped before accepting",
                "no readable output stream",
                "cuda",
            )
        )

    def _read_response(
        self,
        *,
        expected_id: str,
        timeout: float,
        cancel_check: Optional[Callable[[], bool]],
    ) -> dict[str, Any]:
        process = self.process
        if process is None:
            raise ParakeetError("Parakeet worker has no readable output stream")
        deadline = time.monotonic() + timeout
        while True:
            if cancel_check is not None and cancel_check():
                self.close(force=True)
                raise ParakeetCancelledError("Parakeet transcription cancelled")
            if time.monotonic() >= deadline:
                self.close(force=True)
                raise ParakeetError("Parakeet worker timed out; it was stopped safely")
            try:
                line = self._output.get(timeout=0.25)
            except queue.Empty:
                # On Windows a native CUDA abort can close stdout a fraction
                # before ``Popen.poll()`` reports the final exit code.  The
                # reader has already delivered its single EOF marker by then;
                # without checking again here the parent waits until the full
                # transcription timeout rather than taking the one safe retry.
                return_code = process.poll()
                if return_code is not None:
                    self.close()
                    raise ParakeetError(
                        f"Parakeet worker exited unexpectedly (code {return_code})"
                    )
                continue
            if line is None:
                return_code = process.poll()
                if return_code is not None:
                    self.close()
                    raise ParakeetError(f"Parakeet worker exited unexpectedly (code {return_code})")
                continue
            value = line.strip()
            if not value.startswith(PROTOCOL_MARKER):
                if value:
                    self.log(f"[Parakeet] {value}\n")
                continue
            try:
                response = json.loads(value[len(PROTOCOL_MARKER):])
            except ValueError:
                self.log(f"[Parakeet] malformed worker response ignored: {value[:200]}\n")
                continue
            if response.get("id") == expected_id:
                return response

    def close(self, *, force: bool = False) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if not force and process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"id": "shutdown", "op": "shutdown"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=20)
            elif process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
        finally:
            for stream in (process.stdin,):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass


__all__ = [
    "DEFAULT_PARAKEET_MODEL",
    "MIN_TRANSCRIBABLE_SECONDS",
    "ParakeetCancelledError",
    "ParakeetError",
    "ParakeetSession",
    "parakeet_python_path",
]
