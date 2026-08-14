"""Small, durable filesystem primitives shared by AudioProcessor tools.

All writes in this module are replacement writes: a fully flushed temporary
file is moved into place only after it is complete.  The JSONL helper is an
append-only exception; a short O_EXCL sidecar lock makes one line plus fsync a
single serialised publication event across processes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, TypeVar


ATOMIC_REPLACE_MAX_ATTEMPTS = 5
ATOMIC_REPLACE_RETRY_BASE_DELAY_SECONDS = 0.1
_T = TypeVar("_T")


def windows_extended_path(path: str | Path) -> str:
    """Return an absolute extended Windows path, preserving UNC paths."""

    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def sha256_text(value: str, *, truncate: int | None = None) -> str:
    """Return the SHA-256 digest of UTF-8 text, optionally explicitly truncated."""

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return _truncate_digest(digest, truncate)


def sha256_file(
    path: str | Path,
    *,
    block_size: int = 4 * 1024 * 1024,
    truncate: int | None = None,
) -> str:
    """Return the SHA-256 digest of file bytes, including over MAX_PATH roots."""

    digest = hashlib.sha256()
    with Path(windows_extended_path(path)).open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return _truncate_digest(digest.hexdigest(), truncate)


def _truncate_digest(digest: str, truncate: int | None) -> str:
    if truncate is None:
        return digest
    if not isinstance(truncate, int) or not 1 <= truncate <= len(digest):
        raise ValueError("truncate must be an integer from 1 to the digest length")
    return digest[:truncate]


def is_retryable_replace_error(exc: OSError) -> bool:
    """Whether a transient Windows file lock should be retried."""

    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32)


def retry_with_backoff(
    operation: Callable[[], _T],
    *,
    retry_if: Callable[[OSError], bool] = is_retryable_replace_error,
    max_attempts: int = ATOMIC_REPLACE_MAX_ATTEMPTS,
    base_delay_seconds: float = ATOMIC_REPLACE_RETRY_BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Run a filesystem operation with bounded exponential backoff."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(max_attempts):
        try:
            return operation()
        except OSError as exc:
            if not retry_if(exc) or attempt == max_attempts - 1:
                raise
            sleep(base_delay_seconds * (2**attempt))
    raise AssertionError("unreachable")


def atomic_write_bytes(
    path: str | Path,
    value: bytes,
    *,
    max_attempts: int = ATOMIC_REPLACE_MAX_ATTEMPTS,
    base_delay_seconds: float = ATOMIC_REPLACE_RETRY_BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Durably replace ``path`` with ``value``, tolerating brief sync locks."""

    requested = Path(path)
    parent = Path(windows_extended_path(requested.parent))
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".write-", suffix=".tmp", dir=str(parent)
    )
    temporary = Path(temporary_name)
    destination = Path(windows_extended_path(requested))
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        retry_with_backoff(
            lambda: os.replace(temporary, destination),
            max_attempts=max_attempts,
            base_delay_seconds=base_delay_seconds,
            sleep=time.sleep if sleep is None else sleep,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, value: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, value.encode(encoding))


def atomic_write_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=indent) + "\n")


def append_jsonl_locked(
    path: str | Path,
    value: Any,
    *,
    max_attempts: int = 50,
    base_delay_seconds: float = 0.05,
) -> None:
    """Append one fsynced JSONL event while holding a cross-process sidecar lock.

    A publication event is encoded before the lock is acquired.  The lock
    serialises writers, while O_APPEND and one write make the data event a
    durable append rather than a read/modify/write race.
    """

    requested = Path(path)
    parent = Path(windows_extended_path(requested.parent))
    parent.mkdir(parents=True, exist_ok=True)
    destination = Path(windows_extended_path(requested))
    lock = Path(str(destination) + ".lock")
    encoded = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor: int | None = None
    lock_acquired = False
    try:
        for attempt in range(max_attempts):
            try:
                descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                lock_acquired = True
                break
            except FileExistsError:
                if attempt == max_attempts - 1:
                    raise TimeoutError(f"timed out waiting to append {requested}")
                time.sleep(base_delay_seconds * min(2**attempt, 8))
        assert descriptor is not None
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(str(os.getpid()).encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = None
        with destination.open("ab") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if lock_acquired:
            lock.unlink(missing_ok=True)
