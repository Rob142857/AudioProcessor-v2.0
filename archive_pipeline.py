"""Resumable archive transcription -> conservative cleanup -> DOCX pipeline.

This module is intentionally orchestration-only.  The existing transcription
engine remains the local STT backend, while cleanup is delegated to the
protected research service.  Raw STT artifacts are immutable inputs to later
stages so a cleanup or rendering failure never requires another transcription.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sqlite3
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


PIPELINE_VERSION = "3.0.0"
DEFAULT_CLEANUP_ENDPOINT = (
    "https://pg.objectiveartefacts.com.au/api/tooling/cleanup-chunk"
)
DEFAULT_CLEANUP_MODEL = "@cf/zai-org/glm-4.7-flash"
SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {
        ".aac",
        ".aif",
        ".aiff",
        ".avi",
        ".flac",
        ".flv",
        ".m4a",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".ogg",
        ".3gp",
        ".wav",
        ".webm",
        ".wma",
        ".wmv",
    }
)
FINAL_STATUSES = frozenset({"verified", "needs_review"})

# Values which can materially change recognition, preprocessing, or prompt
# bias.  They form part of the per-source STT request signature so a changed
# environment cannot silently reuse an older transcript.
STT_CONTENT_ENV_DEFAULTS = {
    "TRANSCRIBE_VERBATIM": "1",
    "TRANSCRIBE_QUALITY_MODE": "1",
    "TRANSCRIBE_PREPROCESS": "1",
    "TRANSCRIBE_PREPROC_MODE": "vintage_tape",
    "TRANSCRIBE_PREPROC_STRONG_FILTERS": "0",
    "TRANSCRIBE_SKIP_PREPROCESS": "",
    "TRANSCRIBE_VAD": "0",
    "TRANSCRIBE_DISABLE_VAD": "",
    "TRANSCRIBE_ALLOW_PROMPT": "1",
    "TRANSCRIBE_FORCE_NATIVE_WHISPER": "0",
    "TRANSCRIBE_FW_COMPUTE_TYPES": "",
    "TRANSCRIBE_FORCE_FP16": "0",
    "TRANSCRIBE_FALLBACK_ALLOW_PROMPT": "0",
    "TRANSCRIBE_FW_RETRY2": "0",
}
STT_RUNTIME_PACKAGES = (
    "faster-whisper",
    "ctranslate2",
    "openai-whisper",
    "torch",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def file_hash_matches(path: Path, expected: Any) -> bool:
    """Return false, rather than trusting existence, for stale/corrupt artifacts."""
    if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
        return False
    try:
        return sha256_file(path) == expected
    except OSError:
        return False


def installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def cleanup_record_summary(value: dict[str, Any]) -> dict[str, Any]:
    """Keep transcript/chunk bodies out of the small per-job manifest."""
    return {
        key: item
        for key, item in value.items()
        if key not in {"text", "chunks", "chunk_results"}
    }


def quick_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fingerprints_match(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return (
        first.get("size") == second.get("size")
        and first.get("mtime_ns") == second.get("mtime_ns")
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_audio(input_path: Path, output_root: Path) -> list[Path]:
    """Return deterministic, collision-preserving audio discovery results."""
    input_path = input_path.resolve()
    output_root = output_root.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio/video extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    files: list[Path] = []
    for candidate in input_path.rglob("*"):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if _is_relative_to(resolved, output_root):
            continue
        if candidate.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            files.append(resolved)
    return sorted(files, key=lambda item: str(item).casefold())


def source_relative_path(source: Path, input_path: Path) -> Path:
    root = input_path if input_path.is_dir() else input_path.parent
    return source.resolve().relative_to(root.resolve())


def artifact_directory(source: Path, input_path: Path, output_root: Path) -> Path:
    relative = source_relative_path(source, input_path)
    extension = relative.suffix.lower().lstrip(".") or "audio"
    # Including the original extension prevents lecture.mp3 and lecture.flac
    # from ever sharing a transcript destination.
    return output_root / relative.parent / f"{relative.stem}__{extension}"


def artifact_paths(job_directory: Path) -> dict[str, Path]:
    return {
        "manifest": job_directory / "manifest.json",
        "events": job_directory / "run.jsonl",
        "raw_text": job_directory / "raw.txt",
        "formatted_stt": job_directory / "stt.formatted.txt",
        "segments": job_directory / "raw.segments.json",
        "vtt": job_directory / "raw.vtt",
        "srt": job_directory / "raw.srt",
        "clean_text": job_directory / "cleaned.txt",
        "cleanup": job_directory / "cleanup.json",
        "cleanup_chunks": job_directory / "cleanup-chunks",
        "qa": job_directory / "qa.json",
        "docx": job_directory / "final.docx",
    }


def append_event(path: Path, event: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": utc_now(), "event": event, **details}
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(stable_json(record) + "\n")
        output.flush()


def _seconds(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def normalize_segments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            segment = dict(item)
        else:
            segment = {
                name: getattr(item, name)
                for name in ("id", "start", "end", "text", "words")
                if hasattr(item, name)
            }
        segment["start"] = _seconds(segment.get("start"))
        segment["end"] = _seconds(segment.get("end"))
        segment["text"] = str(segment.get("text") or "").strip()
        words = segment.get("words")
        if isinstance(words, list):
            serial_words = []
            for word in words:
                if isinstance(word, dict):
                    serial_words.append(dict(word))
                else:
                    serial_words.append(
                        {
                            key: getattr(word, key)
                            for key in ("start", "end", "word", "probability")
                            if hasattr(word, key)
                        }
                    )
            segment["words"] = serial_words
        normalized.append(segment)
    return normalized


def _timestamp(seconds: float, separator: str) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def segments_to_vtt(segments: Iterable[dict[str, Any]]) -> str:
    blocks = ["WEBVTT", ""]
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        blocks.extend(
            [
                f"{_timestamp(_seconds(segment.get('start')), '.')} --> "
                f"{_timestamp(_seconds(segment.get('end')), '.')}",
                text,
                "",
            ]
        )
    return "\n".join(blocks).rstrip() + "\n"


def segments_to_srt(segments: Iterable[dict[str, Any]]) -> str:
    blocks: list[str] = []
    index = 1
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        blocks.extend(
            [
                str(index),
                f"{_timestamp(_seconds(segment.get('start')), ',')} --> "
                f"{_timestamp(_seconds(segment.get('end')), ',')}",
                text,
                "",
            ]
        )
        index += 1
    return "\n".join(blocks).rstrip() + "\n"


def validate_artifacts(
    raw_text: str,
    cleaned_text: str,
    segments: list[dict[str, Any]],
    cleanup_needs_review: bool,
    *,
    requested_stt_model: Optional[str] = None,
    actual_stt_model: Optional[str] = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    raw_words = len(raw_text.split())
    cleaned_words = len(cleaned_text.split())
    if not raw_text.strip() or raw_words < 2:
        reasons.append("raw transcription is empty or implausibly short")
    lowered = raw_text.strip().casefold()
    if "no speech detected or transcription failed" in lowered:
        reasons.append("raw transcription contains a failure placeholder")
    if not cleaned_text.strip():
        reasons.append("cleaned transcript is empty")
    if raw_words:
        ratio = cleaned_words / raw_words
        if ratio < 0.85 or ratio > 1.15:
            reasons.append(f"cleaned/raw word ratio is {ratio:.3f}")
    else:
        ratio = 0.0

    previous_start = -1.0
    timestamp_errors = 0
    for segment in segments:
        start = _seconds(segment.get("start"))
        end = _seconds(segment.get("end"))
        if start < previous_start or end < start:
            timestamp_errors += 1
        previous_start = start
    if timestamp_errors:
        reasons.append(f"{timestamp_errors} non-monotonic/invalid timestamp segment(s)")
    if cleanup_needs_review:
        reasons.append("cleanup service marked one or more chunks for review")
    if requested_stt_model and actual_stt_model:
        requested = requested_stt_model.casefold()
        actual = actual_stt_model.casefold()
        if requested.startswith("faster-whisper-"):
            requested_name = requested.removeprefix("faster-whisper-")
            if "faster-whisper" not in actual or requested_name not in actual:
                reasons.append(
                    f"STT fell back from {requested_stt_model} to {actual_stt_model}"
                )
        elif requested not in actual:
            reasons.append(
                f"actual STT model {actual_stt_model} differs from {requested_stt_model}"
            )

    return {
        "status": "needs_review" if reasons else "passed",
        "reasons": reasons,
        "raw_words": raw_words,
        "cleaned_words": cleaned_words,
        "cleaned_to_raw_ratio": ratio,
        "segments": len(segments),
        "checked_at": utc_now(),
    }


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_root: Path
    stt_model: str = "faster-whisper-large-v3"
    cleanup_enabled: bool = True
    cleanup_endpoint: str = DEFAULT_CLEANUP_ENDPOINT
    cleanup_model: str = DEFAULT_CLEANUP_MODEL
    threads: Optional[int] = None
    force: bool = False
    cleanup_only: bool = False
    render_only: bool = False
    retry_review: bool = False
    dry_run: bool = False
    limit: Optional[int] = None

    @property
    def cleanup_signature(self) -> str:
        return sha256_text(
            stable_json(
                {
                    "pipeline": PIPELINE_VERSION,
                    "enabled": self.cleanup_enabled,
                    "endpoint": self.cleanup_endpoint,
                    "model": self.cleanup_model,
                    "profile": "verbatim-conservative",
                }
            )
        )

    @property
    def render_signature(self) -> str:
        return sha256_text(
            stable_json(
                {
                    "pipeline": PIPELINE_VERSION,
                    "format": "docx",
                    "australian_semantic_substitution": False,
                }
            )
        )


class JobIndex:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                source_path TEXT PRIMARY KEY,
                relative_path TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                source_size INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL,
                stt_model TEXT NOT NULL,
                cleanup_model TEXT,
                updated_at TEXT NOT NULL,
                error TEXT
            )
            """
        )
        self.connection.commit()

    def update(
        self,
        *,
        source: Path,
        relative: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
    ) -> None:
        fingerprint = manifest.get("source", {})
        cleanup = manifest.get("cleanup", {})
        self.connection.execute(
            """
            INSERT INTO jobs (
                source_path, relative_path, status, stage, manifest_path,
                source_size, source_mtime_ns, stt_model, cleanup_model,
                updated_at, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                relative_path=excluded.relative_path,
                status=excluded.status,
                stage=excluded.stage,
                manifest_path=excluded.manifest_path,
                source_size=excluded.source_size,
                source_mtime_ns=excluded.source_mtime_ns,
                stt_model=excluded.stt_model,
                cleanup_model=excluded.cleanup_model,
                updated_at=excluded.updated_at,
                error=excluded.error
            """,
            (
                str(source),
                relative.as_posix(),
                manifest.get("status", "unknown"),
                manifest.get("stage", "unknown"),
                str(manifest_path),
                int(fingerprint.get("size", 0)),
                int(fingerprint.get("mtime_ns", 0)),
                manifest.get("stt", {}).get("model", ""),
                cleanup.get("model"),
                utc_now(),
                manifest.get("error"),
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class PipelineRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self.index = JobIndex(self.config.output_root / "pipeline.sqlite3")
        self.cleanup_client: Any = None
        self.stt_runtime_versions = {
            package: installed_version(package) for package in STT_RUNTIME_PACKAGES
        }

    def close(self) -> None:
        self.index.close()

    def _get_cleanup_client(self) -> Any:
        if self.cleanup_client is None:
            try:
                from cleanup_client import CleanupClient
            except ImportError as exc:
                raise RuntimeError(
                    "cleanup_client.py is missing; cannot run the GLM cleanup stage"
                ) from exc
            self.cleanup_client = CleanupClient.from_environment(
                endpoint=self.config.cleanup_endpoint,
                model=self.config.cleanup_model,
            )
        return self.cleanup_client

    def _stt_request_signature(self, source: Path) -> str:
        """Fingerprint every known input which can affect the raw transcript."""
        prompt_files: list[dict[str, Any]] = []

        configured_prompt_file = os.environ.get(
            "TRANSCRIBE_AWKWARD_FILE", ""
        ).strip()
        if configured_prompt_file:
            candidate = Path(configured_prompt_file).expanduser()
            prompt_files.append(
                {
                    "kind": "configured",
                    "present": candidate.is_file(),
                    "sha256": sha256_file(candidate) if candidate.is_file() else None,
                }
            )

        for kind, directory in (
            ("source", source.parent),
            ("repository", Path(__file__).resolve().parent),
        ):
            for filename in ("special_words.txt", "special_words.md"):
                candidate = directory / filename
                if candidate.is_file():
                    prompt_files.append(
                        {
                            "kind": kind,
                            "filename": filename,
                            "sha256": sha256_file(candidate),
                        }
                    )
                    break

        content_environment = {
            key: os.environ.get(key, default)
            for key, default in STT_CONTENT_ENV_DEFAULTS.items()
        }
        prompt_values = {
            # Store only digests: user-supplied terminology need not be copied
            # into every manifest to make reuse deterministic.
            "awkward_terms_sha256": sha256_text(
                os.environ.get("TRANSCRIBE_AWKWARD_TERMS", "")
            ),
            "initial_prompt_sha256": sha256_text(
                os.environ.get("TRANSCRIBE_INITIAL_PROMPT", "")
            ),
            "files": prompt_files,
        }
        return sha256_text(
            stable_json(
                {
                    "pipeline": PIPELINE_VERSION,
                    "backend": "local",
                    "requested_model": self.config.stt_model,
                    "threads": self.config.threads,
                    "environment": content_environment,
                    "prompt": prompt_values,
                    "runtime": self.stt_runtime_versions,
                }
            )
        )

    def _base_manifest(
        self,
        source: Path,
        relative: Path,
        fingerprint: dict[str, Any],
        stt_request_signature: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "pipeline_version": PIPELINE_VERSION,
            "source": {
                "path": str(source),
                "relative_path": relative.as_posix(),
                **fingerprint,
            },
            "status": "pending",
            "stage": "discovered",
            "attempts": 0,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "stt": {
                "backend": "local",
                "model": self.config.stt_model,
                "request_signature": stt_request_signature,
                "signature": None,
            },
            "cleanup": {
                "enabled": self.config.cleanup_enabled,
                "endpoint": self.config.cleanup_endpoint,
                "model": self.config.cleanup_model,
                "signature": self.config.cleanup_signature,
            },
            "render": {"signature": self.config.render_signature},
            "artifacts": {},
        }

    def _save_manifest(
        self,
        source: Path,
        relative: Path,
        paths: dict[str, Path],
        manifest: dict[str, Any],
    ) -> None:
        manifest["updated_at"] = utc_now()
        atomic_write_json(paths["manifest"], manifest)
        self.index.update(
            source=source,
            relative=relative,
            manifest_path=paths["manifest"],
            manifest=manifest,
        )

    def _raw_is_reusable(
        self,
        manifest: dict[str, Any],
        fingerprint: dict[str, Any],
        paths: dict[str, Path],
        stt_request_signature: str,
    ) -> bool:
        stt = manifest.get("stt", {})
        return (
            (not self.config.force or self.config.cleanup_only or self.config.render_only)
            and fingerprints_match(manifest.get("source", {}), fingerprint)
            and (
                self.config.cleanup_only
                or self.config.render_only
                or stt.get("request_signature") == stt_request_signature
            )
            and file_hash_matches(paths["raw_text"], stt.get("raw_sha256"))
            and file_hash_matches(paths["segments"], stt.get("segments_sha256"))
        )

    def _clean_is_reusable(
        self,
        manifest: dict[str, Any],
        raw_sha256: str,
        paths: dict[str, Path],
        glossary_sha256: Optional[str],
        *,
        retrying_review: bool = False,
    ) -> bool:
        cleanup = manifest.get("cleanup", {})
        return (
            not self.config.force
            and not self.config.render_only
            and not self.config.cleanup_only
            and not retrying_review
            and cleanup.get("signature") == self.config.cleanup_signature
            and cleanup.get("glossary_sha256") == glossary_sha256
            and self._clean_artifacts_intact(manifest, raw_sha256, paths)
        )

    @staticmethod
    def _clean_artifacts_intact(
        manifest: dict[str, Any], raw_sha256: str, paths: dict[str, Path]
    ) -> bool:
        cleanup = manifest.get("cleanup", {})
        return (
            cleanup.get("input_sha256") == raw_sha256
            and file_hash_matches(
                paths["clean_text"], cleanup.get("output_sha256")
            )
            and file_hash_matches(
                paths["cleanup"], cleanup.get("record_sha256")
            )
        )

    def _render_is_reusable(
        self, manifest: dict[str, Any], clean_sha256: str, paths: dict[str, Path]
    ) -> bool:
        render = manifest.get("render", {})
        return (
            not self.config.force
            and not self.config.render_only
            and render.get("signature") == self.config.render_signature
            and render.get("input_sha256") == clean_sha256
            and file_hash_matches(paths["docx"], render.get("output_sha256"))
        )

    def _transcribe(self, source: Path) -> dict[str, Any]:
        os.environ["TRANSCRIBE_MODEL_NAME"] = self.config.stt_model
        os.environ.setdefault("TRANSCRIBE_VERBATIM", "1")
        os.environ.setdefault("TRANSCRIBE_ALLOW_PROMPT", "1")
        os.environ.setdefault("TRANSCRIBE_USE_DATASET", "0")
        from transcribe_optimised import transcribe_file_simple_auto

        result = transcribe_file_simple_auto(
            str(source),
            threads_override=self.config.threads,
            return_details=True,
            write_docx=False,
        )
        if not isinstance(result, dict):
            raise RuntimeError("transcription engine did not return structured details")
        return result

    def _render_docx(
        self,
        source: Path,
        text: str,
        output_path: Path,
        metadata: dict[str, Any],
    ) -> Path:
        from txt_to_docx import convert_txt_to_docx_from_text

        rendered = convert_txt_to_docx_from_text(
            text,
            source,
            metadata=metadata,
            use_australian_spelling=False,
            output_path=output_path,
        )
        rendered = Path(rendered)
        if not rendered.is_file() or rendered.stat().st_size < 1_000:
            raise RuntimeError(f"DOCX validation failed: {rendered}")
        return rendered

    def process_one(self, source: Path) -> str:
        relative = source_relative_path(source, self.config.input_path)
        job_directory = artifact_directory(
            source, self.config.input_path, self.config.output_root
        )
        paths = artifact_paths(job_directory)
        fingerprint = quick_fingerprint(source)
        stt_request_signature = self._stt_request_signature(source)
        manifest = read_json(paths["manifest"])
        if not manifest:
            manifest = self._base_manifest(
                source, relative, fingerprint, stt_request_signature
            )

        glossary_sha256: Optional[str] = None
        glossary_error: Optional[Exception] = None
        if self.config.cleanup_enabled and not self.config.render_only:
            try:
                glossary_sha256 = self._get_cleanup_client().ensure_glossary().sha256
            except Exception as exc:
                # Record this as a normal job failure below rather than losing
                # provenance before a manifest exists.
                glossary_error = exc

        final_candidate = (
            not self.config.force
            and not self.config.cleanup_only
            and not self.config.render_only
            and glossary_error is None
            and manifest.get("status") in FINAL_STATUSES
            and not (
                self.config.retry_review
                and manifest.get("status") == "needs_review"
            )
        )
        if final_candidate and self._raw_is_reusable(
            manifest, fingerprint, paths, stt_request_signature
        ):
            stt = manifest.get("stt", {})
            timestamp_artifacts_valid = (
                file_hash_matches(paths["vtt"], stt.get("vtt_sha256"))
                and file_hash_matches(paths["srt"], stt.get("srt_sha256"))
            )
            raw_text_for_reuse = paths["raw_text"].read_text(encoding="utf-8")
            raw_sha256_for_reuse = sha256_text(raw_text_for_reuse)
            if (
                timestamp_artifacts_valid
                and self._clean_is_reusable(
                    manifest,
                    raw_sha256_for_reuse,
                    paths,
                    glossary_sha256,
                )
            ):
                clean_text_for_reuse = paths["clean_text"].read_text(
                    encoding="utf-8"
                )
                if self._render_is_reusable(
                    manifest,
                    sha256_text(clean_text_for_reuse),
                    paths,
                ):
                    return (
                        "needs_review"
                        if manifest.get("status") == "needs_review"
                        else "skipped"
                    )

        if self.config.dry_run:
            return "queued"

        retrying_review = bool(
            self.config.retry_review and manifest.get("status") == "needs_review"
        )
        job_directory.mkdir(parents=True, exist_ok=True)
        manifest["attempts"] = int(manifest.get("attempts", 0)) + 1
        manifest["status"] = "running"
        manifest["error"] = None
        self._save_manifest(source, relative, paths, manifest)
        append_event(paths["events"], "job_started", attempt=manifest["attempts"])

        try:
            if glossary_error is not None:
                raise glossary_error
            if self._raw_is_reusable(
                manifest, fingerprint, paths, stt_request_signature
            ):
                raw_text = paths["raw_text"].read_text(encoding="utf-8")
                segments_value = json.loads(paths["segments"].read_text(encoding="utf-8"))
                segments = normalize_segments(segments_value)
                expected_vtt = segments_to_vtt(segments)
                expected_srt = segments_to_srt(segments)
                if not file_hash_matches(
                    paths["vtt"], manifest.get("stt", {}).get("vtt_sha256")
                ):
                    atomic_write_text(paths["vtt"], expected_vtt)
                if not file_hash_matches(
                    paths["srt"], manifest.get("stt", {}).get("srt_sha256")
                ):
                    atomic_write_text(paths["srt"], expected_srt)
                manifest["stt"]["vtt_sha256"] = sha256_file(paths["vtt"])
                manifest["stt"]["srt_sha256"] = sha256_file(paths["srt"])
                append_event(paths["events"], "raw_reused")
            else:
                if self.config.cleanup_only or self.config.render_only:
                    raise RuntimeError(
                        "raw artifacts are unavailable or stale; cleanup/render-only cannot continue"
                    )
                manifest["stage"] = "transcribing"
                self._save_manifest(source, relative, paths, manifest)
                append_event(paths["events"], "transcription_started")
                details = self._transcribe(source)
                # Preserve and clean the closest available representation of the
                # model output.  The engine's `text` value may already contain
                # derived paragraph formatting, so it is stored separately.
                raw_value = details.get("raw_text")
                if raw_value is None:
                    raw_value = details.get("text")
                raw_text = str(raw_value or "")
                formatted_value = details.get("text")
                formatted_stt = (
                    str(formatted_value)
                    if formatted_value is not None
                    else raw_text
                )
                if not raw_text.strip():
                    raise RuntimeError("speech-to-text returned no transcript")
                segments = normalize_segments(details.get("segments"))
                # Persist the exact stage output. Adding convenience newlines here
                # would change its content hash after a restart and invalidate an
                # otherwise reusable cleanup checkpoint.
                atomic_write_text(paths["raw_text"], raw_text)
                if formatted_stt and formatted_stt != raw_text:
                    atomic_write_text(paths["formatted_stt"], formatted_stt)
                atomic_write_json(paths["segments"], segments)
                atomic_write_text(paths["vtt"], segments_to_vtt(segments))
                atomic_write_text(paths["srt"], segments_to_srt(segments))
                manifest["source"] = {
                    "path": str(source),
                    "relative_path": relative.as_posix(),
                    **fingerprint,
                    "sha256": sha256_file(source),
                }
                details_metadata = details.get("metadata", {})
                if not isinstance(details_metadata, dict):
                    details_metadata = {}
                actual_model = str(
                    details_metadata.get("model") or self.config.stt_model
                )
                actual_signature = sha256_text(
                    stable_json(
                        {
                            "request_signature": stt_request_signature,
                            "actual_model": actual_model,
                            "device": details_metadata.get("device"),
                        }
                    )
                )
                manifest["stt"] = {
                    **manifest.get("stt", {}),
                    "request_signature": stt_request_signature,
                    "signature": actual_signature,
                    "requested_model": self.config.stt_model,
                    "actual_model": actual_model,
                    "model": actual_model,
                    "metadata": details_metadata,
                    "elapsed_seconds": details.get("elapsed_seconds"),
                    "raw_sha256": sha256_file(paths["raw_text"]),
                    "segments_sha256": sha256_file(paths["segments"]),
                    "vtt_sha256": sha256_file(paths["vtt"]),
                    "srt_sha256": sha256_file(paths["srt"]),
                    "formatted_stt_sha256": (
                        sha256_file(paths["formatted_stt"])
                        if paths["formatted_stt"].is_file()
                        else None
                    ),
                }
                manifest["artifacts"].update(
                    {
                        "raw_text": str(paths["raw_text"]),
                        "formatted_stt": (
                            str(paths["formatted_stt"])
                            if paths["formatted_stt"].is_file()
                            else None
                        ),
                        "segments": str(paths["segments"]),
                        "vtt": str(paths["vtt"]),
                        "srt": str(paths["srt"]),
                    }
                )
                manifest["stage"] = "raw_complete"
                self._save_manifest(source, relative, paths, manifest)
                append_event(
                    paths["events"],
                    "transcription_completed",
                    words=len(raw_text.split()),
                    segments=len(segments),
                )

            raw_sha256 = sha256_text(raw_text)
            cleanup_needs_review = False
            cleanup_metadata: dict[str, Any]
            if not self.config.cleanup_enabled:
                cleaned_text = raw_text
                cleanup_metadata = {
                    "enabled": False,
                    "model": None,
                    "input_sha256": raw_sha256,
                    "output_sha256": raw_sha256,
                    "signature": self.config.cleanup_signature,
                    "needs_review": False,
                    "glossary_sha256": None,
                    "glossary_count": 0,
                    "chunk_count": 0,
                    "warnings": [],
                }
            elif self.config.render_only:
                if not self._clean_artifacts_intact(manifest, raw_sha256, paths):
                    raise RuntimeError(
                        "cleaned artifacts are unavailable, stale, or corrupt; "
                        "render-only cannot continue"
                    )
                cleaned_text = paths["clean_text"].read_text(encoding="utf-8")
                cleanup_metadata = cleanup_record_summary(read_json(paths["cleanup"]))
                cleanup_metadata["record_sha256"] = sha256_file(paths["cleanup"])
                cleanup_needs_review = bool(cleanup_metadata.get("needs_review"))
            elif self._clean_is_reusable(
                manifest,
                raw_sha256,
                paths,
                glossary_sha256,
                retrying_review=retrying_review,
            ):
                cleaned_text = paths["clean_text"].read_text(encoding="utf-8")
                cleanup_metadata = cleanup_record_summary(read_json(paths["cleanup"]))
                cleanup_metadata["record_sha256"] = sha256_file(paths["cleanup"])
                cleanup_needs_review = bool(cleanup_metadata.get("needs_review"))
                append_event(paths["events"], "cleanup_reused")
            else:
                manifest["stage"] = "cleaning"
                self._save_manifest(source, relative, paths, manifest)
                append_event(paths["events"], "cleanup_started")
                cleanup_result = self._get_cleanup_client().cleanup_text(
                    raw_text,
                    checkpoint_dir=paths["cleanup_chunks"],
                    # These flags explicitly ask for a fresh cleanup pass.
                    # Normal restarts still reuse completed per-chunk calls.
                    reuse_checkpoints=not (
                        self.config.cleanup_only
                        or self.config.retry_review
                        or self.config.force
                    ),
                )
                cleaned_text = str(cleanup_result.text)
                if not cleaned_text.strip():
                    raise RuntimeError("cleanup service returned no transcript")
                result_payload = cleanup_result.to_dict()
                chunk_payloads = result_payload.get("chunks", [])
                grounding_counts = [
                    int(chunk["grounding"]["glossary_terms_considered"])
                    for chunk in chunk_payloads
                    if isinstance(chunk, dict)
                    and isinstance(chunk.get("grounding"), dict)
                    and isinstance(
                        chunk["grounding"].get("glossary_terms_considered"), int
                    )
                ]
                cleanup_metadata = {
                    "model": cleanup_result.model,
                    "glossary_sha256": cleanup_result.glossary_sha256,
                    "glossary_count": cleanup_result.glossary_count,
                    "chunk_count": len(cleanup_result.chunks),
                    "needs_review": bool(cleanup_result.needs_review),
                    "warnings": list(cleanup_result.warnings),
                    "grounding_glossary_terms_min": (
                        min(grounding_counts) if grounding_counts else None
                    ),
                    "grounding_glossary_terms_max": (
                        max(grounding_counts) if grounding_counts else None
                    ),
                }
                cleanup_metadata.update(
                    {
                        "enabled": True,
                        "input_sha256": raw_sha256,
                        "output_sha256": sha256_text(cleaned_text),
                        "signature": self.config.cleanup_signature,
                    }
                )
                cleanup_needs_review = bool(cleanup_result.needs_review)
                if (
                    grounding_counts
                    and cleanup_result.glossary_count
                    and min(grounding_counts) < cleanup_result.glossary_count
                ):
                    cleanup_needs_review = True
                    cleanup_metadata["needs_review"] = True
                    cleanup_metadata["warnings"].append(
                        "server considered fewer glossary terms than the pinned editable overlay"
                    )
                atomic_write_text(paths["clean_text"], cleaned_text)
                atomic_write_json(
                    paths["cleanup"],
                    {**cleanup_metadata, "chunk_results": chunk_payloads},
                )
                append_event(
                    paths["events"],
                    "cleanup_completed",
                    model=cleanup_result.model,
                    chunks=len(cleanup_result.chunks),
                    needs_review=cleanup_needs_review,
                )

            if not self.config.cleanup_enabled:
                atomic_write_text(paths["clean_text"], cleaned_text)
                atomic_write_json(paths["cleanup"], cleanup_metadata)
            cleanup_metadata["record_sha256"] = sha256_file(paths["cleanup"])
            manifest["cleanup"] = cleanup_metadata
            manifest["artifacts"].update(
                {
                    "clean_text": str(paths["clean_text"]),
                    "cleanup": str(paths["cleanup"]),
                }
            )
            manifest["stage"] = "clean_complete"
            self._save_manifest(source, relative, paths, manifest)

            clean_sha256 = sha256_text(cleaned_text)
            if not self._render_is_reusable(manifest, clean_sha256, paths):
                manifest["stage"] = "rendering"
                self._save_manifest(source, relative, paths, manifest)
                append_event(paths["events"], "render_started")
                metadata = {
                    "model": (
                        f"{self.config.stt_model} -> {cleanup_metadata.get('model')}"
                        if cleanup_metadata.get("model")
                        else self.config.stt_model
                    ),
                    "device": "See manifest",
                    "time_taken": "See manifest",
                    "preprocessing": "See manifest",
                    "source_sha256": manifest.get("source", {}).get("sha256"),
                    "pipeline_version": PIPELINE_VERSION,
                }
                rendered = self._render_docx(source, cleaned_text, paths["docx"], metadata)
                manifest["render"] = {
                    "signature": self.config.render_signature,
                    "input_sha256": clean_sha256,
                    "output_path": str(rendered),
                    "output_sha256": sha256_file(rendered),
                }
                manifest["artifacts"]["docx"] = str(rendered)
                append_event(
                    paths["events"], "render_completed", artifact_path=str(rendered)
                )

            qa = validate_artifacts(
                raw_text,
                cleaned_text,
                segments,
                cleanup_needs_review,
                requested_stt_model=manifest.get("stt", {}).get("requested_model"),
                actual_stt_model=manifest.get("stt", {}).get("actual_model"),
            )
            atomic_write_json(paths["qa"], qa)
            manifest["qa"] = qa
            manifest["artifacts"]["qa"] = str(paths["qa"])
            manifest["status"] = (
                "needs_review" if qa["status"] == "needs_review" else "verified"
            )
            manifest["stage"] = manifest["status"]
            self._save_manifest(source, relative, paths, manifest)
            append_event(paths["events"], "job_finished", status=manifest["status"])
            return str(manifest["status"])
        except KeyboardInterrupt:
            manifest["status"] = "cancelled"
            manifest["stage"] = "cancelled"
            manifest["error"] = "cancelled by user"
            self._save_manifest(source, relative, paths, manifest)
            append_event(paths["events"], "job_cancelled")
            raise
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["stage"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest["traceback"] = traceback.format_exc(limit=20)
            self._save_manifest(source, relative, paths, manifest)
            append_event(
                paths["events"], "job_failed", error=manifest["error"]
            )
            return "failed"

    def run(self) -> dict[str, int]:
        files = discover_audio(self.config.input_path, self.config.output_root)
        if self.config.limit is not None:
            files = files[: max(0, self.config.limit)]
        counts = {
            "discovered": len(files),
            "queued": 0,
            "skipped": 0,
            "verified": 0,
            "needs_review": 0,
            "failed": 0,
            "cancelled": 0,
        }
        print(f"Discovered {len(files):,} supported recording(s).")
        if self.config.dry_run:
            counts["queued"] = len(files)
            self._write_summary(counts)
            print("Dry run only; no source recording or transcript was changed.")
            return counts
        if files and self.config.cleanup_enabled and not self.config.render_only:
            print("Validating protected cleanup access and pinning the glossary...")
            self._get_cleanup_client().ensure_glossary()
        for index, source in enumerate(files, 1):
            print(f"[{index:,}/{len(files):,}] {source}")
            try:
                status = self.process_one(source)
            except KeyboardInterrupt:
                counts["cancelled"] += 1
                print("Pipeline cancelled; completed checkpoints are preserved.")
                break
            counts[status] = counts.get(status, 0) + 1
            print(f"  -> {status}")
        self._write_summary(counts)
        return counts

    def _write_summary(self, counts: dict[str, int]) -> None:
        atomic_write_json(
            self.config.output_root / "last-run-summary.json",
            {
                "pipeline_version": PIPELINE_VERSION,
                "finished_at": utc_now(),
                "input": str(self.config.input_path),
                "output": str(self.config.output_root),
                "counts": counts,
            },
        )
        cursor = self.index.connection.execute(
            """
            SELECT relative_path, status, stage, stt_model, cleanup_model,
                   manifest_path, error, updated_at
            FROM jobs ORDER BY relative_path COLLATE NOCASE
            """
        )
        rows = cursor.fetchall()
        csv_path = self.config.output_root / "pipeline-status.csv"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{csv_path.name}.", suffix=".tmp", dir=str(csv_path.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(
                    [
                        "relative_path",
                        "status",
                        "stage",
                        "stt_model",
                        "cleanup_model",
                        "manifest_path",
                        "error",
                        "updated_at",
                    ]
                )
                writer.writerows(rows)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, csv_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def default_output_root(input_path: Path) -> Path:
    input_path = input_path.resolve()
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name} - Polished"
    return input_path.parent / "Polished Transcripts"


def choose_input_folder() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(title="Choose the audio archive to process")
        root.destroy()
        return Path(selected).resolve() if selected else None
    except Exception:
        return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-safe recursive audio transcription and GLM cleanup"
    )
    parser.add_argument("input", nargs="?", help="Audio file or archive folder")
    parser.add_argument("--output", help="Separate artifact/output root")
    parser.add_argument(
        "--model", default="faster-whisper-large-v3", help="Local STT model"
    )
    parser.add_argument("--threads", type=int, help="CPU thread limit")
    parser.add_argument(
        "--cleanup-endpoint", default=os.environ.get("PG_CLEANUP_ENDPOINT", DEFAULT_CLEANUP_ENDPOINT)
    )
    parser.add_argument(
        "--cleanup-model",
        default=os.environ.get("PG_CLEANUP_MODEL", DEFAULT_CLEANUP_MODEL),
        help="Pinned cleanup model (default: GLM-4.7-Flash)",
    )
    parser.add_argument("--no-cleanup", action="store_true", help="Skip remote GLM cleanup")
    parser.add_argument("--cleanup-only", action="store_true", help="Reuse raw artifacts; rerun cleanup/render")
    parser.add_argument("--render-only", action="store_true", help="Reuse raw/cleaned artifacts; rerender DOCX")
    parser.add_argument(
        "--retry-review",
        action="store_true",
        help="Retry cleanup for jobs previously marked needs_review",
    )
    parser.add_argument("--force", action="store_true", help="Re-run all requested stages")
    parser.add_argument("--dry-run", action="store_true", help="Discover and count without processing")
    parser.add_argument("--limit", type=int, help="Process only the first N files (for a trial run)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.cleanup_only and args.render_only:
        print("--cleanup-only and --render-only are mutually exclusive", file=sys.stderr)
        return 2
    input_path = Path(args.input).expanduser() if args.input else choose_input_folder()
    if input_path is None:
        print("No input selected.")
        return 2
    input_path = input_path.resolve()
    output_root = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_root(input_path)
    )
    config = PipelineConfig(
        input_path=input_path,
        output_root=output_root,
        stt_model=args.model,
        cleanup_enabled=not args.no_cleanup,
        cleanup_endpoint=args.cleanup_endpoint,
        cleanup_model=args.cleanup_model,
        threads=args.threads,
        force=args.force,
        cleanup_only=args.cleanup_only,
        render_only=args.render_only,
        retry_review=args.retry_review,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    runner = PipelineRunner(config)
    try:
        try:
            counts = runner.run()
        except Exception as exc:
            print(
                f"Pipeline preflight failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
    finally:
        runner.close()
    print(stable_json(counts))
    if counts.get("failed") or counts.get("cancelled"):
        return 1
    if counts.get("needs_review"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
