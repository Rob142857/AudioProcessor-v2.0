"""Explicit, rollback-safe replacement of legacy transcript DOCX files.

The command is a two-step operation: inspect the plan first, then repeat with
``--confirm-replace --expect N``.  Targets are confined to one caller-supplied
scope root.  Every original is copied to a new backup tree before any target is
atomically replaced; backups are retained after success or failure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Optional
import zipfile

from stt_coverage import assess_stt_coverage, coverage_record_is_passed, finite_seconds


class ReplacementError(RuntimeError):
    """A replacement plan or transaction failed a safety condition."""


@dataclass(frozen=True)
class LegacyDocxReplacement:
    manifest: Path
    generated: Path
    source: Path
    target: Path
    target_relative: Path
    generated_sha256: str
    original_sha256: Optional[str]

    @property
    def operation(self) -> str:
        return "replace" if self.original_sha256 is not None else "create"

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest),
            "generated": str(self.generated),
            "source": str(self.source),
            "target": str(self.target),
            "target_relative": self.target_relative.as_posix(),
            "operation": self.operation,
            "generated_sha256": self.generated_sha256,
            "original_sha256": self.original_sha256,
        }


@dataclass(frozen=True)
class LegacyReplacementPlan:
    generated_root: Path
    scope_root: Path
    items: tuple[LegacyDocxReplacement, ...]

    @property
    def plan_sha256(self) -> str:
        encoded = json.dumps(
            {
                "generated_root": str(self.generated_root),
                "scope_root": str(self.scope_root),
                "items": [item.to_dict() for item in self.items],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_root": str(self.generated_root),
            "scope_root": str(self.scope_root),
            "count": len(self.items),
            "plan_sha256": self.plan_sha256,
            "items": [item.to_dict() for item in self.items],
        }


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _normal_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path.resolve())))


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((_normal_path(path), _normal_path(root))) == _normal_path(root)
    except (OSError, ValueError):
        return False


def _require_disjoint(first: Path, second: Path, labels: str) -> None:
    if _is_within(first, second) or _is_within(second, first):
        raise ReplacementError(f"{labels} must be separate, non-nested directories")


def validate_docx(path: Path) -> None:
    """Validate ZIP integrity and the minimum WordprocessingML members."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ReplacementError(f"DOCX is missing, not a file, or a symlink: {path}")
    try:
        with zipfile.ZipFile(path) as package:
            names = set(package.namelist())
            required = {"[Content_Types].xml", "word/document.xml"}
            missing = required - names
            if missing:
                raise ReplacementError(
                    f"DOCX package is missing {', '.join(sorted(missing))}: {path}"
                )
            damaged = package.testzip()
            if damaged:
                raise ReplacementError(f"DOCX has a damaged member {damaged}: {path}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReplacementError(f"Invalid DOCX package {path}: {exc}") from exc


def _safe_relative_source(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReplacementError("manifest source.relative_path is missing")
    relative = Path(value.replace("/", os.sep))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ReplacementError(f"unsafe source-relative path in manifest: {value!r}")
    return relative


def _require_publishable_cleanup(manifest: dict[str, Any], manifest_path: Path) -> None:
    """Require evidence that GLM cleanup used the complete pinned glossary."""

    cleanup = manifest.get("cleanup")
    if not isinstance(cleanup, dict):
        raise ReplacementError(f"manifest cleanup record is missing: {manifest_path}")
    if cleanup.get("enabled") is not True:
        raise ReplacementError(f"cleanup was not enabled: {manifest_path}")
    if cleanup.get("needs_review") is not False:
        raise ReplacementError(f"cleanup is unreviewed or needs review: {manifest_path}")
    model = cleanup.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ReplacementError(f"cleanup model is missing: {manifest_path}")
    glossary_sha256 = cleanup.get("glossary_sha256")
    if not isinstance(glossary_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", glossary_sha256
    ):
        raise ReplacementError(f"cleanup glossary SHA-256 is invalid: {manifest_path}")
    glossary_count = cleanup.get("glossary_count")
    if type(glossary_count) is not int or glossary_count <= 0:
        raise ReplacementError(f"cleanup glossary count is invalid: {manifest_path}")

    grounding_keys = (
        "grounding_glossary_terms_min",
        "grounding_glossary_terms_max",
    )
    grounding_values = [cleanup.get(key) for key in grounding_keys]
    if any(type(value) is not int for value in grounding_values):
        raise ReplacementError(
            f"cleanup grounding counts are incomplete or invalid: {manifest_path}"
        )
    if any(value < glossary_count for value in grounding_values):
        raise ReplacementError(
            f"cleanup did not ground every chunk with the full glossary: {manifest_path}"
        )


def _require_publishable_stt(
    manifest: dict[str, Any],
    manifest_path: Path,
    generated_root: Path,
) -> None:
    """Recheck timestamp coverage and its hashed segment artifact before publish."""

    qa = manifest.get("qa")
    coverage = qa.get("stt_coverage") if isinstance(qa, dict) else None
    if not coverage_record_is_passed(coverage):
        raise ReplacementError(
            f"manifest lacks passing STT coverage evidence: {manifest_path}"
        )

    stt = manifest.get("stt")
    if not isinstance(stt, dict):
        raise ReplacementError(f"manifest STT record is missing: {manifest_path}")
    metadata = stt.get("metadata")
    audio_duration = finite_seconds(
        metadata.get("audio_duration_seconds") if isinstance(metadata, dict) else None,
        positive=True,
    )
    if audio_duration is None:
        raise ReplacementError(
            f"manifest STT audio duration is missing or invalid: {manifest_path}"
        )

    artifacts = manifest.get("artifacts")
    segment_value = artifacts.get("segments") if isinstance(artifacts, dict) else None
    if not isinstance(segment_value, str) or not segment_value.strip():
        raise ReplacementError(f"manifest STT segment artifact is missing: {manifest_path}")
    segment_candidate = Path(segment_value)
    if not segment_candidate.is_absolute():
        segment_candidate = manifest_path.parent / segment_candidate
    if segment_candidate.is_symlink():
        raise ReplacementError(f"STT segment artifact is a symlink: {segment_candidate}")
    segment_path = segment_candidate.resolve()
    if not _is_within(segment_path, generated_root) or not segment_path.is_file():
        raise ReplacementError(
            f"STT segment artifact is missing or escapes its output root: {segment_path}"
        )

    expected_hash = stt.get("segments_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_hash
    ):
        raise ReplacementError(f"STT segment SHA-256 is invalid: {manifest_path}")
    if _sha256_file(segment_path).casefold() != expected_hash.casefold():
        raise ReplacementError(f"STT segment artifact hash mismatch: {segment_path}")

    try:
        segments = json.loads(segment_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplacementError(f"invalid STT segment artifact {segment_path}: {exc}") from exc
    assessment = assess_stt_coverage(segments, audio_duration)
    if assessment["status"] != "passed":
        detail = "; ".join(assessment["reasons"])
        raise ReplacementError(f"STT coverage is not publishable: {detail}: {manifest_path}")

    for key in ("segment_count", "text_segment_count"):
        if coverage.get(key) != assessment.get(key):
            raise ReplacementError(
                f"manifest STT coverage does not match its segment artifact: {manifest_path}"
            )
    for key in (
        "audio_duration_seconds",
        "last_segment_end_seconds",
        "trailing_silence_seconds",
        "trailing_silence_tolerance_seconds",
    ):
        recorded = finite_seconds(coverage.get(key), positive=key == "audio_duration_seconds")
        measured = finite_seconds(
            assessment.get(key), positive=key == "audio_duration_seconds"
        )
        if recorded is None or measured is None or abs(recorded - measured) > 0.001:
            raise ReplacementError(
                f"manifest STT coverage does not match its segment artifact: {manifest_path}"
            )


def plan_legacy_docx_replacements(
    generated_root: Path,
    legacy_root: Path,
    *,
    manifest_paths: Optional[Iterable[Path]] = None,
) -> LegacyReplacementPlan:
    """Build a read-only replacement plan from verified pipeline manifests."""

    generated_root = Path(generated_root).resolve()
    legacy_root = Path(legacy_root).resolve()
    if not generated_root.is_dir():
        raise ReplacementError(f"generated root is not a directory: {generated_root}")
    if not legacy_root.is_dir():
        raise ReplacementError(f"legacy scope root is not a directory: {legacy_root}")
    _require_disjoint(generated_root, legacy_root, "generated and legacy roots")

    if manifest_paths is None:
        manifests = sorted(
            generated_root.rglob("manifest.json"),
            key=lambda path: str(path).casefold(),
        )
    else:
        manifests = []
        seen_manifests: set[str] = set()
        for value in manifest_paths:
            candidate = Path(value)
            if candidate.name != "manifest.json" or candidate.is_symlink():
                raise ReplacementError(f"invalid selected manifest path: {candidate}")
            resolved = candidate.resolve()
            if not _is_within(resolved, generated_root):
                raise ReplacementError(
                    f"selected manifest escapes generated root: {resolved}"
                )
            key = os.path.normcase(str(resolved))
            if key in seen_manifests:
                raise ReplacementError(f"duplicate selected manifest: {resolved}")
            seen_manifests.add(key)
            manifests.append(resolved)
        manifests.sort(key=lambda path: str(path).casefold())
    if not manifests:
        raise ReplacementError(f"no pipeline manifests found under {generated_root}")

    items: list[LegacyDocxReplacement] = []
    seen_targets: dict[str, Path] = {}
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReplacementError(f"invalid manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ReplacementError(f"manifest is not a JSON object: {manifest_path}")
        if manifest.get("status") != "verified":
            raise ReplacementError(
                f"refusing non-verified transcript ({manifest.get('status')!r}): {manifest_path}"
            )
        qa = manifest.get("qa")
        if not isinstance(qa, dict) or qa.get("status") != "passed":
            raise ReplacementError(f"refusing transcript without passed QA: {manifest_path}")
        _require_publishable_stt(manifest, manifest_path, generated_root)
        _require_publishable_cleanup(manifest, manifest_path)

        source = manifest.get("source")
        if not isinstance(source, dict):
            raise ReplacementError(f"manifest source record is missing: {manifest_path}")
        source_relative = _safe_relative_source(source.get("relative_path"))
        legacy_source_candidate = legacy_root / source_relative
        if not _is_within(legacy_source_candidate, legacy_root):
            raise ReplacementError(f"source escapes the dedicated scope: {legacy_source_candidate}")
        if not legacy_source_candidate.is_file() or legacy_source_candidate.is_symlink():
            raise ReplacementError(
                f"source recording is missing, not a file, or a symlink: {legacy_source_candidate}"
            )
        legacy_source = legacy_source_candidate.resolve()

        target_relative = source_relative.with_suffix(".docx")
        target_candidate = legacy_root / target_relative
        if not _is_within(target_candidate, legacy_root):
            raise ReplacementError(f"target escapes the dedicated scope: {target_candidate}")
        if target_candidate.is_symlink():
            raise ReplacementError(f"legacy DOCX target is a symlink: {target_candidate}")
        if target_candidate.exists() and not target_candidate.is_file():
            raise ReplacementError(f"legacy DOCX target is not a file: {target_candidate}")
        target = target_candidate.resolve()

        artifacts = manifest.get("artifacts")
        generated_value = artifacts.get("docx") if isinstance(artifacts, dict) else None
        generated_candidate = (
            Path(generated_value)
            if isinstance(generated_value, str) and generated_value.strip()
            else manifest_path.parent / "final.docx"
        )
        if not generated_candidate.is_absolute():
            generated_candidate = manifest_path.parent / generated_candidate
        if generated_candidate.is_symlink():
            raise ReplacementError(f"generated DOCX is a symlink: {generated_candidate}")
        generated = generated_candidate.resolve()
        if not _is_within(generated, generated_root):
            raise ReplacementError(f"generated DOCX escapes its output root: {generated}")
        validate_docx(generated)

        target_key = _normal_path(target)
        if target_key in seen_targets:
            raise ReplacementError(
                "multiple source formats map to one legacy DOCX target: "
                f"{seen_targets[target_key]} and {manifest_path} -> {target}"
            )
        seen_targets[target_key] = manifest_path
        items.append(
            LegacyDocxReplacement(
                manifest=manifest_path.resolve(),
                generated=generated,
                source=legacy_source,
                target=target,
                target_relative=target_relative,
                generated_sha256=_sha256_file(generated),
                original_sha256=_sha256_file(target) if target.is_file() else None,
            )
        )

    return LegacyReplacementPlan(generated_root, legacy_root, tuple(items))


def _copy_to_new_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if destination.exists():
            raise ReplacementError(f"refusing to overwrite backup: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _stage_generated(source: Path, target: Path, *, validate: bool = True) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.replacement.", suffix=".tmp.docx", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if validate:
            validate_docx(temporary)
        return temporary
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _commit_stage(staged: Path, target: Path) -> None:
    os.replace(staged, target)


def _restore_backup(backup: Path, target: Path) -> None:
    # The original is restored byte-for-byte even when it was an old or damaged
    # DOCX package.  Replacement is not permission to discard source material.
    staged = _stage_generated(backup, target, validate=False)
    try:
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def apply_legacy_docx_replacements(
    plan: LegacyReplacementPlan,
    *,
    expected_scope_root: Path,
    backup_root: Path,
    confirm: bool = False,
    expected_count: Optional[int] = None,
) -> tuple[Path, ...]:
    """Apply an explicitly confirmed plan, retaining backups in all outcomes."""

    if not confirm:
        raise ReplacementError("replacement is dry-run only unless confirm=True")
    if expected_count is None or expected_count != len(plan.items):
        raise ReplacementError(
            f"expected_count must exactly match the planned {len(plan.items)} replacement(s)"
        )
    scope_root = Path(expected_scope_root).resolve()
    if _normal_path(scope_root) != _normal_path(plan.scope_root):
        raise ReplacementError("expected_scope_root does not exactly match the planned scope")
    backup_root = Path(backup_root).resolve()
    _require_disjoint(backup_root, plan.scope_root, "backup and legacy roots")
    _require_disjoint(backup_root, plan.generated_root, "backup and generated roots")
    if backup_root.exists():
        raise ReplacementError(f"backup root must be a new directory: {backup_root}")

    # Revalidate the whole batch before creating a backup directory or touching a target.
    for item in plan.items:
        if not _is_within(item.target, plan.scope_root):
            raise ReplacementError(f"target no longer lies within scope: {item.target}")
        if not item.source.is_file() or item.source.is_symlink():
            raise ReplacementError(f"source recording changed type or disappeared: {item.source}")
        if not _is_within(item.generated, plan.generated_root):
            raise ReplacementError(f"generated DOCX no longer lies within scope: {item.generated}")
        validate_docx(item.generated)
        if _sha256_file(item.generated) != item.generated_sha256:
            raise ReplacementError(f"generated DOCX changed after planning: {item.generated}")
        if item.original_sha256 is None:
            if item.target.exists() or item.target.is_symlink():
                raise ReplacementError(f"new publication target appeared after planning: {item.target}")
        else:
            if not item.target.is_file() or item.target.is_symlink():
                raise ReplacementError(f"legacy target changed type or disappeared: {item.target}")
            if _sha256_file(item.target) != item.original_sha256:
                raise ReplacementError(f"legacy target changed after planning: {item.target}")

    backup_root.mkdir(parents=True, exist_ok=False)
    backups: dict[Path, Path] = {}
    staged_files: dict[Path, Path] = {}
    committed: list[LegacyDocxReplacement] = []
    try:
        for item in plan.items:
            if item.original_sha256 is not None:
                backup = backup_root / item.target_relative
                _copy_to_new_path(item.target, backup)
                if _sha256_file(backup) != item.original_sha256:
                    raise ReplacementError(f"backup verification failed: {backup}")
                backups[item.target] = backup

        for item in plan.items:
            staged = _stage_generated(item.generated, item.target)
            if _sha256_file(staged) != item.generated_sha256:
                raise ReplacementError(f"staged replacement hash mismatch: {item.generated}")
            staged_files[item.target] = staged

        for item in plan.items:
            staged = staged_files[item.target]
            if item.original_sha256 is None:
                if item.target.exists() or item.target.is_symlink():
                    raise ReplacementError(
                        f"new publication target appeared before commit: {item.target}"
                    )
            else:
                if not item.target.is_file() or item.target.is_symlink():
                    raise ReplacementError(
                        f"legacy target changed type or disappeared before commit: {item.target}"
                    )
                if _sha256_file(item.target) != item.original_sha256:
                    raise ReplacementError(f"legacy target changed before commit: {item.target}")
            _commit_stage(staged, item.target)
            staged_files.pop(item.target, None)
            committed.append(item)
            validate_docx(item.target)
            if _sha256_file(item.target) != item.generated_sha256:
                raise ReplacementError(f"replacement verification failed: {item.target}")
    except BaseException as exc:
        rollback_errors: list[str] = []
        for item in reversed(committed):
            try:
                if item.original_sha256 is None:
                    if not item.target.is_file() or item.target.is_symlink():
                        raise ReplacementError("new target changed type before rollback")
                    if _sha256_file(item.target) != item.generated_sha256:
                        raise ReplacementError("new target changed after publication")
                    item.target.unlink()
                else:
                    backup = backups[item.target]
                    _restore_backup(backup, item.target)
                    if _sha256_file(item.target) != item.original_sha256:
                        raise ReplacementError("restored file hash does not match the original")
            except BaseException as rollback_exc:
                rollback_errors.append(f"{item.target}: {rollback_exc}")
        if rollback_errors:
            raise ReplacementError(
                f"replacement failed ({exc}); rollback also failed for "
                + "; ".join(rollback_errors)
                + f". Verified backups remain under {backup_root}"
            ) from exc
        raise ReplacementError(
            f"replacement failed and committed targets were restored; backups remain under {backup_root}: {exc}"
        ) from exc
    finally:
        for staged in staged_files.values():
            staged.unlink(missing_ok=True)

    return tuple(item.target for item in plan.items)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or explicitly apply verified legacy DOCX replacements"
    )
    parser.add_argument("generated_root", type=Path)
    parser.add_argument("legacy_scope_root", type=Path)
    parser.add_argument("backup_root", type=Path)
    parser.add_argument("--confirm-replace", action="store_true")
    parser.add_argument("--expect", type=int, help="Exact number shown by the dry-run plan")
    args = parser.parse_args(argv)

    plan = plan_legacy_docx_replacements(args.generated_root, args.legacy_scope_root)
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    if not args.confirm_replace:
        print("Dry run only. No legacy DOCX file was changed.")
        return 0
    replaced = apply_legacy_docx_replacements(
        plan,
        expected_scope_root=args.legacy_scope_root,
        backup_root=args.backup_root,
        confirm=True,
        expected_count=args.expect,
    )
    print(f"Replaced {len(replaced)} legacy DOCX file(s); originals retained under {args.backup_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LegacyDocxReplacement",
    "LegacyReplacementPlan",
    "ReplacementError",
    "apply_legacy_docx_replacements",
    "plan_legacy_docx_replacements",
    "validate_docx",
]
