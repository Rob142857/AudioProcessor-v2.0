"""Plan a safe source-folder DOCX cleanup after an archive review pass.

This tool is deliberately separate from transcription.  Its default mode is
read-only: it preserves only source-side `` - GLM Review.docx`` documents that
are proven by a completed local Parakeet manifest in the polished workspace,
then lists every other DOCX in the selected archive.  A later explicit apply
moves the reviewed list into a recoverable quarantine under the polished
workspace; it never silently deletes a source file.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FINAL_STATUSES = frozenset({"verified", "needs_review"})
PARAKEET_MODEL_PREFIX = "nvidia/parakeet-"
PLAN_VERSION = 1


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path.resolve(strict=False))))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_parakeet_manifest(manifest: dict[str, Any]) -> bool:
    stt = manifest.get("stt")
    if not isinstance(stt, dict):
        return False
    values = [stt.get("actual_model"), stt.get("model"), stt.get("requested_model")]
    metadata = stt.get("metadata")
    if isinstance(metadata, dict):
        values.append(metadata.get("model"))
        if metadata.get("backend") == "nvidia-parakeet":
            return True
    return any(
        isinstance(value, str) and value.casefold().startswith(PARAKEET_MODEL_PREFIX)
        for value in values
    )


def _proven_final_review_target(
    manifest: dict[str, Any], *, archive_root: Path, polished_root: Path
) -> Path | None:
    """Return a source-side GLM review target only for a completed Parakeet job."""

    if manifest.get("status") not in FINAL_STATUSES or not _is_parakeet_manifest(manifest):
        return None
    cleanup = manifest.get("cleanup")
    render = manifest.get("render")
    source = manifest.get("source")
    if not isinstance(cleanup, dict) or not isinstance(render, dict) or not isinstance(source, dict):
        return None
    if not isinstance(cleanup.get("output_sha256"), str) or not isinstance(
        render.get("output_sha256"), str
    ):
        return None
    source_value = source.get("path")
    output_value = render.get("output_path")
    if not isinstance(source_value, str) or not isinstance(output_value, str):
        return None
    source_path = Path(source_value).resolve(strict=False)
    final_docx = Path(output_value).resolve(strict=False)
    if not _is_relative_to(source_path, archive_root) or not _is_relative_to(
        final_docx, polished_root
    ):
        return None
    if not final_docx.is_file() or final_docx.is_symlink():
        return None
    return source_path.with_name(f"{source_path.stem} - GLM Review.docx")


@dataclass(frozen=True)
class DocxCleanupPlan:
    archive_root: Path
    polished_root: Path
    kept_final_glm_docx: tuple[Path, ...]
    delete_candidates: tuple[Path, ...]
    manifest_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "archive_root": str(self.archive_root),
            "polished_root": str(self.polished_root),
            "manifest_count": self.manifest_count,
            "kept_final_glm_docx": [str(path) for path in self.kept_final_glm_docx],
            "delete_candidates": [str(path) for path in self.delete_candidates],
        }


def build_cleanup_plan(archive_root: Path, polished_root: Path) -> DocxCleanupPlan:
    """Build a read-only plan which keeps only completed Parakeet GLM reviews."""

    archive_root = Path(archive_root).resolve()
    polished_root = Path(polished_root).resolve()
    if not archive_root.is_dir():
        raise FileNotFoundError(f"Archive folder does not exist: {archive_root}")
    if not polished_root.is_dir():
        raise FileNotFoundError(f"Polished workspace does not exist: {polished_root}")
    if _is_relative_to(polished_root, archive_root) or _is_relative_to(archive_root, polished_root):
        raise ValueError("Archive folder and polished workspace must be separate, non-nested folders")

    kept: dict[str, Path] = {}
    manifest_count = 0
    for manifest_path in sorted(polished_root.rglob("manifest.json"), key=lambda item: str(item).casefold()):
        if manifest_path.is_symlink():
            continue
        manifest = _read_json(manifest_path)
        if manifest is None:
            continue
        manifest_count += 1
        target = _proven_final_review_target(
            manifest, archive_root=archive_root, polished_root=polished_root
        )
        if target is not None and target.is_file() and not target.is_symlink():
            kept[_path_key(target)] = target.resolve()

    candidates: list[Path] = []
    for path in archive_root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.suffix.casefold() != ".docx":
            continue
        resolved = path.resolve()
        if _path_key(resolved) not in kept:
            candidates.append(resolved)

    return DocxCleanupPlan(
        archive_root=archive_root,
        polished_root=polished_root,
        kept_final_glm_docx=tuple(sorted(kept.values(), key=lambda item: str(item).casefold())),
        delete_candidates=tuple(sorted(candidates, key=lambda item: str(item).casefold())),
        manifest_count=manifest_count,
    )


def write_plan(plan: DocxCleanupPlan, path: Path) -> Path:
    path = Path(path).resolve()
    if not _is_relative_to(path, plan.polished_root):
        raise ValueError("Cleanup plan must be written inside the polished workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def quarantine_candidates(plan: DocxCleanupPlan, *, expected_count: int) -> tuple[Path, ...]:
    """Move a reviewed plan into a recoverable workspace quarantine.

    ``expected_count`` is intentionally required to prevent a changed archive
    from deleting more documents than the user approved in the dry run.
    """

    if expected_count != len(plan.delete_candidates):
        raise ValueError(
            f"Refusing cleanup: reviewed count was {expected_count}, but this plan has "
            f"{len(plan.delete_candidates)} candidate(s)"
        )
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    quarantine_root = plan.polished_root / "docx-cleanup-quarantine" / run_id
    moved: list[Path] = []
    for source in plan.delete_candidates:
        source = source.resolve(strict=False)
        if not _is_relative_to(source, plan.archive_root):
            raise ValueError(f"Refusing to move a file outside the archive: {source}")
        if not source.is_file() or source.is_symlink() or source.suffix.casefold() != ".docx":
            raise FileNotFoundError(f"Cleanup candidate changed since the plan was made: {source}")
        target = quarantine_root / source.relative_to(plan.archive_root)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Quarantine target already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        moved.append(target)
    return tuple(moved)


def _default_polished_root(archive_root: Path) -> Path:
    return archive_root.parent / f"{archive_root.name} - Polished"


def _print_paths(label: str, paths: Iterable[Path]) -> None:
    values = tuple(paths)
    print(f"{label}: {len(values):,}")
    for path in values:
        print(f"  {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a cleanup of source DOCX files while preserving proven Parakeet GLM Review files"
    )
    parser.add_argument("archive", type=Path, help="Source recordings folder")
    parser.add_argument(
        "--polished-output",
        type=Path,
        help="Polished workspace folder (defaults to the archive sibling ending in ' - Polished')",
    )
    parser.add_argument("--write-plan", type=Path, help="Write the read-only plan JSON inside the polished workspace")
    parser.add_argument("--apply", action="store_true", help="Move the reviewed candidates into recoverable quarantine")
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Required with --apply; must match the dry-run candidate count exactly",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive_root = args.archive.resolve()
    polished_root = (args.polished_output or _default_polished_root(archive_root)).resolve()
    plan = build_cleanup_plan(archive_root, polished_root)
    print(f"Completed Parakeet manifests inspected: {plan.manifest_count:,}")
    _print_paths("Kept proven Parakeet GLM Review DOCX", plan.kept_final_glm_docx)
    _print_paths("Source DOCX candidates for cleanup", plan.delete_candidates)
    if args.write_plan:
        print(f"Plan written: {write_plan(plan, args.write_plan)}")
    if not args.apply:
        print("Dry run only. No source document was changed.")
        return 0
    if args.expected_count is None:
        raise ValueError("--apply requires --expected-count from the reviewed dry run")
    moved = quarantine_candidates(plan, expected_count=args.expected_count)
    print(f"Moved {len(moved):,} source DOCX file(s) into recoverable quarantine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
