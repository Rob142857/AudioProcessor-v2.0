"""One safe entry point for archive maintenance operations.

Every subcommand is a read-only dry run by default.  To change files, repeat
the same command with ``--confirm --expect N``, where ``N`` is the exact count
shown by the reviewed dry-run plan.  The underlying tools retain their own
additional integrity checks and never permanently delete source material.

The older scripts remain supported for existing runbooks and automation:

* ``requeue`` wraps ``reset_corrupted_transcripts``;
* ``quarantine`` wraps ``archive_older_transcripts``;
* ``replace-docx`` wraps ``legacy_docx_replace``;
* ``prepare-cleanup`` wraps ``prepare_docx_cleanup``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import archive_older_transcripts as older_transcripts
import legacy_docx_replace as legacy_replacement
import prepare_docx_cleanup as docx_cleanup
import reset_corrupted_transcripts as corrupted_transcripts


def _add_confirmation_flags(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--confirm",
        action="store_true",
        help="Apply this reviewed plan. Requires --expect N.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly print the plan only (the default).",
    )
    parser.add_argument(
        "--expect",
        type=int,
        metavar="N",
        help="Exact count shown by the immediately preceding dry run; required with --confirm.",
    )


def _require_confirmation(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.confirm:
        return -1
    if args.expect is None or args.expect < 0:
        parser.error("--confirm requires a non-negative --expect N from the reviewed dry run")
    return args.expect


def _print_requeue_plan(candidates: Sequence[corrupted_transcripts.Candidate], output_root: Path) -> None:
    if not candidates:
        print("Nothing to requeue: no jobs match the guarded truncation-recovery criteria.")
        return
    print(f"{len(candidates)} job(s) would be quarantined, then become eligible for a fresh run:")
    for candidate in candidates:
        evidence = (
            "source audio could not be probed"
            if candidate.silent_fraction is None
            else f"{candidate.silent_fraction * 100:.0f}% measured silent"
        )
        print(f"  {candidate.title or candidate.job_directory.name}: {candidate.gap_seconds:.0f}s gap; {evidence}")
        print(f"    {candidate.job_directory}")
        if candidate.legacy_docx is not None:
            print(f"    (also moving legacy DOCX: {candidate.legacy_docx})")
    print(f"Quarantine destination: {corrupted_transcripts.quarantine_root_for(output_root)}")
    print(f"Re-run with --confirm --expect {len(candidates)} to move them aside.")


def _run_requeue(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    candidates = corrupted_transcripts.find_candidates(
        args.output_root, min_gap_seconds=args.min_gap_seconds
    )
    expected = _require_confirmation(args, parser)
    if expected < 0:
        _print_requeue_plan(candidates, args.output_root)
        return 0
    moved = corrupted_transcripts.apply_reset(
        args.output_root, candidates, confirm=True, expected_count=expected
    )
    print(f"Quarantined {len(moved)} job(s). A normal pipeline run can now requeue them.")
    return 0


def _run_quarantine(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    moves = older_transcripts.plan_moves(args.source_root)
    expected = _require_confirmation(args, parser)
    if expected < 0:
        if not moves:
            print("Nothing to quarantine: no superseded source DOCX files found.")
            return 0
        print(f"{len(moves)} superseded DOCX file(s) would move to the recoverable older-transcripts tree:")
        for move in moves:
            print(f"  {move.source}\n    -> {move.destination}")
        print(f"Re-run with --confirm --expect {len(moves)} to move them.")
        return 0
    moved = older_transcripts.apply_moves(
        moves,
        confirm=True,
        expected_count=expected,
        replace_identical_destination=args.replace_identical_destination,
    )
    print(f"Moved {len(moved)} superseded DOCX file(s) into recoverable quarantine.")
    return 0


def _run_replace_docx(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    plan = legacy_replacement.plan_legacy_docx_replacements(
        args.generated_root, args.legacy_scope_root
    )
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    expected = _require_confirmation(args, parser)
    if expected < 0:
        print(f"Dry run only. Re-run with --confirm --expect {len(plan.items)} to apply this plan.")
        return 0
    replaced = legacy_replacement.apply_legacy_docx_replacements(
        plan,
        expected_scope_root=args.legacy_scope_root,
        backup_root=args.backup_root,
        confirm=True,
        expected_count=expected,
    )
    print(f"Replaced {len(replaced)} DOCX file(s); originals remain under {args.backup_root.resolve()}.")
    return 0


def _run_prepare_cleanup(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    archive_root = args.archive.resolve()
    polished_root = (args.polished_output or docx_cleanup.default_polished_root(archive_root)).resolve()
    plan = docx_cleanup.build_cleanup_plan(archive_root, polished_root)
    print(f"Completed Parakeet manifests inspected: {plan.manifest_count:,}")
    print(f"Kept proven Parakeet GLM Review DOCX: {len(plan.kept_final_glm_docx):,}")
    print(f"Source DOCX candidates for cleanup: {len(plan.delete_candidates):,}")
    for path in plan.delete_candidates:
        print(f"  {path}")
    if args.write_plan:
        print(f"Plan written: {docx_cleanup.write_plan(plan, args.write_plan)}")
    expected = _require_confirmation(args, parser)
    if expected < 0:
        print(f"Dry run only. Re-run with --confirm --expect {len(plan.delete_candidates)} to quarantine candidates.")
        return 0
    moved = docx_cleanup.quarantine_candidates(plan, expected_count=expected)
    print(f"Moved {len(moved):,} source DOCX file(s) into recoverable quarantine.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    requeue = subcommands.add_parser("requeue", help="Quarantine known truncated jobs so the pipeline can redo them")
    requeue.add_argument("output_root", type=Path, help="Pipeline polished/output root containing job manifests")
    requeue.add_argument("--min-gap-seconds", type=float, default=corrupted_transcripts.DEFAULT_MIN_GAP_SECONDS)
    _add_confirmation_flags(requeue)
    requeue.set_defaults(handler=_run_requeue)

    quarantine = subcommands.add_parser("quarantine", help="Move superseded source DOCX files aside after a GLM review exists")
    quarantine.add_argument("source_root", type=Path, help="Source recording archive root")
    quarantine.add_argument("--replace-identical-destination", action="store_true")
    _add_confirmation_flags(quarantine)
    quarantine.set_defaults(handler=_run_quarantine)

    replace_docx = subcommands.add_parser("replace-docx", help="Publish verified generated DOCX into a guarded legacy scope")
    replace_docx.add_argument("generated_root", type=Path)
    replace_docx.add_argument("legacy_scope_root", type=Path)
    replace_docx.add_argument("backup_root", type=Path)
    _add_confirmation_flags(replace_docx)
    replace_docx.set_defaults(handler=_run_replace_docx)

    cleanup = subcommands.add_parser("prepare-cleanup", help="Plan or quarantine non-final source DOCX after review")
    cleanup.add_argument("archive", type=Path, help="Source recordings folder")
    cleanup.add_argument("--polished-output", type=Path)
    cleanup.add_argument("--write-plan", type=Path, help="Write plan JSON inside polished output")
    _add_confirmation_flags(cleanup)
    cleanup.set_defaults(handler=_run_prepare_cleanup)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Reject an incomplete confirmation before planning.  Some planning modes
    # probe audio or inspect a very large archive, and there is no value in
    # doing that work when an apply run cannot possibly proceed safely.
    if args.confirm and (args.expect is None or args.expect < 0):
        parser.error("--confirm requires a non-negative --expect N from the reviewed dry run")
    handler: Callable[[argparse.Namespace, argparse.ArgumentParser], int] = args.handler
    return handler(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
