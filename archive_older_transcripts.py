"""Move superseded DOCX transcripts aside once a GLM-reviewed copy exists.

Two-step, safety-first operation matching legacy_docx_replace.py's pattern:
inspect the plan first (the default), then repeat with ``--confirm-move
--expect N`` to actually move files.

For every audio file under a source root, if "<stem> - GLM Review.docx"
already exists next to it, every *other* DOCX belonging to that same
recording (the old/legacy transcript, and any other same-stem variant) is
moved -- never deleted -- to a parallel folder next to the source root named
"<source root name> - Older transcripts for review", preserving the source's
relative directory structure. A recording with no GLM Review copy yet is
left completely untouched: that is the safety gate, not an incidental check.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import NamedTuple

AUDIO_EXTENSIONS = frozenset(
    {
        ".aac",
        ".aiff",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".opus",
        ".wav",
        ".wma",
        ".wmv",
    }
)
GLM_REVIEW_SUFFIX = " - GLM Review.docx"


def _windows_extended_path(path: str | Path) -> str:
    """Return an extended Windows path when a name exceeds MAX_PATH.

    This archive's folder names (e.g. "Tapes From Joe (MW, RL, et al)")
    already push plenty of paths close to Windows' 260-char limit; moving
    files into a second, differently-named parallel root adds more length
    on top, not less. Same helper as cleanup_client.py / archive_pipeline.py
    / legacy_docx_replace.py, duplicated rather than imported.
    """

    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


class PlannedMove(NamedTuple):
    source: Path
    destination: Path


def plan_moves(source_root: Path) -> tuple[PlannedMove, ...]:
    """Read-only: decide what would move, touching nothing."""

    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise ValueError(f"source root is not a directory: {source_root}")
    dest_root = source_root.parent / f"{source_root.name} - Older transcripts for review"

    seen: dict[Path, Path] = {}  # destination -> source
    for audio_path in sorted(source_root.rglob("*")):
        if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        stem = audio_path.stem
        directory = audio_path.parent
        glm_review = directory / f"{stem}{GLM_REVIEW_SUFFIX}"
        if not glm_review.is_file():
            continue  # No reviewed replacement yet -- leave everything alone.

        for candidate in sorted(directory.glob("*.docx")):
            if candidate.name.endswith(GLM_REVIEW_SUFFIX):
                # Never sweep in a GLM Review file -- not even one that isn't
                # this recording's own keeper. Two distinct recordings can
                # share a directory where one's stem is a strict prefix of
                # the other's (e.g. "A" and "A - clean no music"); recording
                # B's own keeper "A - clean no music - GLM Review.docx"
                # starts with "A - " and would otherwise be misidentified as
                # belonging to recording A's DOCX family and moved away.
                continue
            if candidate.stem != stem and not candidate.stem.startswith(f"{stem} - "):
                continue  # Not part of this recording's DOCX family.
            relative = candidate.relative_to(source_root)
            destination = dest_root / relative
            existing_source = seen.get(destination)
            if existing_source is not None and existing_source != candidate:
                # Two different files planned for the same destination is a
                # real conflict. The same file reached twice -- e.g. a
                # recording kept as both .flac and .mp3, each matching the
                # same shared "<stem>.docx" -- is an expected duplicate, not
                # an error.
                raise ValueError(
                    f"two different files are both planned to move to "
                    f"{destination}: {existing_source} and {candidate}"
                )
            seen[destination] = candidate
    moves = tuple(
        PlannedMove(source=source, destination=destination)
        for destination, source in sorted(seen.items(), key=lambda item: item[1])
    )
    return moves


def apply_moves(
    moves: tuple[PlannedMove, ...], *, confirm: bool, expected_count: int
) -> tuple[Path, ...]:
    if not confirm:
        raise ValueError("apply_moves is dry-run only unless confirm=True")
    if expected_count != len(moves):
        raise ValueError(
            f"expected_count must exactly match the planned {len(moves)} move(s); "
            "the plan may have changed since it was generated -- re-run without "
            "--confirm-move to get a fresh plan"
        )

    moved: list[Path] = []
    for move in moves:
        if not move.source.is_file() or move.source.is_symlink():
            raise ValueError(f"planned source changed type or disappeared: {move.source}")
        if move.destination.exists():
            raise ValueError(f"refusing to overwrite an existing file: {move.destination}")
        extended_parent = Path(_windows_extended_path(move.destination.parent))
        extended_parent.mkdir(parents=True, exist_ok=True)
        shutil.move(
            _windows_extended_path(move.source), _windows_extended_path(move.destination)
        )
        moved.append(move.destination)
    return tuple(moved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument(
        "--confirm-move",
        action="store_true",
        help="Actually move files. Omit to only print the plan.",
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="Required with --confirm-move: exact number of files the plan must match.",
    )
    args = parser.parse_args()

    moves = plan_moves(args.source_root)

    if not args.confirm_move:
        if not moves:
            print("Nothing to move: no superseded DOCX files found.")
            return 0
        print(f"{len(moves)} file(s) would move:")
        for move in moves:
            print(f"  {move.source}\n    -> {move.destination}")
        print()
        print(f"Re-run with --confirm-move --expect {len(moves)} to actually move them.")
        return 0

    if args.expect is None:
        parser.error("--confirm-move requires --expect N (see the dry-run output)")

    moved = apply_moves(moves, confirm=True, expected_count=args.expect)
    print(f"Moved {len(moved)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
