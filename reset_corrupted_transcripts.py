"""Quarantine transcripts corrupted by the pre-fix Parakeet timestamp bug.

Two-step, safety-first operation matching legacy_docx_replace.py's and
archive_older_transcripts.py's pattern: inspect the plan first (the
default), then repeat with ``--confirm-reset --expect N`` to actually act.

Before this session's fix, parakeet_worker.py could silently misalign clip
results with clip timestamps, causing the true tail of long recordings to
fall off the transcript with no error raised anywhere (see
docs/RESILIENCE_FIX_PLAN.md and the commit that added the count-mismatch
guard to parakeet_worker.py). Affected jobs are stuck: their manifest
already holds a "durable" raw transcript, so a plain pipeline re-run just
reuses the corrupted checkpoint forever instead of re-transcribing.

This tool finds jobs whose coverage check failed specifically because of a
trailing gap, then -- rather than trust the heuristic alone -- verifies
against the *actual source audio* with ffmpeg's silencedetect that the
"missing" portion is not simply blank tape. Candidates where real,
non-silent audio was measured in the gap (or where the gap could not be
probed at all -- excluding those would repeat the exact silent-failure
pattern this tool exists to fix) have their whole job directory moved
(never deleted) to a parallel quarantine root next to the output root.

Moving the job directory alone is *not* sufficient: archive_pipeline.py's
discover_audio() decides whether to even look at a source again based on a
separate, source-adjacent legacy DOCX (`<source>.docx`, independent of any
"- GLM Review.docx") and the project's current existing-docx-mode/
replace-before-date GUI setting -- a transient setting this tool has no
control over and should not have to trust. If that legacy DOCX exists, it
is moved into quarantine alongside the job directory too, so
should_process_existing_docx() sees no file and unconditionally re-selects
the source regardless of whatever mode is configured when the pipeline next
runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple, Optional

DEFAULT_MIN_GAP_SECONDS = 30.0
# A gap where ffmpeg measures at least this fraction as silence is treated as
# genuine trailing silence (tape kept rolling after the talk ended), not lost
# speech -- left alone rather than queued for an expensive re-transcription.
SILENCE_FRACTION_THRESHOLD = 0.5
SILENCEDETECT_NOISE_FLOOR = "-35dB"
SILENCEDETECT_MIN_DURATION = "1"
# Whole-word match, not substring: a naive substring check would wrongly
# exclude a real lecture titled e.g. "The Merchant of Venice" (contains
# "chant") or genre "Enchantment" (contains "chant") from candidacy --
# silently dropping a genuinely bug-corrupted job from the plan.
_MUSIC_KEYWORDS_PATTERN = re.compile(
    r"\b(music|chant|chime|song|hymn|mantra|bhajan|kirtan|gong|bell)\b"
)


def _windows_extended_path(path: str | Path) -> str:
    """Return an extended Windows path when a name exceeds MAX_PATH.

    Same helper as cleanup_client.py / archive_pipeline.py /
    legacy_docx_replace.py / archive_older_transcripts.py, duplicated rather
    than imported.
    """

    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def _extended(path: Path) -> Path:
    return Path(_windows_extended_path(path))


def _find_ffmpeg() -> str:
    bundled = Path(__file__).with_name("ffmpeg.exe")
    return str(bundled) if bundled.is_file() else "ffmpeg"


class Candidate(NamedTuple):
    job_directory: Path
    source_audio: Path
    legacy_docx: Optional[Path]  # <source>.docx if present; must move too.
    title: Optional[str]
    gap_seconds: float
    audio_duration_seconds: float
    last_segment_end_seconds: float
    silent_fraction: Optional[float]  # None means "could not be probed".


def _is_music_tagged(manifest: dict[str, Any]) -> bool:
    meta = ((manifest.get("publication") or {}).get("metadata")) or {}
    blob = f"{meta.get('genre') or ''} {meta.get('title') or ''}".lower()
    return bool(_MUSIC_KEYWORDS_PATTERN.search(blob))


def _measure_silence_fraction(
    ffmpeg: str, source: Path, start_seconds: float, gap_seconds: float
) -> Optional[float]:
    """Return the fraction of [start, start+gap] ffmpeg measures as silent.

    Returns None if the source can't be probed (missing, unreadable, etc.);
    callers must not treat that as "confirmed silent".
    """

    if gap_seconds <= 0 or not _extended(source).is_file():
        return None
    command = [
        ffmpeg,
        "-nostdin",
        "-ss",
        str(start_seconds),
        "-i",
        _windows_extended_path(source),
        "-af",
        f"silencedetect=noise={SILENCEDETECT_NOISE_FLOOR}:d={SILENCEDETECT_MIN_DURATION}",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return None
    starts = [float(x) for x in re.findall(r"silence_start:\s*([\-0-9.]+)", completed.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([\-0-9.]+)", completed.stderr)]
    silent_seconds = sum(
        ends[i] - max(0.0, starts[i]) for i in range(min(len(starts), len(ends)))
    )
    return min(1.0, silent_seconds / gap_seconds)


def find_candidates(
    output_root: Path,
    *,
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS,
    ffmpeg: Optional[str] = None,
    progress: bool = True,
) -> tuple[Candidate, ...]:
    """Read-only: identify jobs to quarantine. Touches nothing."""

    output_root = Path(output_root).resolve()
    ffmpeg = ffmpeg or _find_ffmpeg()
    candidates: list[Candidate] = []

    manifest_paths = sorted(output_root.rglob("manifest.json"))
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("status") != "needs_review":
            continue
        coverage = manifest.get("qa", {}).get("stt_coverage", {})
        if coverage.get("status") == "passed":
            continue
        gap = coverage.get("trailing_silence_seconds")
        if not isinstance(gap, (int, float)) or gap < min_gap_seconds:
            continue
        if _is_music_tagged(manifest):
            continue
        source_value = manifest.get("source", {}).get("path")
        last_end = coverage.get("last_segment_end_seconds")
        duration = coverage.get("audio_duration_seconds")
        if not source_value or last_end is None or duration is None:
            continue
        source_audio = Path(source_value)
        if progress:
            print(f"Checking {source_audio.name} ...", file=sys.stderr)
        silent_fraction = _measure_silence_fraction(ffmpeg, source_audio, last_end, gap)
        if silent_fraction is not None and silent_fraction >= SILENCE_FRACTION_THRESHOLD:
            continue  # Confirmed genuine trailing silence -- leave alone.
        # silent_fraction is either < threshold (confirmed real content) or
        # None (couldn't be probed -- included anyway, clearly marked, since
        # silently excluding an unmeasurable job would repeat the same
        # silent-failure pattern this tool exists to fix).
        legacy_docx = source_audio.with_suffix(".docx")
        meta = ((manifest.get("publication") or {}).get("metadata")) or {}
        candidates.append(
            Candidate(
                job_directory=manifest_path.parent,
                source_audio=source_audio,
                legacy_docx=legacy_docx if _extended(legacy_docx).is_file() else None,
                title=meta.get("title"),
                gap_seconds=float(gap),
                audio_duration_seconds=float(duration),
                last_segment_end_seconds=float(last_end),
                silent_fraction=silent_fraction,
            )
        )
    return tuple(candidates)


def quarantine_root_for(output_root: Path) -> Path:
    output_root = Path(output_root).resolve()
    return output_root.parent / f"{output_root.name} - corrupted transcripts (pre-parakeet-fix)"


def apply_reset(
    output_root: Path,
    candidates: tuple[Candidate, ...],
    *,
    confirm: bool,
    expected_count: int,
) -> tuple[Path, ...]:
    if not confirm:
        raise ValueError("apply_reset is dry-run only unless confirm=True")
    if expected_count != len(candidates):
        raise ValueError(
            f"expected_count must exactly match the planned {len(candidates)} "
            "job(s); the plan may have changed since it was generated -- "
            "re-run without --confirm-reset to get a fresh plan"
        )

    quarantine_root = quarantine_root_for(output_root)
    output_root = Path(output_root).resolve()

    # Validate every destination up front, before moving anything, so a
    # conflict discovered on candidate 30 of 45 can't leave a mix of moved
    # and un-moved jobs from a single call that could have been caught
    # entirely up front.
    planned: list[tuple[Candidate, Path, Optional[Path]]] = []
    for candidate in candidates:
        job_directory = candidate.job_directory
        if not _extended(job_directory).is_dir():
            raise ValueError(f"planned job directory disappeared: {job_directory}")
        # Resolve defensively rather than trust the caller's path already
        # matches output_root's resolved form (e.g. Windows 8.3 short names
        # can otherwise make an equivalent path fail relative_to()).
        relative = job_directory.resolve().relative_to(output_root)
        destination = quarantine_root / relative
        if _extended(destination).exists():
            raise ValueError(f"refusing to overwrite an existing destination: {destination}")
        legacy_destination = None
        if candidate.legacy_docx is not None:
            if not _extended(candidate.legacy_docx).is_file():
                raise ValueError(
                    f"planned legacy DOCX disappeared: {candidate.legacy_docx}"
                )
            legacy_destination = destination / candidate.legacy_docx.name
            if _extended(legacy_destination).exists():
                raise ValueError(
                    f"refusing to overwrite an existing destination: {legacy_destination}"
                )
        planned.append((candidate, destination, legacy_destination))

    moved: list[Path] = []
    for candidate, destination, legacy_destination in planned:
        extended_parent = _extended(destination.parent)
        extended_parent.mkdir(parents=True, exist_ok=True)
        shutil.move(_windows_extended_path(candidate.job_directory), _windows_extended_path(destination))
        moved.append(destination)
        if legacy_destination is not None:
            shutil.move(
                _windows_extended_path(candidate.legacy_docx),
                _windows_extended_path(legacy_destination),
            )
    return tuple(moved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path, help="The pipeline's generated/output root.")
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=DEFAULT_MIN_GAP_SECONDS,
        help=f"Only consider jobs with a trailing gap at least this long (default {DEFAULT_MIN_GAP_SECONDS}s).",
    )
    parser.add_argument(
        "--confirm-reset",
        action="store_true",
        help="Actually quarantine the affected job directories. Omit to only print the plan.",
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="Required with --confirm-reset: exact number of jobs the plan must match.",
    )
    args = parser.parse_args()

    candidates = find_candidates(args.output_root, min_gap_seconds=args.min_gap_seconds)

    if not args.confirm_reset:
        if not candidates:
            print("Nothing to reset: no jobs confirmed affected by the truncation bug.")
            return 0
        unverified = sum(1 for c in candidates if c.silent_fraction is None)
        print(f"{len(candidates)} job(s) planned for quarantine:")
        for c in candidates:
            if c.silent_fraction is None:
                evidence = "UNVERIFIED -- source audio could not be probed"
            else:
                evidence = f"{c.silent_fraction * 100:.0f}% measured silent"
            print(f"  {c.title or c.job_directory.name}: {c.gap_seconds:.0f}s gap, {evidence}")
            print(f"    {c.job_directory}")
            if c.legacy_docx is not None:
                print(f"    (also moving legacy DOCX: {c.legacy_docx})")
        print()
        if unverified:
            print(
                f"{unverified} of these could not be verified against the source audio "
                "(missing/unreadable file, or ffmpeg failure) -- included anyway rather "
                "than silently skipped, since that would repeat this tool's whole reason "
                "for existing. Review them individually if in doubt."
            )
            print()
        print(
            f"Re-run with --confirm-reset --expect {len(candidates)} to move these aside "
            f"to:\n  {quarantine_root_for(args.output_root)}"
        )
        print("The next full pipeline run will then redo them from scratch.")
        return 0

    if args.expect is None:
        parser.error("--confirm-reset requires --expect N (see the dry-run output)")

    moved = apply_reset(
        args.output_root, candidates, confirm=True, expected_count=args.expect
    )
    print(f"Quarantined {len(moved)} job(s). Re-run the pipeline to redo them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
