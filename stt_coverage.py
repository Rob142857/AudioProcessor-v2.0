"""Shared completeness checks for timestamped speech-to-text output.

Trailing silence is legitimate on tape recordings, so a transcript need not end
at the exact media duration.  The accepted trailing-silence allowance is the
greater of two seconds or five percent of the recording, capped at two minutes.
This is deliberately a review gate rather than an attempt to infer whether a
long silent tail is intentional.

Some recordings end with a genuinely non-speech tail -- music, applause, dead
air -- long enough to exceed that tolerance even though nothing was lost. Such
a tail is indistinguishable from a truncated or silently-dropped transcription
*unless* every clip covering it is independently known to have produced a
verified result.  ``assess_stt_coverage`` accepts an optional ``stt_metadata``
mapping carrying that proof: the STT worker's fixed clip length, the number of
clips it processed, and a ``clip_results_verified`` flag that a worker may
only ever set to ``True`` when it raises rather than silently drops a clip on
a batch-size mismatch (see parakeet_worker.transcribe_one's count-mismatch
guard).  When that evidence is present, the clip grid it describes spans the
recording, and at least one text-bearing segment still exists, an
over-tolerance trailing gap is downgraded from a blocking "reasons" entry to a
"notes" entry documenting why the gap is trusted to be non-speech rather than
lost audio -- instead of looping the transcript through review forever.

Transcripts produced before the count-mismatch guard existed carry no
``clip_results_verified`` field (or carry it unset/falsy) in their metadata,
so they present no evidence and keep failing on an over-tolerance trailing gap
exactly as before: there is no way to retroactively prove a pre-guard clip
was not silently dropped.  Evidence never excuses an entirely empty segment
list (no transcription happened at all) or a last segment that ends *beyond*
the audio duration (a timestamp/duration mismatch, not a silent tail) -- both
remain hard failures regardless of stt_metadata.
"""

from __future__ import annotations

import math
from typing import Any


TRAILING_SILENCE_MIN_SECONDS = 2.0
TRAILING_SILENCE_FRACTION = 0.05
TRAILING_SILENCE_MAX_SECONDS = 120.0

# A clip grid is built from fixed-size clips, and split_wav may fold a
# sub-one-second trailing remainder into the final clip rather than emitting
# a short clip of its own.  That means a genuinely complete clip grid can
# legitimately fall a little short of the measured audio duration.  A
# genuinely missing clip, by contrast, falls short by close to a full
# CLIP_SECONDS (~19s+ at the current 20s clip length) -- nowhere near this
# grace window.
CLIP_GRID_GRACE_SECONDS = 2.0


def finite_seconds(value: Any, *, positive: bool = False) -> float | None:
    """Return a finite seconds value, rejecting booleans and invalid numbers."""

    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    if seconds < 0 or (positive and seconds <= 0):
        return None
    return seconds


def trailing_silence_tolerance_seconds(audio_duration_seconds: float) -> float:
    """Return the documented duration-scaled trailing-silence allowance."""

    duration = finite_seconds(audio_duration_seconds, positive=True)
    if duration is None:
        raise ValueError("audio duration must be a positive finite number")
    return min(
        TRAILING_SILENCE_MAX_SECONDS,
        max(TRAILING_SILENCE_MIN_SECONDS, duration * TRAILING_SILENCE_FRACTION),
    )


def _clip_grid_evidence(
    stt_metadata: Any, duration: float | None
) -> tuple[float, int] | None:
    """Return (clip_seconds, clip_count) proving a verified clip grid spans the audio.

    Returns ``None`` unless every one of the following holds:

    * ``stt_metadata`` is a dict and ``duration`` is a known finite positive
      number (evidence is meaningless without a duration to compare against).
    * ``stt_metadata.get("clip_results_verified") is True`` -- an identity
      check, not a truthiness check, so nothing short of the literal ``True``
      a guarded worker writes counts as evidence.
    * ``clip_seconds`` is a finite positive number.
    * ``clip_count`` is an exact ``int`` (bools rejected) greater than zero.
    * the described grid, plus :data:`CLIP_GRID_GRACE_SECONDS` of slack, spans
      at least the full audio duration -- i.e. the worker's clips actually
      covered the recording rather than stopping partway through it.
    """

    if not isinstance(stt_metadata, dict) or duration is None:
        return None
    if stt_metadata.get("clip_results_verified") is not True:
        return None
    clip_seconds = finite_seconds(stt_metadata.get("clip_seconds"), positive=True)
    if clip_seconds is None:
        return None
    clip_count = stt_metadata.get("clip_count")
    if type(clip_count) is not int or clip_count <= 0:
        return None
    if clip_count * clip_seconds + CLIP_GRID_GRACE_SECONDS < duration:
        return None
    return clip_seconds, clip_count


def assess_stt_coverage(
    segments: Any,
    audio_duration_seconds: Any,
    stt_metadata: Any = None,
) -> dict[str, Any]:
    """Assess whether text-bearing STT segments adequately cover the recording.

    ``stt_metadata``, when supplied, is checked for verified-clip-grid
    evidence (see :func:`_clip_grid_evidence` and the module docstring).  When
    that evidence is valid and covers an over-tolerance trailing gap, the gap
    is recorded as a "notes" entry documenting the non-speech tail instead of
    a blocking "reasons" entry, and the evidence fields are echoed back onto
    the returned record.  The empty-segment-list and beyond-duration-overrun
    reasons are never excused by evidence.
    """

    reasons: list[str] = []
    notes: list[str] = []
    segment_list = segments if isinstance(segments, list) else []
    if not segment_list:
        reasons.append("STT segment list is empty")

    text_segment_ends: list[float] = []
    for segment in segment_list:
        if not isinstance(segment, dict) or not str(segment.get("text") or "").strip():
            continue
        end = finite_seconds(segment.get("end"))
        if end is not None:
            text_segment_ends.append(end)

    if segment_list and not text_segment_ends:
        reasons.append("STT segments contain no text-bearing segment with a valid end time")

    duration = finite_seconds(audio_duration_seconds, positive=True)
    if duration is None:
        reasons.append("audio duration is unavailable; STT coverage cannot be verified")

    last_end = text_segment_ends[-1] if text_segment_ends else None
    tolerance = (
        trailing_silence_tolerance_seconds(duration)
        if duration is not None
        else None
    )
    trailing_silence = (
        max(0.0, duration - last_end)
        if duration is not None and last_end is not None
        else None
    )
    coverage_ratio = (
        min(1.0, last_end / duration)
        if duration is not None and last_end is not None
        else None
    )

    clip_evidence = _clip_grid_evidence(stt_metadata, duration)

    if (
        trailing_silence is not None
        and tolerance is not None
        and trailing_silence > tolerance
    ):
        if clip_evidence is not None and text_segment_ends:
            clip_seconds, clip_count = clip_evidence
            notes.append(
                f"trailing {trailing_silence:.3f}s produced no speech text across "
                f"{clip_count} verified {clip_seconds:g}s clips "
                "(non-speech audio such as music)"
            )
        else:
            reasons.append(
                "last STT segment ends "
                f"{trailing_silence:.3f}s before the audio ends; documented "
                f"trailing-silence tolerance is {tolerance:.3f}s"
            )
    if (
        duration is not None
        and last_end is not None
        and tolerance is not None
        and last_end - duration > tolerance
    ):
        reasons.append(
            "last STT segment extends "
            f"{last_end - duration:.3f}s beyond the audio duration"
        )

    record: dict[str, Any] = {
        "status": "needs_review" if reasons else "passed",
        "reasons": reasons,
        "notes": notes,
        "segment_count": len(segment_list),
        "text_segment_count": len(text_segment_ends),
        "audio_duration_seconds": duration,
        "last_segment_end_seconds": last_end,
        "trailing_silence_seconds": trailing_silence,
        "trailing_silence_tolerance_seconds": tolerance,
        "coverage_ratio": coverage_ratio,
    }
    if clip_evidence is not None:
        clip_seconds, clip_count = clip_evidence
        record["clip_seconds"] = clip_seconds
        record["clip_count"] = clip_count
        record["clip_results_verified"] = True
    return record


def coverage_record_is_passed(value: Any) -> bool:
    """Return whether a persisted assessment contains complete passing evidence.

    The trailing-gap requirement is satisfied either the plain way (the gap is
    within the documented silence tolerance) or via verified-clip-grid
    evidence recorded directly on ``value`` (see :func:`_clip_grid_evidence`);
    a record missing or corrupting any evidence field falls back to the plain
    check and fails it exactly as a record with no evidence would.  Every
    other existing condition -- including the beyond-duration overrun check --
    is unchanged and is never excused by evidence.
    """

    if not isinstance(value, dict) or value.get("status") != "passed":
        return False
    if type(value.get("segment_count")) is not int or value["segment_count"] <= 0:
        return False
    if (
        type(value.get("text_segment_count")) is not int
        or value["text_segment_count"] <= 0
    ):
        return False
    duration = finite_seconds(value.get("audio_duration_seconds"), positive=True)
    last_end = finite_seconds(value.get("last_segment_end_seconds"))
    trailing = finite_seconds(value.get("trailing_silence_seconds"))
    tolerance = finite_seconds(value.get("trailing_silence_tolerance_seconds"))
    if None in (duration, last_end, trailing, tolerance):
        return False
    expected_tolerance = trailing_silence_tolerance_seconds(duration)
    trailing_ok = trailing <= tolerance + 0.001 or (
        _clip_grid_evidence(value, duration) is not None
    )
    return (
        math.isclose(tolerance, expected_tolerance, rel_tol=0.0, abs_tol=0.001)
        and trailing_ok
        and last_end - duration <= tolerance + 0.001
    )


__all__ = [
    "CLIP_GRID_GRACE_SECONDS",
    "TRAILING_SILENCE_FRACTION",
    "TRAILING_SILENCE_MAX_SECONDS",
    "TRAILING_SILENCE_MIN_SECONDS",
    "assess_stt_coverage",
    "coverage_record_is_passed",
    "finite_seconds",
    "trailing_silence_tolerance_seconds",
]
