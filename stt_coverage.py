"""Shared completeness checks for timestamped speech-to-text output.

Trailing silence is legitimate on tape recordings, so a transcript need not end
at the exact media duration.  The accepted trailing-silence allowance is the
greater of two seconds or five percent of the recording, capped at two minutes.
This is deliberately a review gate rather than an attempt to infer whether a
long silent tail is intentional.
"""

from __future__ import annotations

import math
from typing import Any


TRAILING_SILENCE_MIN_SECONDS = 2.0
TRAILING_SILENCE_FRACTION = 0.05
TRAILING_SILENCE_MAX_SECONDS = 120.0


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


def assess_stt_coverage(
    segments: Any,
    audio_duration_seconds: Any,
) -> dict[str, Any]:
    """Assess whether text-bearing STT segments adequately cover the recording."""

    reasons: list[str] = []
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

    if (
        trailing_silence is not None
        and tolerance is not None
        and trailing_silence > tolerance
    ):
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

    return {
        "status": "needs_review" if reasons else "passed",
        "reasons": reasons,
        "segment_count": len(segment_list),
        "text_segment_count": len(text_segment_ends),
        "audio_duration_seconds": duration,
        "last_segment_end_seconds": last_end,
        "trailing_silence_seconds": trailing_silence,
        "trailing_silence_tolerance_seconds": tolerance,
        "coverage_ratio": coverage_ratio,
    }


def coverage_record_is_passed(value: Any) -> bool:
    """Return whether a persisted assessment contains complete passing evidence."""

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
    return (
        math.isclose(tolerance, expected_tolerance, rel_tol=0.0, abs_tol=0.001)
        and trailing <= tolerance + 0.001
        and last_end - duration <= tolerance + 0.001
    )


__all__ = [
    "TRAILING_SILENCE_FRACTION",
    "TRAILING_SILENCE_MAX_SECONDS",
    "TRAILING_SILENCE_MIN_SECONDS",
    "assess_stt_coverage",
    "coverage_record_is_passed",
    "finite_seconds",
    "trailing_silence_tolerance_seconds",
]
