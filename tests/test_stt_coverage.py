from __future__ import annotations

import unittest

from stt_coverage import (
    CLIP_GRID_GRACE_SECONDS,
    assess_stt_coverage,
    coverage_record_is_passed,
    trailing_silence_tolerance_seconds,
)


def _segments(*ends: float) -> list[dict[str, object]]:
    return [{"text": "hello", "start": max(0.0, end - 1.0), "end": end} for end in ends]


class AssessSttCoverageExistingBehaviourTests(unittest.TestCase):
    """Behaviour that predates the clip-grid evidence path must be unchanged."""

    def test_small_gap_within_tolerance_passes_without_evidence_fields(self) -> None:
        # duration=100 -> tolerance = max(2, 5) = 5.0; last_end=97 -> gap=3.0 <= tolerance.
        record = assess_stt_coverage(_segments(97.0), 100.0)
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["reasons"], [])
        self.assertEqual(record["notes"], [])
        self.assertNotIn("clip_seconds", record)
        self.assertNotIn("clip_count", record)
        self.assertNotIn("clip_results_verified", record)
        self.assertTrue(coverage_record_is_passed(record))

    def test_empty_segment_list_fails_without_metadata(self) -> None:
        record = assess_stt_coverage([], 100.0)
        self.assertEqual(record["status"], "needs_review")
        self.assertIn("STT segment list is empty", record["reasons"])
        self.assertEqual(record["notes"], [])

    def test_big_gap_without_metadata_still_fails(self) -> None:
        # duration=100 -> tolerance=5.0; last_end=50 -> gap=50, way over tolerance.
        record = assess_stt_coverage(_segments(50.0), 100.0)
        self.assertEqual(record["status"], "needs_review")
        self.assertEqual(len(record["reasons"]), 1)
        self.assertIn("trailing-silence tolerance", record["reasons"][0])
        self.assertEqual(record["notes"], [])

    def test_stt_metadata_defaults_to_none_and_behaves_as_before(self) -> None:
        record_no_arg = assess_stt_coverage(_segments(50.0), 100.0)
        record_none = assess_stt_coverage(_segments(50.0), 100.0, None)
        self.assertEqual(record_no_arg["reasons"], record_none["reasons"])
        self.assertEqual(record_no_arg["status"], record_none["status"])


class ClipGridEvidenceValidityTests(unittest.TestCase):
    """Exercise the exact evidence-validity rules from the contract."""

    def _big_gap_metadata(self, **overrides: object) -> dict[str, object]:
        # duration=100 -> tolerance=5.0; last_end=50 -> trailing gap=50 (well over tolerance).
        metadata: dict[str, object] = {
            "clip_results_verified": True,
            "clip_seconds": 20.0,
            "clip_count": 5,  # 5 * 20 = 100 >= duration(100) - grace
        }
        metadata.update(overrides)
        return metadata

    def test_valid_evidence_downgrades_gap_to_note_and_carries_evidence_fields(self) -> None:
        metadata = self._big_gap_metadata()
        record = assess_stt_coverage(_segments(50.0), 100.0, metadata)
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["reasons"], [])
        self.assertEqual(len(record["notes"]), 1)
        note = record["notes"][0]
        self.assertIn("trailing 50.000s", note)
        self.assertIn("5 verified 20s clips", note)
        self.assertIn("non-speech audio such as music", note)
        self.assertEqual(record["clip_seconds"], 20.0)
        self.assertEqual(record["clip_count"], 5)
        self.assertIs(record["clip_results_verified"], True)

    def test_missing_clip_results_verified_key_fails(self) -> None:
        metadata = self._big_gap_metadata()
        del metadata["clip_results_verified"]
        record = assess_stt_coverage(_segments(50.0), 100.0, metadata)
        self.assertEqual(record["status"], "needs_review")
        self.assertEqual(record["notes"], [])
        self.assertNotIn("clip_seconds", record)

    def test_clip_results_verified_false_fails(self) -> None:
        metadata = self._big_gap_metadata(clip_results_verified=False)
        record = assess_stt_coverage(_segments(50.0), 100.0, metadata)
        self.assertEqual(record["status"], "needs_review")
        self.assertEqual(record["notes"], [])

    def test_clip_results_verified_truthy_but_not_true_fails(self) -> None:
        # Identity check, not truthiness: 1 and "yes" must NOT count as True.
        for truthy in (1, "yes", 1.0):
            with self.subTest(truthy=truthy):
                metadata = self._big_gap_metadata(clip_results_verified=truthy)
                record = assess_stt_coverage(_segments(50.0), 100.0, metadata)
                self.assertEqual(record["status"], "needs_review")
                self.assertEqual(record["notes"], [])

    def test_bool_clip_count_rejected(self) -> None:
        metadata = self._big_gap_metadata(clip_count=True)
        record = assess_stt_coverage(_segments(50.0), 100.0, metadata)
        self.assertEqual(record["status"], "needs_review")
        self.assertEqual(record["notes"], [])

    def test_non_int_clip_count_rejected(self) -> None:
        metadata = self._big_gap_metadata(clip_count=5.0)
        record = assess_stt_coverage(_segments(50.0), 100.0, metadata)
        self.assertEqual(record["status"], "needs_review")
        self.assertEqual(record["notes"], [])

    def test_zero_or_negative_clip_count_rejected(self) -> None:
        for bad_count in (0, -1):
            with self.subTest(bad_count=bad_count):
                metadata = self._big_gap_metadata(clip_count=bad_count)
                record = assess_stt_coverage(_segments(50.0), 100.0, metadata)
                self.assertEqual(record["status"], "needs_review")

    def test_non_finite_or_non_positive_clip_seconds_rejected(self) -> None:
        for bad_seconds in (0.0, -20.0, float("nan"), float("inf")):
            with self.subTest(bad_seconds=bad_seconds):
                metadata = self._big_gap_metadata(clip_seconds=bad_seconds)
                record = assess_stt_coverage(_segments(50.0), 100.0, metadata)
                self.assertEqual(record["status"], "needs_review")

    def test_clip_grid_short_by_a_full_clip_fails(self) -> None:
        # duration=119 -> tolerance ~5.95; 5 clips * 20s = 100, +2 grace = 102 < 119.
        metadata = self._big_gap_metadata(clip_count=5)
        record = assess_stt_coverage(_segments(50.0), 119.0, metadata)
        self.assertEqual(record["status"], "needs_review")
        self.assertEqual(record["notes"], [])
        self.assertNotIn("clip_seconds", record)

    def test_clip_grid_short_by_up_to_grace_seconds_passes(self) -> None:
        # 5 clips * 20s = 100; grace = 2.0 -> grid covers up to duration=102 exactly.
        self.assertEqual(CLIP_GRID_GRACE_SECONDS, 2.0)
        metadata = self._big_gap_metadata(clip_count=5)
        record = assess_stt_coverage(_segments(50.0), 102.0, metadata)
        self.assertEqual(record["status"], "passed")
        self.assertEqual(len(record["notes"]), 1)

        # A hair over the grace window (still short by more than the grace) fails.
        record_over = assess_stt_coverage(_segments(50.0), 102.5, metadata)
        self.assertEqual(record_over["status"], "needs_review")
        self.assertEqual(record_over["notes"], [])

    def test_stt_metadata_not_a_dict_fails(self) -> None:
        for bad_metadata in ("clip_results_verified", 42, ["clip_results_verified"]):
            with self.subTest(bad_metadata=bad_metadata):
                record = assess_stt_coverage(_segments(50.0), 100.0, bad_metadata)
                self.assertEqual(record["status"], "needs_review")
                self.assertEqual(record["notes"], [])


class EvidenceNeverExcusesOtherFailuresTests(unittest.TestCase):
    def test_overrun_beyond_duration_still_fails_even_with_evidence(self) -> None:
        # last_end (130) beyond duration (100) by more than tolerance, regardless
        # of otherwise-valid clip-grid evidence.
        metadata = {
            "clip_results_verified": True,
            "clip_seconds": 20.0,
            "clip_count": 5,
        }
        record = assess_stt_coverage(_segments(130.0), 100.0, metadata)
        self.assertEqual(record["status"], "needs_review")
        self.assertTrue(
            any("beyond the audio duration" in reason for reason in record["reasons"])
        )

    def test_empty_segment_list_still_fails_even_with_evidence(self) -> None:
        metadata = {
            "clip_results_verified": True,
            "clip_seconds": 20.0,
            "clip_count": 5,
        }
        record = assess_stt_coverage([], 100.0, metadata)
        self.assertEqual(record["status"], "needs_review")
        self.assertIn("STT segment list is empty", record["reasons"])
        self.assertEqual(record["notes"], [])

    def test_segments_with_no_text_bearing_entries_still_fails_even_with_evidence(
        self,
    ) -> None:
        metadata = {
            "clip_results_verified": True,
            "clip_seconds": 20.0,
            "clip_count": 5,
        }
        segments = [{"text": "", "start": 0.0, "end": 10.0}]
        record = assess_stt_coverage(segments, 100.0, metadata)
        self.assertEqual(record["status"], "needs_review")
        self.assertIn(
            "STT segments contain no text-bearing segment with a valid end time",
            record["reasons"],
        )
        self.assertEqual(record["notes"], [])


class CoverageRecordIsPassedEvidenceTests(unittest.TestCase):
    def _persisted_evidence_record(self) -> dict[str, object]:
        # A record as assess_stt_coverage would build+persist it, with a big
        # trailing gap excused by valid clip-grid evidence.
        metadata = {
            "clip_results_verified": True,
            "clip_seconds": 20.0,
            "clip_count": 5,
        }
        record = assess_stt_coverage(_segments(50.0), 100.0, metadata)
        self.assertEqual(record["status"], "passed")
        return record

    def test_accepts_persisted_evidence_record_with_large_trailing_gap(self) -> None:
        record = self._persisted_evidence_record()
        tolerance = trailing_silence_tolerance_seconds(100.0)
        self.assertGreater(record["trailing_silence_seconds"], tolerance)
        self.assertTrue(coverage_record_is_passed(record))

    def test_rejects_when_clip_results_verified_removed(self) -> None:
        record = self._persisted_evidence_record()
        del record["clip_results_verified"]
        self.assertFalse(coverage_record_is_passed(record))

    def test_rejects_when_clip_results_verified_corrupted_to_false(self) -> None:
        record = self._persisted_evidence_record()
        record["clip_results_verified"] = False
        self.assertFalse(coverage_record_is_passed(record))

    def test_rejects_when_clip_seconds_removed(self) -> None:
        record = self._persisted_evidence_record()
        del record["clip_seconds"]
        self.assertFalse(coverage_record_is_passed(record))

    def test_rejects_when_clip_count_removed(self) -> None:
        record = self._persisted_evidence_record()
        del record["clip_count"]
        self.assertFalse(coverage_record_is_passed(record))

    def test_rejects_when_clip_count_corrupted_to_bool(self) -> None:
        record = self._persisted_evidence_record()
        record["clip_count"] = True
        self.assertFalse(coverage_record_is_passed(record))

    def test_rejects_when_clip_count_corrupted_to_too_small(self) -> None:
        record = self._persisted_evidence_record()
        record["clip_count"] = 1  # 1 * 20 + grace(2) = 22, far short of duration 100.
        self.assertFalse(coverage_record_is_passed(record))

    def test_rejects_status_not_passed_even_with_evidence(self) -> None:
        record = self._persisted_evidence_record()
        record["status"] = "needs_review"
        self.assertFalse(coverage_record_is_passed(record))

    def test_still_rejects_overrun_beyond_duration_even_with_evidence(self) -> None:
        record = self._persisted_evidence_record()
        # Evidence excuses the trailing-gap check but not an overrun of last_end
        # beyond duration; corrupt last_end to simulate that scenario directly.
        record["last_segment_end_seconds"] = 130.0
        self.assertFalse(coverage_record_is_passed(record))

    def test_small_gap_record_without_any_evidence_fields_still_passes(self) -> None:
        record = assess_stt_coverage(_segments(97.0), 100.0)
        self.assertTrue(coverage_record_is_passed(record))


if __name__ == "__main__":
    unittest.main()
