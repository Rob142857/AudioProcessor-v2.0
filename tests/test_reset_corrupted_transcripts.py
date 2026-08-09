from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import reset_corrupted_transcripts as tool


def write_manifest(
    job_dir: Path,
    *,
    status: str = "needs_review",
    coverage_status: str = "needs_review",
    gap: float = 200.0,
    last_end: float = 7000.0,
    duration: float = 7300.0,
    genre: str = "Spiritual Teachings",
    title: str = "Some Lecture",
    source: str = "C:\\source\\Some Lecture.mp3",
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": status,
        "source": {"path": source},
        "qa": {
            "stt_coverage": {
                "status": coverage_status,
                "trailing_silence_seconds": gap,
                "last_segment_end_seconds": last_end,
                "audio_duration_seconds": duration,
            }
        },
        "publication": {"metadata": {"genre": genre, "title": title}},
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class FindCandidatesFilteringTests(unittest.TestCase):
    """These never touch ffmpeg -- _measure_silence_fraction is mocked."""

    def test_passed_coverage_is_never_a_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(root / "job1__mp3", coverage_status="passed")
            with mock.patch.object(tool, "_measure_silence_fraction", return_value=0.0):
                candidates = tool.find_candidates(root, progress=False)
            self.assertEqual(candidates, ())

    def test_status_other_than_needs_review_is_never_a_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(root / "job1__mp3", status="failed")
            with mock.patch.object(tool, "_measure_silence_fraction", return_value=0.0):
                candidates = tool.find_candidates(root, progress=False)
            self.assertEqual(candidates, ())

    def test_gap_below_threshold_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(root / "job1__mp3", gap=5.0)
            with mock.patch.object(tool, "_measure_silence_fraction", return_value=0.0):
                candidates = tool.find_candidates(root, min_gap_seconds=30.0, progress=False)
            self.assertEqual(candidates, ())

    def test_music_tagged_genre_is_excluded_even_with_a_large_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(root / "job1__mp3", genre="Objective music", gap=1000.0)
            with mock.patch.object(tool, "_measure_silence_fraction", return_value=0.0):
                candidates = tool.find_candidates(root, progress=False)
            self.assertEqual(candidates, ())

    def test_music_tagged_title_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(root / "job1__mp3", title="Gurdjieff music", genre="", gap=1000.0)
            with mock.patch.object(tool, "_measure_silence_fraction", return_value=0.0):
                candidates = tool.find_candidates(root, progress=False)
            self.assertEqual(candidates, ())

    def test_music_keyword_matches_whole_word_only_not_substring(self):
        # "chant" is a substring of "Merchant" -- a naive substring match
        # would wrongly exclude a real, unrelated lecture from candidacy.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(
                root / "job1__mp3",
                title="The Merchant of Venice",
                genre="Literature",
                gap=1000.0,
            )
            with mock.patch.object(tool, "_measure_silence_fraction", return_value=0.0):
                candidates = tool.find_candidates(root, progress=False)
            self.assertEqual(len(candidates), 1)

    def test_mostly_silent_gap_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(root / "job1__mp3")
            with mock.patch.object(tool, "_measure_silence_fraction", return_value=0.95):
                candidates = tool.find_candidates(root, progress=False)
            self.assertEqual(candidates, ())

    def test_unmeasurable_gap_is_included_and_marked_unverified(self):
        # Silently excluding an unmeasurable job would repeat exactly the
        # kind of silent failure this tool exists to fix -- it must be
        # included (clearly marked), not dropped.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(root / "job1__mp3")
            with mock.patch.object(tool, "_measure_silence_fraction", return_value=None):
                candidates = tool.find_candidates(root, progress=False)
            self.assertEqual(len(candidates), 1)
            self.assertIsNone(candidates[0].silent_fraction)

    def test_real_content_in_the_gap_is_a_confirmed_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "job1__mp3"
            write_manifest(job_dir, gap=200.0, title="A Real Lecture")
            with mock.patch.object(tool, "_measure_silence_fraction", return_value=0.1):
                candidates = tool.find_candidates(root, progress=False)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].job_directory, job_dir.resolve())
            self.assertEqual(candidates[0].title, "A Real Lecture")
            self.assertEqual(candidates[0].silent_fraction, 0.1)

    def test_boundary_silent_fraction_at_threshold_is_excluded(self):
        # >= SILENCE_FRACTION_THRESHOLD is treated as genuinely silent.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(root / "job1__mp3")
            with mock.patch.object(
                tool, "_measure_silence_fraction", return_value=tool.SILENCE_FRACTION_THRESHOLD
            ):
                candidates = tool.find_candidates(root, progress=False)
            self.assertEqual(candidates, ())

    def test_legacy_docx_is_detected_when_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            source_audio = source_dir / "Some Lecture.mp3"
            source_audio.write_bytes(b"audio")
            legacy_docx = source_dir / "Some Lecture.docx"
            legacy_docx.write_bytes(b"old transcript")
            write_manifest(root / "job1__mp3", source=str(source_audio))

            with mock.patch.object(tool, "_measure_silence_fraction", return_value=0.1):
                candidates = tool.find_candidates(root, progress=False)

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].legacy_docx, legacy_docx)

    def test_legacy_docx_is_none_when_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            source_audio = source_dir / "Some Lecture.mp3"
            source_audio.write_bytes(b"audio")
            write_manifest(root / "job1__mp3", source=str(source_audio))

            with mock.patch.object(tool, "_measure_silence_fraction", return_value=0.1):
                candidates = tool.find_candidates(root, progress=False)

            self.assertEqual(len(candidates), 1)
            self.assertIsNone(candidates[0].legacy_docx)


class ApplyResetTests(unittest.TestCase):
    def _make_candidate(self, job_dir: Path, *, legacy_docx: Path | None = None) -> tool.Candidate:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "raw.txt").write_text("corrupted transcript", encoding="utf-8")
        (job_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return tool.Candidate(
            job_directory=job_dir,
            source_audio=Path("C:\\source\\x.mp3"),
            legacy_docx=legacy_docx,
            title="X",
            gap_seconds=200.0,
            audio_duration_seconds=7300.0,
            last_segment_end_seconds=7000.0,
            silent_fraction=0.1,
        )

    def test_refuses_dry_run_without_confirm(self):
        with self.assertRaisesRegex(ValueError, "dry-run only"):
            tool.apply_reset(Path("."), (), confirm=False, expected_count=0)

    def test_requires_exact_expected_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            candidate = self._make_candidate(root / "1985 MW" / "job1__mp3")

            with self.assertRaisesRegex(ValueError, "expected_count"):
                tool.apply_reset(root, (candidate,), confirm=True, expected_count=2)
            self.assertTrue(candidate.job_directory.exists())

    def test_moves_whole_job_directory_preserving_relative_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            candidate = self._make_candidate(root / "1985 MW" / "job1__mp3")

            moved = tool.apply_reset(root, (candidate,), confirm=True, expected_count=1)

            self.assertFalse(candidate.job_directory.exists())
            self.assertEqual(len(moved), 1)
            self.assertTrue((moved[0] / "raw.txt").is_file())
            self.assertEqual(
                (moved[0] / "raw.txt").read_text(encoding="utf-8"), "corrupted transcript"
            )
            self.assertEqual(
                moved[0],
                tool.quarantine_root_for(root) / "1985 MW" / "job1__mp3",
            )

    def test_refuses_to_overwrite_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            candidate = self._make_candidate(root / "1985 MW" / "job1__mp3")
            conflict = tool.quarantine_root_for(root) / "1985 MW" / "job1__mp3"
            conflict.mkdir(parents=True)
            (conflict / "already-here.txt").write_text("x", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                tool.apply_reset(root, (candidate,), confirm=True, expected_count=1)
            self.assertTrue(candidate.job_directory.exists())

    def test_legacy_docx_moves_alongside_the_job_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            source_dir = Path(temporary) / "source"
            source_dir.mkdir()
            legacy_docx = source_dir / "Some Lecture.docx"
            legacy_docx.write_text("old whisper transcript", encoding="utf-8")
            candidate = self._make_candidate(
                root / "1985 MW" / "job1__mp3", legacy_docx=legacy_docx
            )

            moved = tool.apply_reset(root, (candidate,), confirm=True, expected_count=1)

            self.assertFalse(legacy_docx.exists())
            moved_docx = moved[0] / "Some Lecture.docx"
            self.assertTrue(moved_docx.is_file())
            self.assertEqual(
                moved_docx.read_text(encoding="utf-8"), "old whisper transcript"
            )

    def test_missing_legacy_docx_at_apply_time_aborts_before_moving_anything(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            # legacy_docx points at a file that never existed / already moved.
            candidate = self._make_candidate(
                root / "1985 MW" / "job1__mp3",
                legacy_docx=Path(temporary) / "source" / "gone.docx",
            )

            with self.assertRaisesRegex(ValueError, "planned legacy DOCX disappeared"):
                tool.apply_reset(root, (candidate,), confirm=True, expected_count=1)
            self.assertTrue(candidate.job_directory.exists())


class RealFfmpegSilenceMeasurementTests(unittest.TestCase):
    """One end-to-end check against the real bundled ffmpeg.exe."""

    def _write_wav(self, path: Path, *, silent: bool, seconds: float = 3.0, sample_rate: int = 16000) -> None:
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            frame_count = int(seconds * sample_rate)
            if silent:
                writer.writeframes(b"\x00\x00" * frame_count)
            else:
                # A full-scale square wave is loud and clearly non-silent.
                pattern = (b"\xff\x7f" + b"\x00\x80") * (frame_count // 2)
                writer.writeframes(pattern)

    def test_silent_audio_measures_as_fully_silent(self):
        ffmpeg = tool._find_ffmpeg()
        with tempfile.TemporaryDirectory() as temporary:
            wav = Path(temporary) / "silent.wav"
            self._write_wav(wav, silent=True)
            fraction = tool._measure_silence_fraction(ffmpeg, wav, 0.0, 3.0)
            self.assertIsNotNone(fraction)
            self.assertGreater(fraction, 0.9)

    def test_loud_audio_measures_as_not_silent(self):
        ffmpeg = tool._find_ffmpeg()
        with tempfile.TemporaryDirectory() as temporary:
            wav = Path(temporary) / "loud.wav"
            self._write_wav(wav, silent=False)
            fraction = tool._measure_silence_fraction(ffmpeg, wav, 0.0, 3.0)
            self.assertIsNotNone(fraction)
            self.assertLess(fraction, 0.1)

    def test_missing_source_returns_none(self):
        ffmpeg = tool._find_ffmpeg()
        fraction = tool._measure_silence_fraction(
            ffmpeg, Path("does-not-exist.wav"), 0.0, 3.0
        )
        self.assertIsNone(fraction)


if __name__ == "__main__":
    unittest.main()
