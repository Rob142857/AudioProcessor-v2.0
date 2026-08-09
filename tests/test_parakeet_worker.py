from __future__ import annotations

import importlib.util
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import parakeet_worker

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _write_wav(path: Path, *, seconds: float, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(seconds * sample_rate))


class SplitWavTrailingRemainderTests(unittest.TestCase):
    """`split_wav` needs no torch/nemo -- it only uses the stdlib `wave` module."""

    def test_exact_multiple_of_clip_length_produces_no_remainder(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            _write_wav(source, seconds=40.0)  # exactly two 20s clips
            clips, duration = parakeet_worker.split_wav(source, Path(temporary), seconds=20)
            self.assertEqual(len(clips), 2)
            self.assertAlmostEqual(duration, 40.0, places=2)

    def test_tiny_trailing_remainder_is_merged_into_previous_clip_not_standalone(self):
        # 20s + 0.3s: production crash shape -- a long recording whose
        # duration isn't an exact multiple of the clip length leaves a
        # sub-second final chunk that used to be submitted to NeMo on its
        # own and crash transcribe() with a fatal ValueError.
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            _write_wav(source, seconds=20.3)
            clips, duration = parakeet_worker.split_wav(source, Path(temporary), seconds=20)
            self.assertEqual(len(clips), 1, "the short remainder must not become its own clip")
            with wave.open(str(clips[0]), "rb") as reader:
                merged_seconds = reader.getnframes() / reader.getframerate()
            self.assertAlmostEqual(merged_seconds, 20.3, places=2)

    def test_substantial_trailing_remainder_stays_its_own_clip(self):
        # 20s + 15s: a real, well-formed final clip -- must NOT be merged
        # away just because it's shorter than a full 20s period.
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            _write_wav(source, seconds=35.0)
            clips, duration = parakeet_worker.split_wav(source, Path(temporary), seconds=20)
            self.assertEqual(len(clips), 2)
            with wave.open(str(clips[-1]), "rb") as reader:
                last_seconds = reader.getnframes() / reader.getframerate()
            self.assertAlmostEqual(last_seconds, 15.0, places=2)

    def test_single_short_clip_total_has_nothing_to_merge_into(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            _write_wav(source, seconds=0.3)
            clips, duration = parakeet_worker.split_wav(source, Path(temporary), seconds=20)
            self.assertEqual(len(clips), 1)

    def test_merged_clip_never_falls_below_the_trailing_minimum(self):
        for extra_seconds in (0.05, 0.5, 0.99):
            with self.subTest(extra_seconds=extra_seconds):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "source.wav"
                    _write_wav(source, seconds=20 + extra_seconds)
                    clips, _ = parakeet_worker.split_wav(source, Path(temporary), seconds=20)
                    self.assertEqual(len(clips), 1)


class ClipBatchesTests(unittest.TestCase):
    """`_clip_batches` needs no torch/nemo -- it is pure integer arithmetic."""

    def test_covers_every_clip_with_no_gaps_or_overlaps(self):
        for clip_count in (0, 1, 2, 47, 48, 49, 96, 97, 100, 145, 558):
            with self.subTest(clip_count=clip_count):
                boundaries = parakeet_worker._clip_batches(clip_count, 48)
                covered = []
                for start, size in boundaries:
                    covered.extend(range(start, start + size))
                self.assertEqual(covered, list(range(clip_count)))

    def test_never_leaves_a_trailing_singleton_batch(self):
        # 145 clips is the exact production shape (48+48+48+1) that crashed
        # NeMo's transcribe() with a native, non-Python-catchable CUDA abort
        # every time it occurred.
        for clip_count in range(2, 300):
            with self.subTest(clip_count=clip_count):
                boundaries = parakeet_worker._clip_batches(clip_count, 48)
                for start, size in boundaries:
                    is_last = start + size == clip_count
                    if is_last and clip_count > 1:
                        self.assertNotEqual(
                            size,
                            1,
                            f"clip_count={clip_count} produced a trailing "
                            f"singleton batch {boundaries}",
                        )

    def test_145_clips_matches_the_observed_production_crash_shape(self):
        self.assertEqual(
            parakeet_worker._clip_batches(145, 48), [(0, 48), (48, 48), (96, 49)]
        )

    def test_single_clip_total_has_nothing_to_merge_into(self):
        # The only case a size-1 batch can't be avoided: the whole recording
        # is one clip.
        self.assertEqual(parakeet_worker._clip_batches(1, 48), [(0, 1)])

    def test_no_batch_exceeds_max_size_by_more_than_one(self):
        for clip_count in range(2, 300):
            for start, size in parakeet_worker._clip_batches(clip_count, 48):
                self.assertLessEqual(size, 49)


@unittest.skipUnless(_HAS_TORCH, "requires the isolated .parakeet-venv (torch)")
class TranscribeOneMismatchGuardTests(unittest.TestCase):
    """Regression test for silently misaligned timestamps.

    Production lectures had 5-30+ minutes of real, non-silent speech missing
    from the end of their transcripts because `hypotheses.extend(...)` never
    checked that NeMo returned one result per submitted clip. Confirmed via
    direct audio analysis (ffmpeg silencedetect) that the "missing" tail was
    not silence -- the alignment between clips and results had drifted.
    """

    def _write_silence_wav(self, path: Path, seconds: float, sample_rate: int = 16000) -> None:
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(b"\x00\x00" * int(seconds * sample_rate))

    def test_raises_when_a_batch_returns_fewer_results_than_clips(self):
        import torch  # noqa: F401  (imported for the module's own `import torch`)

        with mock.patch.object(parakeet_worker, "prepare_mono_audio") as prepare:
            def fake_prepare(source, destination):
                self._write_silence_wav(Path(destination), seconds=25.0)

            prepare.side_effect = fake_prepare

            model = mock.MagicMock()
            fake_parameter = mock.MagicMock(device=mock.MagicMock(type="cpu"))
            model.parameters.side_effect = lambda: iter([fake_parameter])
            # Two clips submitted (0-20s, 20-25s); only one result returned.
            model.transcribe.return_value = [mock.MagicMock(text="only one result")]

            with self.assertRaisesRegex(RuntimeError, "refusing to build a transcript"):
                parakeet_worker.transcribe_one(model, Path("unused.wav"), model_name="test-model")

    def test_matching_counts_proceed_normally(self):
        import torch  # noqa: F401

        with mock.patch.object(parakeet_worker, "prepare_mono_audio") as prepare:
            def fake_prepare(source, destination):
                self._write_silence_wav(Path(destination), seconds=25.0)

            prepare.side_effect = fake_prepare

            model = mock.MagicMock()
            fake_parameter = mock.MagicMock(device=mock.MagicMock(type="cpu"))
            model.parameters.side_effect = lambda: iter([fake_parameter])
            model.transcribe.return_value = [
                mock.MagicMock(text="first clip"),
                mock.MagicMock(text="second clip"),
            ]

            result = parakeet_worker.transcribe_one(
                model, Path("unused.wav"), model_name="test-model"
            )

            self.assertEqual(len(result["segments"]), 2)
            self.assertEqual(result["segments"][0]["text"], "first clip")
            self.assertEqual(result["segments"][1]["text"], "second clip")


if __name__ == "__main__":
    unittest.main()
