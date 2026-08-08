import queue
import unittest
from pathlib import Path
from unittest import mock

import parakeet_stt
from parakeet_stt import MIN_TRANSCRIBABLE_SECONDS, ParakeetError, ParakeetSession


class _ExitedWorker:
    stdin = None

    @staticmethod
    def poll():
        return -1073740791


class ParakeetSessionTests(unittest.TestCase):
    def test_read_response_detects_exit_after_output_queue_drains(self):
        """A native abort can report its exit just after stdout closes."""

        session = ParakeetSession()
        session.process = _ExitedWorker()
        session._output = queue.Queue()

        with self.assertRaisesRegex(ParakeetError, "exited unexpectedly"):
            session._read_response(
                expected_id="job-1",
                timeout=1,
                cancel_check=None,
            )


class ParakeetPreflightLengthCheckTests(unittest.TestCase):
    """Fix 6 -- reject near-empty audio before it ever reaches the model.

    NeMo's own failure for this case is a raw ``ValueError`` out of
    ``normalize_batch`` ("received a tensor of length 1 ... torch.std() ...
    nan"). The pre-flight check must turn that into a clear ``ParakeetError``
    and must never start the (GPU) worker to do it.
    """

    def test_rejects_audio_shorter_than_minimum_without_starting_worker(self):
        session = ParakeetSession()
        session.start = mock.Mock(
            side_effect=AssertionError("pre-flight must reject before starting the worker")
        )

        with mock.patch.object(parakeet_stt, "_probe_duration_seconds", return_value=0.03):
            with self.assertRaisesRegex(
                ParakeetError, r"audio too short to transcribe: 0\.03s"
            ):
                session.transcribe(Path("near-empty.wav"))

        session.start.assert_not_called()

    def test_rejects_audio_at_zero_duration(self):
        session = ParakeetSession()
        session.start = mock.Mock(side_effect=AssertionError("must not start the worker"))

        with mock.patch.object(parakeet_stt, "_probe_duration_seconds", return_value=0.0):
            with self.assertRaisesRegex(ParakeetError, r"audio too short to transcribe"):
                session.transcribe(Path("empty.wav"))

        session.start.assert_not_called()

    def test_allows_audio_at_or_above_minimum_duration(self):
        session = ParakeetSession()
        session.start = mock.Mock(side_effect=RuntimeError("reached worker dispatch"))

        with mock.patch.object(
            parakeet_stt, "_probe_duration_seconds", return_value=MIN_TRANSCRIBABLE_SECONDS
        ):
            with self.assertRaisesRegex(RuntimeError, "reached worker dispatch"):
                session.transcribe(Path("plenty-long.wav"))

        session.start.assert_called_once()

    def test_skips_check_when_duration_cannot_be_determined(self):
        """A probe failure (None) must not block transcription."""

        session = ParakeetSession()
        session.start = mock.Mock(side_effect=RuntimeError("reached worker dispatch"))

        with mock.patch.object(parakeet_stt, "_probe_duration_seconds", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "reached worker dispatch"):
                session.transcribe(Path("unknown-duration.wav"))

        session.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
