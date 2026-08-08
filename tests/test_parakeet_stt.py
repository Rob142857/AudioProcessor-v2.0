import queue
import unittest

from parakeet_stt import ParakeetError, ParakeetSession


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


if __name__ == "__main__":
    unittest.main()
