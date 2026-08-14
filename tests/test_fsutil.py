import json
import tempfile
import unittest
from pathlib import Path

from fsutil import (
    append_jsonl_locked,
    atomic_write_json,
    sha256_file,
    sha256_text,
)


class FsutilTests(unittest.TestCase):
    def test_atomic_write_json_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "state.json"
            atomic_write_json(path, {"value": "é"})
            self.assertEqual({"value": "é"}, json.loads(path.read_text(encoding="utf-8")))

    def test_hash_helpers_are_canonical_and_explicitly_truncatable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "value.txt"
            path.write_text("hello", encoding="utf-8")
            self.assertEqual(sha256_text("hello"), sha256_file(path))
            self.assertEqual(12, len(sha256_text("hello", truncate=12)))
            with self.assertRaises(ValueError):
                sha256_text("hello", truncate=0)

    def test_locked_jsonl_append_preserves_one_complete_event_per_call(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "feed" / "events.jsonl"
            append_jsonl_locked(path, {"event": 1})
            append_jsonl_locked(path, {"event": 2})
            self.assertEqual(
                [{"event": 1}, {"event": 2}],
                [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()],
            )

    def test_locked_append_never_removes_another_writers_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            lock = Path(str(path) + ".lock")
            lock.write_text("other-process", encoding="utf-8")
            with self.assertRaises(TimeoutError):
                append_jsonl_locked(path, {"event": 1}, max_attempts=1)
            self.assertTrue(lock.exists())
