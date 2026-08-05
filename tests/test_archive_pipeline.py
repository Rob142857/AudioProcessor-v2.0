import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from archive_pipeline import (
    PipelineConfig,
    PipelineRunner,
    artifact_directory,
    artifact_paths,
    discover_audio,
    sha256_text,
)


class FakeCleanupResult:
    def __init__(self, text: str, *, needs_review: bool = False):
        self.text = text
        self.model = "@cf/zai-org/glm-4.7-flash"
        self.glossary_sha256 = "abc123"
        self.glossary_count = 1635
        self.chunks = (object(),)
        self.needs_review = needs_review
        self.warnings = ()

    def to_dict(self):
        return {
            "model": self.model,
            "glossary_sha256": self.glossary_sha256,
            "glossary_count": self.glossary_count,
            "chunks": [
                {
                    "grounding": {"glossary_terms_considered": 1635},
                    "quality": {"status": "passed"},
                }
            ],
            "needs_review": self.needs_review,
        }


class FakeCleanupClient:
    def __init__(self, *, needs_review: bool = False):
        self.needs_review = needs_review
        self.calls = 0
        self.glossary_sha256 = "abc123"
        self.reuse_checkpoints = []

    def ensure_glossary(self):
        return SimpleNamespace(sha256=self.glossary_sha256)

    def cleanup_text(self, text, checkpoint_dir=None, *, reuse_checkpoints=True):
        self.calls += 1
        self.reuse_checkpoints.append(reuse_checkpoints)
        return FakeCleanupResult(text.rstrip() + ".", needs_review=self.needs_review)


class ArchivePipelineTests(unittest.TestCase):
    def test_discovery_includes_legacy_formats_and_excludes_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "archive"
            output = root / "generated"
            root.mkdir()
            output.mkdir()
            for name in ("one.mp3", "two.aiff", "three.3gp", "ignore.txt"):
                (root / name).write_bytes(b"data")
            (output / "nested.mp3").write_bytes(b"data")

            found = [item.name for item in discover_audio(root, output)]

            self.assertEqual(found, ["one.mp3", "three.3gp", "two.aiff"])

    def test_same_stem_different_formats_have_distinct_artifact_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = root / "lecture.mp3"
            second = root / "lecture.flac"
            output = root / "out"
            self.assertNotEqual(
                artifact_directory(first, root, output),
                artifact_directory(second, root, output),
            )

    def _runner(
        self,
        root: Path,
        *,
        cleanup=True,
        review=False,
        cleanup_only=False,
        retry_review=False,
        render_only=False,
        force=False,
    ):
        source_root = root / "archive"
        output_root = root / "artifacts"
        source_root.mkdir(exist_ok=True)
        source = source_root / "lecture.aiff"
        if not source.exists():
            source.write_bytes(b"not-real-audio")
        config = PipelineConfig(
            input_path=source_root,
            output_root=output_root,
            cleanup_enabled=cleanup,
            cleanup_only=cleanup_only,
            retry_review=retry_review,
            render_only=render_only,
            force=force,
        )
        runner = PipelineRunner(config)
        transcribe_calls = []

        def transcribe(path):
            transcribe_calls.append(path)
            return {
                "text": "These are the faithfully transcribed spoken words",
                "raw_text": "These are the faithfully transcribed spoken words",
                "segments": [
                    {"start": 0.0, "end": 3.2, "text": "These are the faithfully"},
                    {"start": 3.4, "end": 7.0, "text": "transcribed spoken words"},
                ],
                "metadata": {"model": "Faster-Whisper large-v3"},
                "elapsed_seconds": 1.0,
            }

        render_calls = []

        def render(_source, _text, output_path, _metadata):
            render_calls.append(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"x" * 1_200)
            return output_path

        runner._transcribe = transcribe
        runner._render_docx = render
        runner.render_calls = render_calls
        if cleanup:
            runner.cleanup_client = FakeCleanupClient(needs_review=review)
        return runner, source, transcribe_calls

    def test_full_run_writes_immutable_artifacts_and_resume_skips(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, transcribe_calls = self._runner(root)
            try:
                self.assertEqual(runner.process_one(source), "verified")
                self.assertEqual(runner.process_one(source), "skipped")
                self.assertEqual(len(transcribe_calls), 1)
                paths = artifact_paths(
                    artifact_directory(source, runner.config.input_path, runner.config.output_root)
                )
                for key in ("manifest", "raw_text", "segments", "vtt", "srt", "clean_text", "cleanup", "qa", "docx"):
                    self.assertTrue(paths[key].is_file(), key)
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "verified")
                self.assertEqual(manifest["source"]["relative_path"], "lecture.aiff")
                self.assertEqual(len(manifest["source"]["sha256"]), 64)
                saved_raw = paths["raw_text"].read_text(encoding="utf-8")
                saved_clean = paths["clean_text"].read_text(encoding="utf-8")
                self.assertEqual(
                    manifest["stt"]["raw_sha256"], sha256_text(saved_raw)
                )
                self.assertEqual(
                    manifest["cleanup"]["input_sha256"], sha256_text(saved_raw)
                )
                self.assertEqual(
                    manifest["cleanup"]["output_sha256"], sha256_text(saved_clean)
                )
                self.assertEqual(
                    manifest["cleanup"]["model"], "@cf/zai-org/glm-4.7-flash"
                )
                self.assertNotIn("chunk_results", manifest["cleanup"])
                self.assertIn(
                    "chunk_results",
                    json.loads(paths["cleanup"].read_text(encoding="utf-8")),
                )
            finally:
                runner.close()

    def test_changed_source_invalidates_completed_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, transcribe_calls = self._runner(root, cleanup=False)
            try:
                self.assertEqual(runner.process_one(source), "verified")
                source.write_bytes(source.read_bytes() + b"changed")
                self.assertEqual(runner.process_one(source), "verified")
                self.assertEqual(len(transcribe_calls), 2)
            finally:
                runner.close()

    def test_changed_glossary_reuses_stt_but_reruns_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, transcribe_calls = self._runner(root)
            try:
                self.assertEqual(runner.process_one(source), "verified")
                self.assertEqual(runner.cleanup_client.calls, 1)
                runner.cleanup_client.glossary_sha256 = "updated-glossary"
                self.assertEqual(runner.process_one(source), "verified")
                self.assertEqual(len(transcribe_calls), 1)
                self.assertEqual(runner.cleanup_client.calls, 2)
            finally:
                runner.close()

    def test_cleanup_warning_becomes_needs_review_not_silent_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, _calls = self._runner(root, review=True)
            try:
                self.assertEqual(runner.process_one(source), "needs_review")
                self.assertEqual(runner.process_one(source), "needs_review")
                self.assertEqual(runner.cleanup_client.calls, 1)
                counts = runner.run()
                self.assertEqual(counts["needs_review"], 1)
                self.assertEqual(counts["skipped"], 0)
            finally:
                runner.close()

    def test_render_only_reuses_verified_text_but_rebuilds_docx(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, source, _calls = self._runner(root)
            try:
                self.assertEqual(first.process_one(source), "verified")
            finally:
                first.close()

            resumed, source, resumed_transcribe_calls = self._runner(
                root, render_only=True
            )
            try:
                self.assertEqual(resumed.process_one(source), "verified")
                self.assertEqual(resumed_transcribe_calls, [])
                self.assertEqual(resumed.cleanup_client.calls, 0)
                self.assertEqual(len(resumed.render_calls), 1)
            finally:
                resumed.close()

    def test_corrupt_cleaned_text_reruns_cleanup_without_retranscribing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, transcribe_calls = self._runner(root)
            try:
                self.assertEqual(runner.process_one(source), "verified")
                paths = artifact_paths(
                    artifact_directory(
                        source, runner.config.input_path, runner.config.output_root
                    )
                )
                paths["clean_text"].write_text("corrupt", encoding="utf-8")
                self.assertEqual(runner.process_one(source), "verified")
                self.assertEqual(len(transcribe_calls), 1)
                self.assertEqual(runner.cleanup_client.calls, 2)
            finally:
                runner.close()

    def test_changed_prompt_terms_invalidate_raw_transcription(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, transcribe_calls = self._runner(root, cleanup=False)
            try:
                self.assertEqual(runner.process_one(source), "verified")
                (runner.config.input_path / "special_words.txt").write_text(
                    "Gurdjieff\n", encoding="utf-8"
                )
                self.assertEqual(runner.process_one(source), "verified")
                self.assertEqual(len(transcribe_calls), 2)
            finally:
                runner.close()

    def test_force_bypasses_cleanup_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, source, _calls = self._runner(root)
            try:
                self.assertEqual(first.process_one(source), "verified")
            finally:
                first.close()

            forced, source, _calls = self._runner(root, force=True)
            try:
                self.assertEqual(forced.process_one(source), "verified")
                self.assertEqual(forced.cleanup_client.reuse_checkpoints, [False])
            finally:
                forced.close()

    def test_cleanup_only_reuses_raw_and_reruns_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, source, first_transcribe_calls = self._runner(root)
            try:
                self.assertEqual(first.process_one(source), "verified")
                self.assertEqual(len(first_transcribe_calls), 1)
            finally:
                first.close()

            resumed, source, resumed_transcribe_calls = self._runner(
                root, cleanup_only=True
            )
            try:
                self.assertEqual(resumed.process_one(source), "verified")
                self.assertEqual(resumed_transcribe_calls, [])
                self.assertEqual(resumed.cleanup_client.calls, 1)
                self.assertEqual(resumed.cleanup_client.reuse_checkpoints, [False])
            finally:
                resumed.close()

    def test_retry_review_reuses_raw_but_retries_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, source, first_transcribe_calls = self._runner(root, review=True)
            try:
                self.assertEqual(first.process_one(source), "needs_review")
                self.assertEqual(len(first_transcribe_calls), 1)
            finally:
                first.close()

            resumed, source, resumed_transcribe_calls = self._runner(
                root, retry_review=True
            )
            try:
                self.assertEqual(resumed.process_one(source), "verified")
                self.assertEqual(resumed_transcribe_calls, [])
                self.assertEqual(resumed.cleanup_client.calls, 1)
                self.assertEqual(resumed.cleanup_client.reuse_checkpoints, [False])
            finally:
                resumed.close()

    def test_transcription_exception_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, _calls = self._runner(root, cleanup=False)

            def fail(_source):
                raise RuntimeError("decoder failed")

            runner._transcribe = fail
            try:
                self.assertEqual(runner.process_one(source), "failed")
                paths = artifact_paths(
                    artifact_directory(source, runner.config.input_path, runner.config.output_root)
                )
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "failed")
                self.assertIn("decoder failed", manifest["error"])
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
