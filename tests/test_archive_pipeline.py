import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from archive_pipeline import (
    PipelineConfig,
    PipelineRunner,
    SOURCE_DOCX_PUBLICATION_REPORT,
    artifact_directory,
    artifact_paths,
    compact_stt_metadata,
    discover_audio,
    execute_pipeline,
    main as pipeline_main,
    parse_args,
    publish_source_docx_batch,
    sha256_text,
    source_publication_scope,
    validate_artifacts,
    validate_existing_docx_policy,
    validate_source_docx_target_collisions,
)
from stt_coverage import trailing_silence_tolerance_seconds
from pipeline_control import PipelineCancelledError


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

    def ensure_glossary(self, *, cancel_check=None):
        return SimpleNamespace(sha256=self.glossary_sha256)

    def cleanup_text(
        self,
        text,
        checkpoint_dir=None,
        *,
        reuse_checkpoints=True,
        cancel_check=None,
    ):
        self.calls += 1
        self.reuse_checkpoints.append(reuse_checkpoints)
        return FakeCleanupResult(text.rstrip() + ".", needs_review=self.needs_review)


class ArchivePipelineTests(unittest.TestCase):
    @staticmethod
    def clean_counts(discovered: int = 2) -> dict[str, int]:
        return {
            "discovered": discovered,
            "queued": 0,
            "skipped": 0,
            "verified": discovered,
            "needs_review": 0,
            "failed": 0,
            "cancelled": 0,
        }

    def test_source_docx_publication_defaults_off_and_cli_flag_opts_in(self):
        config = PipelineConfig(input_path=Path("archive"), output_root=Path("output"))

        self.assertFalse(config.publish_source_docx)
        self.assertFalse(config.existing_transcripts_only)
        self.assertTrue(config.retain_troubleshooting_artifacts)
        self.assertFalse(parse_args(["archive"]).publish_source_docx)
        self.assertTrue(
            parse_args(["archive", "--no-troubleshooting-logs"]).no_troubleshooting_logs
        )
        self.assertTrue(
            parse_args(["archive", "--publish-source-docx"]).publish_source_docx
        )
        for flag in (
            "--existing-transcripts-only",
            "--use-existing-docx",
            "--skip-stt",
        ):
            with self.subTest(flag=flag):
                self.assertTrue(
                    parse_args(["archive", flag]).existing_transcripts_only
                )

    def test_off_mode_compacts_terminology_bodies_to_digests(self):
        terminology = {
            "selector_version": "faster-whisper-hotwords-v1",
            "token_budget": 223,
            "applied": True,
            "reason": "selected",
            "hotwords": "esotericism enneagram",
            "selected_terms": ["esotericism", "enneagram"],
            "dropped_terms": ["a", "b"],
        }

        compact = compact_stt_metadata(
            {"terminology": terminology},
            retain_troubleshooting_artifacts=False,
        )["terminology"]

        for key in ("hotwords", "selected_terms", "dropped_terms"):
            self.assertNotIn(key, compact)
            self.assertEqual(len(compact[f"{key}_sha256"]), 64)
        self.assertEqual(compact["selected_terms_count"], 2)
        self.assertEqual(compact["dropped_terms_count"], 2)
        self.assertEqual(compact["selector_version"], terminology["selector_version"])

    def test_off_mode_suppresses_optional_event_log_but_keeps_runner_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "output"
            source_root.mkdir()
            runner = PipelineRunner(
                PipelineConfig(
                    input_path=source_root,
                    output_root=output_root,
                    retain_troubleshooting_artifacts=False,
                )
            )
            try:
                event_log = output_root / "job" / "run.jsonl"
                runner._append_event(event_log, "test_event", proof_sha256="abc123")
                self.assertFalse(event_log.exists())
                self.assertTrue((output_root / "pipeline.sqlite3").exists())
            finally:
                runner.close()

    def test_before_mode_rejects_missing_malformed_and_impossible_dates(self):
        invalid_dates = (
            None,
            "",
            "2026-8-05",
            "2026-08-5",
            "2026/08/05",
            "2026-02-30",
        )
        for value in invalid_dates:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "YYYY-MM-DD"
            ):
                validate_existing_docx_policy("before", value)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "empty-archive"
            output_root = root / "output"
            source_root.mkdir()
            with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
                discover_audio(
                    source_root,
                    output_root,
                    existing_docx_mode="before",
                    replace_before_date="2026-8-05",
                )
            self.assertFalse(output_root.exists())

    def test_existing_docx_skip_all_and_before_filters(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "output"
            source_root.mkdir()
            missing = source_root / "01 missing.mp3"
            old = source_root / "02 old.wav"
            recent = source_root / "03 recent.aiff"
            for source in (missing, old, recent):
                source.write_bytes(b"synthetic-audio-placeholder")

            old_docx = old.with_suffix(".docx")
            recent_docx = recent.with_suffix(".docx")
            old_docx.write_bytes(b"old transcript placeholder")
            recent_docx.write_bytes(b"recent transcript placeholder")
            old_time = datetime(2020, 6, 1, 12, 0).timestamp()
            recent_time = datetime(2024, 6, 1, 12, 0).timestamp()
            os.utime(old_docx, (old_time, old_time))
            os.utime(recent_docx, (recent_time, recent_time))

            selected_all = discover_audio(
                source_root,
                output_root,
                existing_docx_mode="all",
            )
            selected_skip = discover_audio(
                source_root,
                output_root,
                existing_docx_mode="skip",
            )
            selected_before = discover_audio(
                source_root,
                output_root,
                existing_docx_mode="before",
                replace_before_date="2022-01-01",
            )

            self.assertEqual(selected_all, [missing.resolve(), old.resolve(), recent.resolve()])
            self.assertEqual(selected_skip, [missing.resolve()])
            self.assertEqual(selected_before, [missing.resolve(), old.resolve()])

    def test_non_recursive_discovery_excludes_nested_recordings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "output"
            nested = source_root / "year" / "month"
            nested.mkdir(parents=True)
            direct = source_root / "direct.mp3"
            nested_audio = nested / "nested.aiff"
            direct.write_bytes(b"synthetic-audio-placeholder")
            nested_audio.write_bytes(b"synthetic-audio-placeholder")

            recursive = discover_audio(source_root, output_root, recursive=True)
            flat = discover_audio(source_root, output_root, recursive=False)

            self.assertEqual(recursive, [direct.resolve(), nested_audio.resolve()])
            self.assertEqual(flat, [direct.resolve()])

    def test_source_docx_collision_is_rejected_before_any_recording_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "output"
            source_root.mkdir()
            mp3 = source_root / "lecture.mp3"
            wav = source_root / "lecture.wav"
            mp3.write_bytes(b"synthetic-audio-placeholder")
            wav.write_bytes(b"synthetic-audio-placeholder")
            runner = PipelineRunner(
                PipelineConfig(
                    input_path=source_root,
                    output_root=output_root,
                    cleanup_enabled=False,
                    publish_source_docx=True,
                )
            )
            try:
                with mock.patch.object(runner, "process_one") as process_one:
                    with self.assertRaisesRegex(
                        ValueError, "source-adjacent DOCX target collisions"
                    ) as caught:
                        runner.run()
                process_one.assert_not_called()
            finally:
                runner.close()

            message = str(caught.exception)
            self.assertIn(str(mp3.resolve()), message)
            self.assertIn(str(wav.resolve()), message)
            self.assertIn(
                str(mp3.with_name("lecture - GLM Review.docx").resolve()),
                message,
            )

    def test_fresh_stt_rejects_raw_whisper_review_cross_role_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "output"
            source_root.mkdir()
            ordinary = source_root / "lecture.mp3"
            review_named = source_root / "lecture - GLM Review.wav"
            ordinary.write_bytes(b"synthetic-audio-placeholder")
            review_named.write_bytes(b"synthetic-audio-placeholder")
            runner = PipelineRunner(
                PipelineConfig(
                    input_path=source_root,
                    output_root=output_root,
                    cleanup_enabled=False,
                    publish_source_docx=True,
                )
            )
            try:
                with mock.patch.object(runner, "process_one") as process_one:
                    with self.assertRaisesRegex(
                        ValueError, "source-adjacent DOCX target collisions"
                    ) as caught:
                        runner.run()
                process_one.assert_not_called()
            finally:
                runner.close()

            message = str(caught.exception)
            collision = ordinary.with_name("lecture - GLM Review.docx").resolve()
            self.assertIn(str(collision), message)
            self.assertIn(f"GLM review: {ordinary.resolve()}", message)
            self.assertIn(f"raw Whisper: {review_named.resolve()}", message)

    def test_collision_validation_uses_the_already_filtered_selected_set(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "output"
            nested = source_root / "nested"
            nested.mkdir(parents=True)
            mp3 = nested / "lecture.mp3"
            wav = nested / "lecture.wav"
            mp3.write_bytes(b"synthetic-audio-placeholder")
            wav.write_bytes(b"synthetic-audio-placeholder")

            # Non-recursive selection excludes the nested collision.
            flat = discover_audio(source_root, output_root, recursive=False)
            validate_source_docx_target_collisions(flat, source_root)
            self.assertEqual([], flat)

            # Limit is applied before the runner's collision gate.
            limited = discover_audio(source_root, output_root, recursive=True)[:1]
            validate_source_docx_target_collisions(limited, source_root)
            self.assertEqual(1, len(limited))
            limited_runner = PipelineRunner(
                PipelineConfig(
                    input_path=source_root,
                    output_root=output_root,
                    cleanup_enabled=False,
                    publish_source_docx=True,
                    dry_run=True,
                    limit=1,
                )
            )
            try:
                with mock.patch("builtins.print"):
                    limited_counts = limited_runner.run()
            finally:
                limited_runner.close()
            self.assertEqual(1, limited_counts["discovered"])
            self.assertEqual(1, limited_counts["queued"])

            # Skip/before filter both formats together through their shared DOCX.
            target = mp3.with_suffix(".docx")
            target.write_bytes(b"existing transcript")
            recent_time = datetime(2024, 6, 1, 12, 0).timestamp()
            os.utime(target, (recent_time, recent_time))
            skipped = discover_audio(
                source_root,
                output_root,
                existing_docx_mode="skip",
            )
            before = discover_audio(
                source_root,
                output_root,
                existing_docx_mode="before",
                replace_before_date="2022-01-01",
            )
            validate_source_docx_target_collisions(skipped, source_root)
            validate_source_docx_target_collisions(before, source_root)
            self.assertEqual([], skipped)
            self.assertEqual([], before)

    def test_dry_run_has_no_processed_manifests_eligible_for_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "output"
            source_root.mkdir()
            selected = source_root / "01 selected.mp3"
            excluded = source_root / "02 excluded.wav"
            selected.write_bytes(b"synthetic-audio-placeholder")
            excluded.write_bytes(b"synthetic-audio-placeholder")
            excluded.with_suffix(".docx").write_bytes(b"existing transcript")
            config = PipelineConfig(
                input_path=source_root,
                output_root=output_root,
                dry_run=True,
                publish_source_docx=True,
                existing_docx_mode="skip",
            )
            publication_result = {
                "status": "suppressed",
                "blocking_conditions": ["queued", "dry_run"],
            }

            with mock.patch(
                "archive_pipeline.publish_source_docx_batch",
                return_value=publication_result,
            ) as publish_batch, mock.patch("builtins.print"):
                exit_code = execute_pipeline(config)

            self.assertEqual(exit_code, 0)
            publish_batch.assert_called_once()
            called_config, called_counts = publish_batch.call_args.args
            self.assertIs(called_config, config)
            self.assertEqual(called_counts["discovered"], 1)
            self.assertEqual(called_counts["queued"], 1)
            self.assertEqual(
                publish_batch.call_args.kwargs["manifest_paths"],
                (),
            )
            summary = json.loads(
                (output_root / "last-run-summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, summary["counts"]["publication_suppressed"])
            self.assertEqual(
                "suppressed", summary["source_docx_publication"]["status"]
            )

    def test_publication_failure_rewrites_final_run_summary_before_return(self):
        config = PipelineConfig(
            input_path=Path("archive"),
            output_root=Path("output"),
            publish_source_docx=True,
        )
        counts = self.clean_counts(discovered=1)
        runner = mock.Mock()
        runner.run.return_value = counts
        runner.selected_manifest_paths = (Path("output/job/manifest.json"),)
        runner.processed_manifest_paths = [Path("output/job/manifest.json")]

        with mock.patch(
            "archive_pipeline.PipelineRunner", return_value=runner
        ), mock.patch(
            "archive_pipeline.publish_source_docx_batch",
            side_effect=RuntimeError("Word target is locked"),
        ), mock.patch("builtins.print"):
            exit_code = execute_pipeline(config)

        self.assertEqual(1, exit_code)
        self.assertEqual(1, counts["publication_failed"])
        runner._write_summary.assert_called_once()
        summary_counts = runner._write_summary.call_args.args[0]
        publication = runner._write_summary.call_args.kwargs["publication"]
        self.assertEqual(1, summary_counts["publication_failed"])
        self.assertEqual("failed", publication["status"])
        self.assertIn("Word target is locked", publication["error"])
        runner.close.assert_called_once_with()

    def test_publication_cancellation_rewrites_final_run_summary_before_return(self):
        config = PipelineConfig(
            input_path=Path("archive"),
            output_root=Path("output"),
            publish_source_docx=True,
        )
        counts = self.clean_counts(discovered=1)
        runner = mock.Mock()
        runner.run.return_value = counts
        runner.selected_manifest_paths = (Path("output/job/manifest.json"),)
        runner.processed_manifest_paths = [Path("output/job/manifest.json")]

        with mock.patch(
            "archive_pipeline.PipelineRunner", return_value=runner
        ), mock.patch(
            "archive_pipeline.publish_source_docx_batch",
            side_effect=PipelineCancelledError("cancelled before commit"),
        ), mock.patch("builtins.print"):
            exit_code = execute_pipeline(config)

        self.assertEqual(1, exit_code)
        self.assertEqual(1, counts["cancelled"])
        self.assertEqual(1, counts["publication_suppressed"])
        runner._write_summary.assert_called_once()
        publication = runner._write_summary.call_args.kwargs["publication"]
        self.assertEqual("suppressed", publication["status"])
        self.assertEqual(["cancel_requested"], publication["blocking_conditions"])
        runner.close.assert_called_once_with()

    def test_cancelled_run_excludes_unvisited_stale_manifest_from_final_publication(self):
        config = PipelineConfig(
            input_path=Path("archive"),
            output_root=Path("output"),
            publish_source_docx=True,
        )
        counts = self.clean_counts(discovered=2)
        counts["verified"] = 1
        counts["cancelled"] = 1
        completed = Path("output/completed/manifest.json")
        stale_unvisited = Path("output/unvisited/manifest.json")
        runner = mock.Mock()
        runner.run.return_value = counts
        runner.selected_manifest_paths = (completed, stale_unvisited)
        runner.processed_manifest_paths = [completed]

        with mock.patch(
            "archive_pipeline.PipelineRunner", return_value=runner
        ), mock.patch(
            "archive_pipeline.publish_source_docx_batch",
            return_value={
                "status": "published",
                "planned": 1,
                "operations": {"create": 0, "replace": 0, "noop": 1},
            },
        ) as publish_batch, mock.patch("builtins.print"):
            exit_code = execute_pipeline(config)

        self.assertEqual(1, exit_code)
        self.assertEqual(
            publish_batch.call_args.kwargs["manifest_paths"], (completed,)
        )
        self.assertNotIn(
            stale_unvisited, publish_batch.call_args.kwargs["manifest_paths"]
        )
        runner.close.assert_called_once_with()

    def test_cancellation_stops_before_next_file_and_preserves_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "output"
            source_root.mkdir()
            first = source_root / "01 first.mp3"
            second = source_root / "02 second.mp3"
            first.write_bytes(b"synthetic-audio-placeholder")
            second.write_bytes(b"synthetic-audio-placeholder")
            cancelled = {"value": False}
            runner = PipelineRunner(
                PipelineConfig(
                    input_path=source_root,
                    output_root=output_root,
                    cleanup_enabled=False,
                ),
                cancel_check=lambda: cancelled["value"],
            )

            def transcribe(_source):
                return {
                    "text": "These are faithfully transcribed spoken words",
                    "raw_text": "These are faithfully transcribed spoken words",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 9.5,
                            "text": "These are faithfully transcribed spoken words",
                        }
                    ],
                    "metadata": {
                        "model": "Faster-Whisper large-v3",
                        "audio_duration_seconds": 10.0,
                    },
                    "elapsed_seconds": 1.0,
                }

            def render(_source, _text, output_path, _metadata):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"x" * 1_200)
                return output_path

            runner._transcribe = transcribe
            runner._render_docx = render
            process_one = runner.process_one

            def process_then_cancel(source):
                status = process_one(source)
                cancelled["value"] = True
                return status

            runner.process_one = process_then_cancel
            try:
                counts = runner.run()
            finally:
                runner.close()

            first_paths = artifact_paths(
                artifact_directory(first, source_root, output_root)
            )
            second_paths = artifact_paths(
                artifact_directory(second, source_root, output_root)
            )
            self.assertEqual(counts["discovered"], 2)
            self.assertEqual(counts["verified"], 1)
            self.assertEqual(counts["cancelled"], 1)
            self.assertEqual(counts["failed"], 0)
            self.assertTrue(first_paths["manifest"].is_file())
            self.assertTrue(first_paths["raw_text"].is_file())
            self.assertTrue(first_paths["docx"].is_file())
            first_manifest = json.loads(
                first_paths["manifest"].read_text(encoding="utf-8")
            )
            self.assertEqual(first_manifest["status"], "verified")
            self.assertFalse(second_paths["manifest"].exists())

    def test_completed_job_is_published_before_a_later_batch_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "output"
            source_root.mkdir()
            first = source_root / "01 first.mp3"
            second = source_root / "02 second.mp3"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            cancelled = {"value": False}
            runner = PipelineRunner(
                PipelineConfig(
                    input_path=source_root,
                    output_root=output_root,
                    publish_source_docx=True,
                ),
                cancel_check=lambda: cancelled["value"],
            )
            runner.cleanup_client = FakeCleanupClient()

            def finish_first(source):
                manifest = artifact_paths(
                    artifact_directory(source, source_root, output_root)
                )["manifest"]
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(
                    json.dumps({"status": "verified"}), encoding="utf-8"
                )
                cancelled["value"] = True
                return "verified"

            runner.process_one = finish_first
            report = {
                "status": "published",
                "operations": {"create": 1, "replace": 0, "noop": 0},
            }
            try:
                with mock.patch(
                    "archive_pipeline.publish_source_docx_batch",
                    return_value=report,
                ) as publish:
                    counts = runner.run()
            finally:
                runner.close()

            self.assertEqual(counts["verified"], 1)
            self.assertEqual(counts["cancelled"], 1)
            publish.assert_called_once()
            self.assertEqual(
                publish.call_args.kwargs["manifest_paths"],
                (
                    artifact_paths(
                        artifact_directory(first, source_root, output_root)
                    )["manifest"],
                ),
            )

    def test_stage_cancellation_after_transcription_preserves_raw_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, _calls = self._runner(root)
            cancelled = {"value": False}
            runner.cancel_check = lambda: cancelled["value"]
            transcribe = runner._transcribe

            def transcribe_then_cancel(source_path):
                result = transcribe(source_path)
                cancelled["value"] = True
                return result

            runner._transcribe = transcribe_then_cancel
            try:
                status = runner.process_one(source)
            finally:
                runner.close()

            paths = artifact_paths(
                artifact_directory(source, runner.config.input_path, runner.config.output_root)
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(status, "cancelled")
            self.assertEqual(manifest["status"], "cancelled")
            self.assertIn("completed checkpoints preserved", manifest["error"])
            self.assertTrue(paths["raw_text"].is_file())
            self.assertTrue(paths["segments"].is_file())
            self.assertEqual(runner.cleanup_client.calls, 0)
            self.assertEqual(
                runner.render_calls,
                [paths["whisper_docx"]],
            )

    def test_cleanup_preflight_receives_callback_and_stops_before_first_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, _source, transcribe_calls = self._runner(root)
            cancelled = {"value": False}
            observed_callbacks = []
            original_ensure = runner.cleanup_client.ensure_glossary

            def ensure_then_cancel(*, cancel_check=None):
                observed_callbacks.append(cancel_check)
                result = original_ensure(cancel_check=cancel_check)
                cancelled["value"] = True
                return result

            runner.cancel_check = lambda: cancelled["value"]
            runner.cleanup_client.ensure_glossary = ensure_then_cancel
            try:
                counts = runner.run()
            finally:
                runner.close()

            self.assertEqual(len(observed_callbacks), 1)
            self.assertIs(observed_callbacks[0], runner.cancel_check)
            self.assertEqual(counts["cancelled"], 1)
            self.assertEqual(counts["verified"], 0)
            self.assertEqual(transcribe_calls, [])

    def test_dedicated_cleanup_stop_is_classified_cancelled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, _calls = self._runner(root)

            def stop_cleanup(*_args, **_kwargs):
                raise PipelineCancelledError("test cleanup stop")

            runner.cleanup_client.cleanup_text = stop_cleanup
            try:
                status = runner.process_one(source)
            finally:
                runner.close()

            paths = artifact_paths(
                artifact_directory(source, runner.config.input_path, runner.config.output_root)
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(status, "cancelled")
            self.assertEqual(manifest["status"], "cancelled")
            self.assertIn("test cleanup stop", manifest["error"])
            self.assertTrue(paths["raw_text"].is_file())
            self.assertEqual(
                runner.render_calls,
                [paths["whisper_docx"]],
            )

    def test_publication_cancellation_stops_before_planning_or_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "output"
            source_root.mkdir()
            output_root.mkdir()
            config = PipelineConfig(
                input_path=source_root,
                output_root=output_root,
                publish_source_docx=True,
            )
            manifest_path = output_root / "job" / "manifest.json"
            manifest_path.parent.mkdir()
            manifest_path.write_text(
                json.dumps({"status": "verified"}), encoding="utf-8"
            )

            with mock.patch(
                "legacy_docx_replace.plan_legacy_docx_replacements"
            ) as planner, mock.patch(
                "legacy_docx_replace.apply_legacy_docx_replacements"
            ) as apply_batch, self.assertRaises(PipelineCancelledError):
                publish_source_docx_batch(
                    config,
                    self.clean_counts(discovered=1),
                    cancel_check=lambda: True,
                )

            planner.assert_not_called()
            apply_batch.assert_not_called()
            report = json.loads(
                (output_root / SOURCE_DOCX_PUBLICATION_REPORT).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "suppressed")
            self.assertEqual(report["blocking_conditions"], ["cancel_requested"])

    def test_qa_requires_nonempty_timestamp_segments(self):
        qa = validate_artifacts(
            "faithfully spoken words",
            "faithfully spoken words",
            [],
            False,
            audio_duration_seconds=60.0,
        )

        self.assertEqual(qa["status"], "needs_review")
        self.assertIn("STT segment list is empty", qa["reasons"])
        self.assertEqual(qa["stt_coverage"]["segment_count"], 0)

    def test_qa_enforces_documented_trailing_silence_tolerance(self):
        duration = 3_600.0
        tolerance = trailing_silence_tolerance_seconds(duration)
        passing = validate_artifacts(
            "faithfully spoken words",
            "faithfully spoken words",
            [{"start": 0.0, "end": duration - tolerance, "text": "spoken words"}],
            False,
            audio_duration_seconds=duration,
        )
        truncated = validate_artifacts(
            "faithfully spoken words",
            "faithfully spoken words",
            [
                {
                    "start": 0.0,
                    "end": duration - tolerance - 0.1,
                    "text": "spoken words",
                }
            ],
            False,
            audio_duration_seconds=duration,
        )

        self.assertEqual(tolerance, 120.0)
        self.assertEqual(passing["status"], "passed")
        self.assertEqual(truncated["status"], "needs_review")
        self.assertTrue(
            any("trailing-silence tolerance" in reason for reason in truncated["reasons"])
        )

    def test_qa_requires_audio_duration_to_prove_coverage(self):
        qa = validate_artifacts(
            "faithfully spoken words",
            "faithfully spoken words",
            [{"start": 0.0, "end": 20.0, "text": "spoken words"}],
            False,
        )

        self.assertEqual(qa["status"], "needs_review")
        self.assertTrue(
            any("audio duration is unavailable" in reason for reason in qa["reasons"])
        )

    def test_publication_rejects_nested_output_before_creating_runner_state(self):
        with tempfile.TemporaryDirectory() as folder:
            source_root = Path(folder) / "archive"
            source_root.mkdir()
            nested_output = source_root / "generated"
            config = PipelineConfig(
                input_path=source_root,
                output_root=nested_output,
                publish_source_docx=True,
            )

            with self.assertRaisesRegex(ValueError, "separate, non-nested"):
                PipelineRunner(config)

            self.assertFalse(nested_output.exists())

    def test_single_file_publication_scope_is_its_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "archive" / "lecture.mp3"
            source.parent.mkdir()
            source.write_bytes(b"audio")

            self.assertEqual(source_publication_scope(source), source.parent.resolve())

    def test_single_file_publication_rejects_unrelated_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "archive" / "lecture.mp3"
            source.parent.mkdir()
            source.write_bytes(b"audio")
            output_root = root / "generated"
            output_root.mkdir()
            (output_root / "stale-manifest.json").write_text("{}", encoding="utf-8")
            config = PipelineConfig(
                input_path=source,
                output_root=output_root,
                publish_source_docx=True,
            )

            with self.assertRaisesRegex(ValueError, "same source manifest"):
                PipelineRunner(config)

            self.assertFalse((output_root / "pipeline.sqlite3").exists())

    def test_single_file_publication_allows_resume_for_same_source_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "archive" / "lecture.mp3"
            source.parent.mkdir()
            source.write_bytes(b"audio")
            output_root = root / "generated"
            job = output_root / "lecture__mp3"
            job.mkdir(parents=True)
            (job / "manifest.json").write_text(
                json.dumps({"source": {"path": str(source.resolve())}}),
                encoding="utf-8",
            )
            runner = PipelineRunner(
                PipelineConfig(
                    input_path=source,
                    output_root=output_root,
                    publish_source_docx=True,
                )
            )
            runner.close()

            self.assertTrue((output_root / "pipeline.sqlite3").is_file())

    def test_clean_run_invokes_one_create_replace_publication_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "generated"
            source_root.mkdir()
            output_root.mkdir()
            config = PipelineConfig(
                input_path=source_root,
                output_root=output_root,
                publish_source_docx=True,
            )
            manifest_path = output_root / "job" / "manifest.json"
            manifest_path.parent.mkdir()
            manifest_path.write_text(
                json.dumps({"status": "verified"}), encoding="utf-8"
            )
            plan = SimpleNamespace(
                items=(
                    SimpleNamespace(operation="create"),
                    SimpleNamespace(operation="replace"),
                ),
                plan_sha256="a" * 64,
                to_dict=lambda: {"count": 2, "plan_sha256": "a" * 64},
            )
            targets = (
                source_root / "0122 One.docx",
                source_root / "0129 Two.docx",
            )
            report_path = output_root / SOURCE_DOCX_PUBLICATION_REPORT
            pre_apply_reports = []

            def apply_after_reading_plan(*_args, **_kwargs):
                pre_apply_reports.append(
                    json.loads(report_path.read_text(encoding="utf-8"))
                )
                return targets

            with mock.patch(
                "legacy_docx_replace.plan_legacy_docx_replacements",
                return_value=plan,
            ) as planner, mock.patch(
                "legacy_docx_replace.apply_legacy_docx_replacements",
                side_effect=apply_after_reading_plan,
            ) as apply_batch:
                report = publish_source_docx_batch(
                    config,
                    self.clean_counts(),
                    now=datetime(2026, 8, 5, 14, 30, 45),
                )

            expected_backup = (
                output_root.resolve()
                / "publication-backups"
                / "20260805-143045-000000"
            )
            planner.assert_called_once_with(
                output_root.resolve(),
                source_root.resolve(),
                manifest_paths=[output_root.resolve() / "job" / "manifest.json"],
            )
            apply_batch.assert_called_once_with(
                plan,
                expected_scope_root=source_root.resolve(),
                backup_root=expected_backup,
                confirm=True,
                expected_count=2,
            )
            self.assertIsNotNone(report)
            self.assertEqual(report["status"], "published")
            self.assertEqual(report["operations"], {"create": 1, "replace": 1})
            self.assertEqual(pre_apply_reports[0]["status"], "planned")
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "published")
            self.assertEqual(saved["published"], [str(path) for path in targets])
            immutable_report = (
                output_root
                / "source-docx-publication-20260805-143045-000000.json"
            )
            self.assertEqual(
                json.loads(immutable_report.read_text(encoding="utf-8")),
                saved,
            )

    def test_main_publishes_only_after_runner_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "generated"
            source_root.mkdir()
            events = []
            counts = self.clean_counts(discovered=1)
            selected_manifests = (output_root / "selected__mp3" / "manifest.json",)

            with mock.patch("archive_pipeline.PipelineRunner") as runner_type, mock.patch(
                "archive_pipeline.publish_source_docx_batch"
            ) as publish_batch, mock.patch("builtins.print"):
                runner = runner_type.return_value
                runner.selected_manifest_paths = selected_manifests
                runner.processed_manifest_paths = list(selected_manifests)
                runner.run.side_effect = lambda: events.append("run") or counts
                publish_batch.side_effect = (
                    lambda _config, _counts, **_kwargs: events.append("publish")
                    or {"status": "published", "planned": 1}
                )
                exit_code = pipeline_main(
                    [
                        str(source_root),
                        "--output",
                        str(output_root),
                        "--publish-source-docx",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(events, ["run", "publish"])
            publish_config, publish_counts = publish_batch.call_args.args
            self.assertTrue(publish_config.publish_source_docx)
            self.assertIs(publish_counts, counts)
            self.assertEqual(
                publish_batch.call_args.kwargs["manifest_paths"],
                selected_manifests,
            )
            runner.close.assert_called_once_with()

    def test_failed_or_cancelled_job_is_not_published(self):
        for blocking_status in ("failed", "cancelled"):
            with self.subTest(blocking_status=blocking_status), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source_root = root / "archive"
                output_root = root / "generated"
                source_root.mkdir()
                output_root.mkdir()
                manifest_path = output_root / "job" / "manifest.json"
                manifest_path.parent.mkdir()
                manifest_path.write_text(
                    json.dumps({"status": blocking_status}), encoding="utf-8"
                )
                config = PipelineConfig(
                    input_path=source_root,
                    output_root=output_root,
                    publish_source_docx=True,
                )
                counts = self.clean_counts()
                counts["verified"] -= 1
                counts[blocking_status] = 1

                with mock.patch(
                    "legacy_docx_replace.plan_legacy_docx_replacements"
                ) as planner, mock.patch(
                    "legacy_docx_replace.apply_legacy_docx_replacements"
                ) as apply_batch:
                    report = publish_source_docx_batch(config, counts)

                planner.assert_not_called()
                apply_batch.assert_not_called()
                self.assertEqual(report["status"], "suppressed")
                self.assertIn(
                    "no_completed_review_documents", report["blocking_conditions"]
                )

    def test_needs_review_job_is_published_for_human_checking(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "generated"
            source_root.mkdir()
            manifest_path = output_root / "job" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps({"status": "needs_review"}), encoding="utf-8"
            )
            config = PipelineConfig(
                input_path=source_root,
                output_root=output_root,
                publish_source_docx=True,
            )
            plan = SimpleNamespace(
                items=(SimpleNamespace(operation="create"),),
                plan_sha256="b" * 64,
                to_dict=lambda: {"count": 1, "plan_sha256": "b" * 64},
            )
            target = source_root / "lecture - GLM Review.docx"
            with mock.patch(
                "legacy_docx_replace.plan_legacy_docx_replacements",
                return_value=plan,
            ) as planner, mock.patch(
                "legacy_docx_replace.apply_legacy_docx_replacements",
                return_value=(target,),
            ):
                counts = self.clean_counts(discovered=1)
                counts["verified"] = 0
                counts["needs_review"] = 1
                report = publish_source_docx_batch(config, counts)

            self.assertEqual(report["status"], "published")
            planner.assert_called_once_with(
                output_root.resolve(),
                source_root.resolve(),
                manifest_paths=[output_root.resolve() / "job" / "manifest.json"],
            )

    def test_dry_run_suppresses_source_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "generated"
            source_root.mkdir()
            output_root.mkdir()
            config = PipelineConfig(
                input_path=source_root,
                output_root=output_root,
                dry_run=True,
                publish_source_docx=True,
            )

            with mock.patch(
                "legacy_docx_replace.plan_legacy_docx_replacements"
            ) as planner, mock.patch(
                "legacy_docx_replace.apply_legacy_docx_replacements"
            ) as apply_batch:
                report = publish_source_docx_batch(config, self.clean_counts())

            planner.assert_not_called()
            apply_batch.assert_not_called()
            self.assertEqual(report["status"], "suppressed")
            self.assertIn("dry_run", report["blocking_conditions"])

    def test_disabled_cleanup_suppresses_source_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "generated"
            source_root.mkdir()
            output_root.mkdir()
            config = PipelineConfig(
                input_path=source_root,
                output_root=output_root,
                cleanup_enabled=False,
                publish_source_docx=True,
            )

            with mock.patch(
                "legacy_docx_replace.plan_legacy_docx_replacements"
            ) as planner:
                report = publish_source_docx_batch(config, self.clean_counts())

            planner.assert_not_called()
            self.assertEqual(report["status"], "suppressed")
            self.assertIn("cleanup_disabled", report["blocking_conditions"])

    def test_limited_run_suppresses_source_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "archive"
            output_root = root / "generated"
            source_root.mkdir()
            output_root.mkdir()
            config = PipelineConfig(
                input_path=source_root,
                output_root=output_root,
                limit=1,
                publish_source_docx=True,
            )

            with mock.patch(
                "legacy_docx_replace.plan_legacy_docx_replacements"
            ) as planner:
                report = publish_source_docx_batch(
                    config,
                    self.clean_counts(discovered=1),
                )

            planner.assert_not_called()
            self.assertEqual(report["status"], "suppressed")
            self.assertIn("limited_run", report["blocking_conditions"])

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
                "metadata": {
                    "model": "Faster-Whisper large-v3",
                    "audio_duration_seconds": 7.5,
                },
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
                for key in (
                    "manifest",
                    "raw_text",
                    "segments",
                    "vtt",
                    "srt",
                    "clean_text",
                    "cleanup",
                    "qa",
                    "publication",
                    "docx",
                ):
                    self.assertTrue(paths[key].is_file(), key)
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "verified")
                self.assertEqual(manifest["qa"]["stt_coverage"]["status"], "passed")
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
                publication = json.loads(
                    paths["publication"].read_text(encoding="utf-8")
                )
                self.assertEqual(publication["document_state"], "verified")
                self.assertEqual(
                    publication["search_ingestion_state"], "not_published"
                )
                self.assertEqual(publication["metadata"]["speaker"], "Dr Philip Groves")
                self.assertEqual(publication["metadata"]["nature"], "Lecture")
                self.assertEqual(publication["source"]["relative_path"], "lecture.aiff")
                self.assertEqual(publication["content"]["sha256"], sha256_text(saved_clean))
                self.assertEqual(
                    manifest["artifacts"]["publication"], str(paths["publication"])
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

    def test_changed_glossary_invalidates_hotword_bound_stt_and_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, transcribe_calls = self._runner(root)
            try:
                self.assertEqual(runner.process_one(source), "verified")
                self.assertEqual(runner.cleanup_client.calls, 1)
                runner.cleanup_client.glossary_sha256 = "updated-glossary"
                self.assertEqual(runner.process_one(source), "verified")
                # The pinned glossary also drives Whisper hotwords, so a glossary
                # version change deliberately invalidates the raw STT signature.
                self.assertEqual(len(transcribe_calls), 2)
                self.assertEqual(runner.cleanup_client.calls, 2)
            finally:
                runner.close()

    def test_old_verified_manifest_without_coverage_is_revalidated_not_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner, source, transcribe_calls = self._runner(root)
            try:
                self.assertEqual(runner.process_one(source), "verified")
                paths = artifact_paths(
                    artifact_directory(
                        source,
                        runner.config.input_path,
                        runner.config.output_root,
                    )
                )
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                manifest["qa"].pop("stt_coverage")
                paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

                self.assertEqual(runner.process_one(source), "verified")
                self.assertEqual(len(transcribe_calls), 1)
                refreshed = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                self.assertEqual(refreshed["qa"]["stt_coverage"]["status"], "passed")
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

    def test_whisper_prompt_uses_deterministic_publication_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "1985 MW"
            source_root.mkdir()
            source = source_root / "0129 visualization exercise (incomplete).mp3"
            source.write_bytes(b"not-real-audio")
            runner = PipelineRunner(
                PipelineConfig(input_path=source_root, output_root=root / "output")
            )
            try:
                prompt, provenance = runner._effective_initial_prompt(source)
            finally:
                runner.close()

            self.assertEqual(provenance, "publication-metadata")
            self.assertIn("Dr Philip Groves", prompt)
            self.assertIn("Visualization Exercise (Incomplete)", prompt)

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
