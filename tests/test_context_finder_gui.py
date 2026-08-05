from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from docx import Document

import gui_transcribe
from context_finder import SearchResult, apply_boundary_selection
from context_finder import read_result_records
from context_finder_gui import (
    ContextFinderCancelled,
    ContextFinderJobConfig,
    default_context_output_path,
    load_context_finder_settings,
    operational_checkpoint_dir,
    run_context_finder_job,
    save_context_finder_settings,
    validate_job_config,
)


def write_docx(path: Path, paragraphs: list[str]) -> None:
    document = Document()
    for text in paragraphs:
        document.add_paragraph(text)
    document.save(path)


class ContextFinderSettingsTests(unittest.TestCase):
    def test_settings_merge_preserves_transcription_and_future_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".transcribe_settings.json"
            path.write_text(
                json.dumps(
                    {
                        "projects": {"library": {"recursive": 1}},
                        "context_finder": {
                            "last_folder": "old",
                            "future_option": "preserve me",
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual("old", load_context_finder_settings(path)["last_folder"])
            self.assertTrue(
                save_context_finder_settings(
                    {"last_folder": "new", "refine_with_glm": True}, path
                )
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(1, saved["projects"]["library"]["recursive"])
            self.assertEqual(
                "preserve me", saved["context_finder"]["future_option"]
            )
            self.assertEqual("new", saved["context_finder"]["last_folder"])

    def test_invalid_shared_settings_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("{ invalid", encoding="utf-8")
            with self.assertWarns(RuntimeWarning):
                self.assertFalse(save_context_finder_settings({"query": "awakening"}, path))
            self.assertEqual("{ invalid", path.read_text(encoding="utf-8"))


class ContextFinderConfigurationTests(unittest.TestCase):
    def test_default_output_is_beside_library_and_query_named(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "1985 MW"
            library.mkdir()
            output = default_context_output_path(library, "self remembrance")
            self.assertEqual(library.parent.resolve(), output.parent.resolve())
            self.assertEqual(
                "1985 MW - Context - self remembrance.docx", output.name
            )

    def test_existing_non_compilation_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            output = root / "important.docx"
            write_docx(output, ["Human-authored document."])
            with self.assertRaises(FileExistsError):
                validate_job_config(
                    ContextFinderJobConfig(library, "awakening", output)
                )


class ContextFinderWorkflowTests(unittest.TestCase):
    def test_deterministic_run_publishes_without_default_jsonl_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            write_docx(
                library / "lecture.docx",
                ["Before.", "Awakening must occur.", "After."],
            )
            output = root / "contexts.docx"
            updates = []
            outcome = run_context_finder_job(
                ContextFinderJobConfig(
                    library,
                    "awakening",
                    output,
                    refine_with_glm=False,
                    context_words_each_side=0,
                ),
                progress_callback=updates.append,
            )
            self.assertTrue(output.is_file())
            self.assertFalse(output.with_suffix(".jsonl").exists())
            self.assertEqual(1, outcome.occurrence_count)
            self.assertEqual(1, outcome.region_count)
            self.assertEqual(1, outcome.source_count)
            self.assertEqual(1, outcome.fallback_regions)
            self.assertIn("scan_complete", [update.phase for update in updates])

    def test_unavailable_glm_falls_back_clearly_and_still_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            write_docx(library / "lecture.docx", ["Awakening must occur."])
            output = root / "contexts.docx"
            outcome = run_context_finder_job(
                ContextFinderJobConfig(
                    library,
                    "awakening",
                    output,
                    context_words_each_side=0,
                ),
                refiner=None,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(0, outcome.refined_regions)
            self.assertEqual(1, outcome.fallback_regions)
            self.assertTrue(
                any("unavailable" in warning for warning in outcome.warnings)
            )

    def test_one_region_failure_does_not_discard_another_glm_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            write_docx(
                library / "lecture.docx",
                [
                    "Opening.",
                    "Awakening must occur.",
                    "Different theme.",
                    "Another transition.",
                    "Awakening returns.",
                ],
            )
            output = root / "contexts.docx"
            calls: list[int] = []
            checkpoint_details = {}

            def mixed_refiner(
                result: SearchResult,
                *,
                cancel_check=None,
                progress_callback=None,
                checkpoint_dir=None,
                retain_checkpoints=None,
            ) -> SearchResult:
                calls.append(len(result.regions))
                checkpoint_details["path"] = Path(checkpoint_dir)
                checkpoint_details["retain"] = retain_checkpoints
                first, second = result.regions
                if progress_callback:
                    progress_callback(1, 2, first, "fallback")
                refined = apply_boundary_selection(
                    second,
                    second.broad_start_paragraph,
                    second.broad_end_paragraph,
                    method="glm_sentence_boundary_v1",
                    model="@cf/zai-org/glm-4.7-flash",
                    confidence=0.98,
                )
                if progress_callback:
                    progress_callback(2, 2, refined, "refined")
                return result.with_regions((first, refined))

            outcome = run_context_finder_job(
                ContextFinderJobConfig(
                    library,
                    "awakening",
                    output,
                    context_words_each_side=0,
                ),
                refiner=mixed_refiner,
            )
            self.assertEqual([2], calls)
            self.assertEqual(1, outcome.refined_regions)
            self.assertEqual(1, outcome.fallback_regions)
            self.assertEqual(2, outcome.region_count)
            self.assertTrue(output.is_file())
            self.assertFalse(checkpoint_details["retain"])
            self.assertEqual(
                operational_checkpoint_dir(output), checkpoint_details["path"]
            )
            self.assertFalse(checkpoint_details["path"].exists())

    def test_retained_jsonl_resumes_validated_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            write_docx(library / "lecture.docx", ["Awakening must occur."])
            output = root / "contexts.docx"

            def successful_refiner(result: SearchResult, **_kwargs) -> SearchResult:
                region = result.regions[0]
                refined = apply_boundary_selection(
                    region,
                    region.broad_start_paragraph,
                    region.broad_end_paragraph,
                    method="glm_sentence_boundary_v1",
                    model="@cf/zai-org/glm-4.7-flash",
                )
                return result.with_regions((refined,))

            config = ContextFinderJobConfig(
                library,
                "awakening",
                output,
                keep_jsonl=True,
                context_words_each_side=0,
            )
            first = run_context_finder_job(config, refiner=successful_refiner)
            self.assertEqual(1, first.refined_regions)
            self.assertTrue(output.with_suffix(".jsonl").is_file())

            def should_not_run(_result, **_kwargs):
                raise AssertionError("validated resume selection should be reused")

            second = run_context_finder_job(config, refiner=should_not_run)
            self.assertEqual(1, second.resumed_regions)
            self.assertEqual(0, second.refined_regions)
            self.assertEqual(0, second.fallback_regions)

    def test_pre_cancelled_job_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            write_docx(library / "lecture.docx", ["Awakening must occur."])
            output = root / "contexts.docx"
            stopped = threading.Event()
            stopped.set()
            with self.assertRaises(ContextFinderCancelled):
                run_context_finder_job(
                    ContextFinderJobConfig(library, "awakening", output),
                    cancel_check=stopped.is_set,
                    refiner=None,
                )
            self.assertFalse(output.exists())

    def test_each_successful_glm_region_is_checkpointed_before_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            write_docx(
                library / "lecture.docx",
                [
                    "Awakening appears here.",
                    "A separate transition.",
                    "Another unrelated transition.",
                    "Awakening appears again.",
                ],
            )
            output = root / "contexts.docx"
            stopped = threading.Event()

            def refine_then_stop(result: SearchResult, **_kwargs) -> SearchResult:
                first, second = result.regions
                refined = apply_boundary_selection(
                    first,
                    first.broad_start_paragraph,
                    first.broad_end_paragraph,
                    method="glm_sentence_boundary_v1",
                    model="@cf/zai-org/glm-4.7-flash",
                )
                progress_callback = _kwargs.get("progress_callback")
                if progress_callback:
                    progress_callback(1, 2, refined, "refined")
                stopped.set()
                return result.with_regions((refined, second))

            with self.assertRaises(ContextFinderCancelled):
                run_context_finder_job(
                    ContextFinderJobConfig(
                        library,
                        "awakening",
                        output,
                        keep_jsonl=True,
                        context_words_each_side=0,
                    ),
                    cancel_check=stopped.is_set,
                    refiner=refine_then_stop,
                )

            records = output.with_suffix(".jsonl")
            self.assertTrue(records.is_file())
            checkpoint = read_result_records(records)
            methods = [region.selection.method for region in checkpoint.regions]
            self.assertEqual(1, methods.count("glm_sentence_boundary_v1"))
            self.assertEqual(1, methods.count("deterministic_context_window"))
            self.assertFalse(output.exists())


class ContextFinderLauncherTests(unittest.TestCase):
    def test_main_gui_launches_standalone_context_finder_without_shell(self):
        fake_process = object()
        with patch.object(gui_transcribe.subprocess, "Popen", return_value=fake_process) as popen:
            launched = gui_transcribe._launch_context_finder_process()
        self.assertIs(fake_process, launched)
        command = popen.call_args.args[0]
        self.assertEqual(gui_transcribe.sys.executable, command[0])
        self.assertTrue(command[1].endswith("context_finder_gui.py"))
        self.assertEqual(gui_transcribe.REPO_ROOT, popen.call_args.kwargs["cwd"])


if __name__ == "__main__":
    unittest.main()
