import ast
import importlib.util
import json
import os
import queue
import re
import sys
import tempfile
import threading
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import gui_transcribe
from gui_components import SUPPORTED_EXTS


class GuiEngineContractTests(unittest.TestCase):
    def setUp(self):
        gui_transcribe.STOP_FLAG.clear()

    def tearDown(self):
        gui_transcribe.STOP_FLAG.clear()

    def test_run_single_returns_explicit_success_and_failure(self):
        q = queue.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "lecture.docx"
            output.write_bytes(b"docx")
            fake_engine = types.SimpleNamespace(
                transcribe_file_simple_auto=lambda *args, **kwargs: str(output)
            )
            with mock.patch.dict(sys.modules, {"transcribe_optimised": fake_engine}):
                self.assertTrue(gui_transcribe._run_single("lecture.wav", tmp, q))

            failing_engine = types.SimpleNamespace(
                transcribe_file_simple_auto=mock.Mock(
                    side_effect=RuntimeError("decode failed")
                )
            )
            with mock.patch.dict(sys.modules, {"transcribe_optimised": failing_engine}):
                self.assertFalse(gui_transcribe._run_single("lecture.wav", tmp, q))

        messages = ""
        while not q.empty():
            messages += q.get_nowait()
        self.assertIn("Done ->", messages)
        self.assertIn("decode failed", messages)

    def test_batch_counts_boolean_failures(self):
        q = queue.Queue()
        with mock.patch.object(
            gui_transcribe,
            "_run_single",
            side_effect=[True, False, True],
        ):
            gui_transcribe._run_batch(["one.wav", "two.wav", "three.wav"], q)

        messages = ""
        while not q.empty():
            messages += q.get_nowait()
        self.assertIn("2 succeeded, 1 failed", messages)

    def test_settings_warn_on_invalid_json_and_save_atomically(self):
        previous_path = gui_transcribe.SETTINGS_PATH
        try:
            with tempfile.TemporaryDirectory() as tmp:
                settings_path = Path(tmp) / ".transcribe_settings.json"
                gui_transcribe.SETTINGS_PATH = str(settings_path)
                settings_path.write_text("{ invalid", encoding="utf-8")

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.assertEqual({}, gui_transcribe._load_settings())
                self.assertTrue(caught)

                expected = {"projects": {"C:/audio": {"recursive": 1}}}
                with warnings.catch_warnings(record=True) as save_warnings:
                    warnings.simplefilter("always")
                    self.assertTrue(gui_transcribe._save_settings(expected))
                self.assertTrue(save_warnings)
                self.assertEqual(expected, json.loads(settings_path.read_text("utf-8")))
                self.assertEqual(
                    "{ invalid",
                    Path(f"{settings_path}.invalid-backup").read_text("utf-8"),
                )
                self.assertFalse(Path(f"{settings_path}.tmp").exists())
        finally:
            gui_transcribe.SETTINGS_PATH = previous_path

    def test_new_audio_extensions_are_discoverable(self):
        self.assertTrue({".aiff", ".aif", ".3gp"}.issubset(SUPPORTED_EXTS))
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("one.aiff", "two.AIF", "three.3gp"):
                (Path(tmp) / name).write_bytes(b"")
            found = gui_transcribe._collect_files(tmp, False, "skip", "", queue.Queue())
        self.assertEqual(3, len(found))

    def test_replace_before_date_is_strict_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "lecture.wav"
            source.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
                gui_transcribe._should_process(
                    str(source), "before", "not-a-date"
                )
            with self.assertRaisesRegex(ValueError, "selection is invalid"):
                gui_transcribe._should_process(str(source), "surprise", "")

    def test_project_settings_canonicalize_paths_and_preserve_unknown_keys(self):
        previous_path = gui_transcribe.SETTINGS_PATH
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "Recordings"
                root.mkdir()
                settings_path = Path(tmp) / "settings.json"
                gui_transcribe.SETTINGS_PATH = str(settings_path)
                legacy_key = str(root).replace("\\", "/")
                settings_path.write_text(
                    json.dumps(
                        {
                            "projects": {
                                legacy_key: {
                                    "whisper_model": "faster-whisper-medium",
                                    "future_setting": "preserve-me",
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )

                loaded = gui_transcribe._load_project(str(root))
                self.assertEqual("preserve-me", loaded["future_setting"])
                gui_transcribe._save_project(
                    str(root), {"polished_pipeline": 1, "recursive": 1}
                )

                saved = json.loads(settings_path.read_text(encoding="utf-8"))
                key = gui_transcribe._project_key(str(root))
                self.assertEqual([key], list(saved["projects"]))
                self.assertEqual(
                    "preserve-me", saved["projects"][key]["future_setting"]
                )
                self.assertEqual(1, saved["projects"][key]["polished_pipeline"])
        finally:
            gui_transcribe.SETTINGS_PATH = previous_path

    def test_polished_gui_maps_settings_to_pipeline_config(self):
        q = queue.Queue()
        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        def fake_execute(config, *, cancel_check):
            captured["cancelled"] = cancel_check()
            return 3

        fake_archive = types.SimpleNamespace(
            PipelineConfig=FakeConfig,
            default_output_root=lambda source: source.parent / "Polished",
            execute_pipeline=fake_execute,
        )
        settings = {
            "whisper_model": "faster-whisper-large-v3",
            "recursive": 1,
            "replace_mode": "before",
            "replace_before_date": "2026-04-06",
            "force_reprocess": 0,
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sys.modules, {"archive_pipeline": fake_archive}
        ):
            source = Path(tmp) / "Recordings"
            source.mkdir()
            self.assertEqual(
                3,
                gui_transcribe._run_polished_pipeline(
                    str(source), settings, q
                ),
            )

        self.assertEqual("faster-whisper-large-v3", captured["stt_model"])
        self.assertTrue(captured["publish_source_docx"])
        self.assertTrue(captured["recursive"])
        self.assertEqual("before", captured["existing_docx_mode"])
        self.assertEqual("2026-04-06", captured["replace_before_date"])
        self.assertFalse(captured["force"])
        self.assertFalse(captured["existing_transcripts_only"])
        self.assertTrue(captured["retain_troubleshooting_artifacts"])
        self.assertFalse(captured["cancelled"])

    def test_existing_transcript_mode_maps_to_no_whisper_pipeline_contract(self):
        q = queue.Queue()
        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        def fake_execute(config, *, cancel_check):
            captured["executed"] = True
            return 0

        fake_archive = types.SimpleNamespace(
            PipelineConfig=FakeConfig,
            default_output_root=lambda source: source.parent / "Polished",
            execute_pipeline=fake_execute,
        )
        settings = {
            "existing_transcripts_only": 1,
            "whisper_model": "faster-whisper-large-v3",
            "recursive": 1,
            "replace_mode": "all",
            "replace_before_date": "",
            "force_reprocess": 1,
            "retain_troubleshooting_artifacts": 0,
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sys.modules, {"archive_pipeline": fake_archive}
        ):
            source = Path(tmp) / "Recordings"
            source.mkdir()
            self.assertEqual(
                0,
                gui_transcribe._run_polished_pipeline(str(source), settings, q),
            )

        self.assertTrue(captured["existing_transcripts_only"])
        self.assertTrue(captured["force"])
        self.assertFalse(captured["retain_troubleshooting_artifacts"])
        self.assertEqual("all", captured["existing_docx_mode"])
        self.assertTrue(captured["executed"])
        messages = ""
        while not q.empty():
            messages += q.get_nowait()
        self.assertIn("existing speech Word", messages)
        self.assertIn("Whisper and audio skipped", messages)

    def test_existing_transcript_mode_rejects_skip_existing(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined with Skip existing"):
            gui_transcribe._validate_polished_selection(
                {
                    "existing_transcripts_only": 1,
                    "replace_mode": "skip",
                    "replace_before_date": "",
                }
            )

    def test_existing_transcript_preflight_does_not_require_audio_stack(self):
        calls = []
        fake_check = types.SimpleNamespace(
            status="ok", name="Cleanup", detail="ready"
        )

        def fake_run_checks(**kwargs):
            calls.append(kwargs)
            return [fake_check]

        q = queue.Queue()
        with mock.patch.dict(
            sys.modules,
            {"pipeline_doctor": types.SimpleNamespace(run_checks=fake_run_checks)},
        ):
            self.assertTrue(
                gui_transcribe._run_polished_preflight(
                    q, existing_transcripts_only=True
                )
            )

        self.assertEqual(
            [
                {
                    "cleanup_required": True,
                    "mode": "cleanup-only",
                    "require_gpu": False,
                    "stt_model": None,
                }
            ],
            calls,
        )
        messages = ""
        while not q.empty():
            messages += q.get_nowait()
        self.assertIn("no GPU needed", messages)

    def test_polished_single_file_uses_a_dedicated_disjoint_output(self):
        captured = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_archive = types.SimpleNamespace(
            PipelineConfig=FakeConfig,
            default_output_root=mock.Mock(side_effect=AssertionError("folder only")),
            execute_pipeline=lambda _config, *, cancel_check: 0,
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sys.modules, {"archive_pipeline": fake_archive}
        ):
            archive = Path(tmp) / "1985 MW"
            archive.mkdir()
            source = archive / "0129 Visualization.mp3"
            source.write_bytes(b"")
            result = gui_transcribe._run_polished_pipeline(
                str(source),
                {
                    "replace_mode": "all",
                    "whisper_model": "faster-whisper-large-v3",
                },
                queue.Queue(),
            )

            self.assertEqual(0, result)
            output_root = captured["output_root"]
            self.assertEqual("0129 Visualization__mp3", output_root.name)
            self.assertEqual(
                "1985 MW - Polished Single Files", output_root.parent.name
            )
            self.assertNotEqual(archive.resolve(), output_root.parent.resolve())

    def test_gui_source_declares_default_polished_route_without_credentials(self):
        gui_source = (REPO_ROOT / "gui_transcribe.py").read_text(encoding="utf-8")
        component_source = (REPO_ROOT / "gui_components.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('ps.get("polished_pipeline", 1)', component_source)
        self.assertIn("protected GLM-4.7-Flash cleanup", component_source)
        self.assertIn(
            "Completed recordings, stages, and GLM chunks resume", component_source
        )
        self.assertIn("restarts from its beginning", component_source)
        self.assertIn("separate GLM Review copies", component_source)
        self.assertIn("force_reprocess", component_source)
        self.assertIn("Use existing Word transcripts (skip speech-to-text)", component_source)
        self.assertIn("source-adjacent Whisper DOCX", component_source)
        self.assertIn("no audio inference", component_source)
        self.assertIn(
            "Retain detailed troubleshooting logging", component_source
        )
        self.assertIn("retain_troubleshooting_artifacts", gui_source)
        self.assertIn("existing_transcripts_only", gui_source)
        self.assertNotIn("CF_ACCESS_CLIENT_SECRET", component_source)
        self.assertNotIn("CF_ACCESS_CLIENT_SECRET", gui_source)
        self.assertIn('root.protocol("WM_DELETE_WINDOW", on_window_close)', gui_source)

    def test_window_close_waits_for_an_active_worker(self):
        state = {"active": True, "close_pending": False}
        confirm = mock.Mock(return_value=True)
        request_stop = mock.Mock()
        destroy = mock.Mock()

        outcome = gui_transcribe._handle_window_close(
            state,
            confirm_close=confirm,
            request_stop=request_stop,
            destroy=destroy,
        )

        self.assertEqual("stopping", outcome)
        self.assertTrue(state["close_pending"])
        request_stop.assert_called_once_with()
        destroy.assert_not_called()
        self.assertEqual(
            "waiting",
            gui_transcribe._handle_window_close(
                state,
                confirm_close=confirm,
                request_stop=request_stop,
                destroy=destroy,
            ),
        )
        request_stop.assert_called_once_with()

    def test_window_close_is_immediate_only_while_idle(self):
        state = {"active": False, "close_pending": False}
        confirm = mock.Mock(return_value=True)
        request_stop = mock.Mock()
        destroy = mock.Mock()

        outcome = gui_transcribe._handle_window_close(
            state,
            confirm_close=confirm,
            request_stop=request_stop,
            destroy=destroy,
        )

        self.assertEqual("closed", outcome)
        destroy.assert_called_once_with()
        confirm.assert_not_called()
        request_stop.assert_not_called()

    def test_window_close_handles_worker_finishing_during_confirmation(self):
        state = {"active": True, "close_pending": False}

        def confirm():
            state["active"] = False
            return True

        request_stop = mock.Mock()
        destroy = mock.Mock()
        outcome = gui_transcribe._handle_window_close(
            state,
            confirm_close=confirm,
            request_stop=request_stop,
            destroy=destroy,
        )

        self.assertEqual("closed", outcome)
        destroy.assert_called_once_with()
        request_stop.assert_not_called()

    def test_gui_launcher_uses_the_pinned_environment(self):
        launcher = (REPO_ROOT / "run.bat").read_text(encoding="utf-8")
        self.assertIn('.venv\\Scripts\\python.exe', launcher)
        self.assertIn("install_geforce.ps1", launcher)
        self.assertIn("gui_transcribe.py --gui", launcher)
        self.assertNotIn("Activate.bat", launcher)
        self.assertNotIn("install.ps1", launcher)


class TranscriptionApiShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (REPO_ROOT / "transcribe_optimised.py").read_text(
            encoding="utf-8"
        )
        cls.tree = ast.parse(cls.source)
        cls.function = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "transcribe_file_simple_auto"
        )

    def load_top_level_function(self, name):
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        module = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"re": re, "threading": threading}
        exec(compile(module, "transcribe_optimised.py", "exec"), namespace)
        return namespace[name]

    def test_structured_return_options_are_keyword_only(self):
        names = [arg.arg for arg in self.function.args.kwonlyargs]
        self.assertEqual(
            ["return_details", "write_docx", "docx_output_path", "glossary_terms"],
            names,
        )
        defaults = [ast.literal_eval(value) for value in self.function.args.kw_defaults]
        self.assertEqual([False, True, None, None], defaults)

    def test_structured_result_contains_required_fields(self):
        for key in (
            '"text"',
            '"raw_text"',
            '"segments"',
            '"metadata"',
            '"docx_path"',
            '"source_path"',
            '"elapsed_seconds"',
        ):
            self.assertIn(key, self.source)

    def test_empty_transcription_is_not_published_as_placeholder(self):
        self.assertNotIn("[No speech detected or transcription failed]", self.source)
        self.assertIn("No speech text was produced", self.source)

    def test_context_blind_cleanup_is_not_the_default(self):
        self.assertIn('TRANSCRIBE_ARTIFACT_CLEANUP", "0"', self.source)
        self.assertIn("use_australian_spelling=False", self.source)

    def test_obsolete_fake_primer_is_neither_seeded_nor_deleted(self):
        primer = (
            "Now, as you know, we're looking at the biological basis or the "
            "biological manifestation of spiritual things, and this is something "
            "that requires careful attention because we need to understand how "
            "the invisible world of spirit connects with the visible world of matter."
        )
        normalise = self.load_top_level_function("_remove_prompt_artifacts")

        self.assertNotIn("_PRIMER_FRAGMENTS", self.source)
        self.assertNotIn("punctuation_primer", self.source)
        self.assertEqual(normalise(primer), primer)

    def test_prompt_artifact_normalizer_keeps_generic_safe_cleanup(self):
        normalise = self.load_top_level_function("_remove_prompt_artifacts")

        self.assertEqual(
            normalise(
                "  First   sentence, , with spaced . . punctuation.\r\n\r\n\r\n"
                " Second\tline.  "
            ),
            "First sentence, with spaced. punctuation.\n\nSecond line.",
        )

    def test_special_words_text_is_allow_listed_for_version_control(self):
        ignore_rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("!special_words.txt", ignore_rules.splitlines())

    def test_stop_waits_until_the_inference_thread_has_exited(self):
        join_after_stop = self.load_top_level_function(
            "_join_inference_thread_after_stop"
        )
        release = threading.Event()
        joined = threading.Event()
        inference = threading.Thread(target=release.wait)
        inference.start()

        waiter = threading.Thread(
            target=lambda: (
                join_after_stop(inference, poll_interval=0.01),
                joined.set(),
            )
        )
        waiter.start()
        self.assertFalse(joined.wait(0.03))
        release.set()
        self.assertTrue(joined.wait(1.0))
        waiter.join(timeout=1.0)
        inference.join(timeout=1.0)
        self.assertFalse(inference.is_alive())
        self.assertIn("daemon=False", self.source)


class _FakeFont:
    def __init__(self):
        self.name = None
        self.size = None
        self.italic = False
        self.color = types.SimpleNamespace(rgb=None)


class _FakeRun:
    def __init__(self, text=""):
        self.text = text
        self.bold = False
        self.font = _FakeFont()


class _FakeParagraphFormat:
    def __init__(self):
        self.alignment = None
        self.space_before = None
        self.space_after = None
        self.line_spacing = None
        self.keep_with_next = False


class _FakeParagraph:
    def __init__(self, text=""):
        self.text = text
        self.alignment = None
        self.paragraph_format = _FakeParagraphFormat()
        self.runs = [_FakeRun(text)] if text else []

    def add_run(self, text):
        self.text += text
        run = _FakeRun(text)
        self.runs.append(run)
        return run


class _FakeStyle:
    def __init__(self):
        self.font = _FakeFont()
        self.paragraph_format = _FakeParagraphFormat()


class _FakeSection:
    pass


class _FakeDocument:
    def __init__(self, path=None):
        self.sections = [_FakeSection()]
        self.styles = {
            name: _FakeStyle()
            for name in ("Normal", "Heading 1", "Heading 2", "Heading 3")
        }
        self.core_properties = types.SimpleNamespace(
            title=None,
            author=None,
            subject=None,
        )
        if path is not None and Path(path).read_bytes() != b"valid-docx":
            raise ValueError("invalid DOCX")

    def add_heading(self, *args, **kwargs):
        return _FakeParagraph()

    def add_paragraph(self, text="", *args, **kwargs):
        return _FakeParagraph(text)

    def save(self, path):
        Path(path).write_bytes(b"valid-docx")


def _load_converter_with_fake_docx():
    docx = types.ModuleType("docx")
    docx.Document = _FakeDocument
    docx_enum = types.ModuleType("docx.enum")
    docx_enum_text = types.ModuleType("docx.enum.text")
    docx_enum_text.WD_ALIGN_PARAGRAPH = types.SimpleNamespace(
        CENTER="center", JUSTIFY="justify"
    )
    docx_shared = types.ModuleType("docx.shared")
    docx_shared.RGBColor = lambda *values: values
    stubs = {
        "docx": docx,
        "docx.enum": docx_enum,
        "docx.enum.text": docx_enum_text,
        "docx.shared": docx_shared,
    }
    module_name = "_txt_to_docx_contract_test"
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "txt_to_docx.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class DocxOutputContractTests(unittest.TestCase):
    def test_explicit_output_path_is_created_and_validated(self):
        converter = _load_converter_with_fake_docx()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "1988" / "0202 Lecture.wav"
            destination = root / "exports" / "nested" / "lecture.docx"

            with mock.patch("builtins.print"):
                result = converter.convert_txt_to_docx_from_text(
                    "A faithful transcript.",
                    source,
                    use_australian_spelling=False,
                    output_path=destination,
                )

            self.assertEqual(destination, result)
            self.assertEqual(b"valid-docx", destination.read_bytes())
            self.assertEqual([], list(destination.parent.glob("*.tmp.docx")))


if __name__ == "__main__":
    unittest.main()
