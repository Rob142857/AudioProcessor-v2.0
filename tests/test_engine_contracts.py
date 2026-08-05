import ast
import importlib.util
import json
import os
import queue
import sys
import tempfile
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

    def test_structured_return_options_are_keyword_only(self):
        names = [arg.arg for arg in self.function.args.kwonlyargs]
        self.assertEqual(
            ["return_details", "write_docx", "docx_output_path"],
            names,
        )
        defaults = [ast.literal_eval(value) for value in self.function.args.kw_defaults]
        self.assertEqual([False, True, None], defaults)

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


class _FakeFont:
    size = None
    italic = False
    color = types.SimpleNamespace(rgb=None)


class _FakeRun:
    bold = False
    font = _FakeFont()


class _FakeParagraph:
    def __init__(self):
        self.alignment = None
        self.runs = [_FakeRun()]


class _FakeDocument:
    def __init__(self, path=None):
        if path is not None and Path(path).read_bytes() != b"valid-docx":
            raise ValueError("invalid DOCX")

    def add_heading(self, *args, **kwargs):
        return _FakeParagraph()

    def add_paragraph(self, *args, **kwargs):
        return _FakeParagraph()

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
