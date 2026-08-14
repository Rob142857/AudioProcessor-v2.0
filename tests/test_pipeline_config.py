import unittest

from pipeline_config import (
    DEFAULT_STT_MODEL,
    PipelineConfigValidationError,
    resolve_pipeline_settings,
)


class PipelineConfigResolutionTests(unittest.TestCase):
    def test_default_is_parakeet_for_all_non_gui_callers(self):
        self.assertEqual("nvidia/parakeet-tdt-0.6b-v3", DEFAULT_STT_MODEL)
        self.assertEqual(DEFAULT_STT_MODEL, resolve_pipeline_settings(environment={}).stt_model)

    def test_precedence_and_source_logging_are_explicit(self):
        resolved = resolve_pipeline_settings(
            environment={"TRANSCRIBE_MODEL_NAME": "environment-model", "TRANSCRIPT_GLM_WORKERS": "9"},
            saved_settings={"whisper_model": "saved-model", "replace_mode": "skip"},
            cli_values={"stt_model": "cli-model"},
            gui_values={"stt_model": "gui-model", "glm_workers": 30},
        )
        self.assertEqual("gui-model", resolved.stt_model)
        self.assertEqual("skip", resolved.existing_docx_mode)
        self.assertEqual(30, resolved.glm_workers)
        self.assertEqual("gui", resolved.sources["stt_model"])
        self.assertIn("stt_model='gui-model' (gui)", resolved.startup_log())

    def test_legacy_aliases_are_read_but_conflicts_fail_loudly(self):
        self.assertEqual(
            "before",
            resolve_pipeline_settings(environment={}, saved_settings={"replace_mode": "before"}).existing_docx_mode,
        )
        with self.assertRaises(PipelineConfigValidationError):
            resolve_pipeline_settings(
                environment={},
                saved_settings={"replace_mode": "skip", "existing_docx_mode": "all"},
            )
