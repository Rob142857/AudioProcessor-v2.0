import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (REPO_ROOT / "run_full_pipeline.bat").read_text(encoding="utf-8")


class FullPipelineLauncherContractTests(unittest.TestCase):
    def test_enables_utf8_for_console_and_python(self):
        self.assertIn("chcp 65001", LAUNCHER)
        self.assertIn('set "PYTHONUTF8=1"', LAUNCHER)
        self.assertIn('set "PYTHONIOENCODING=utf-8"', LAUNCHER)

    def test_normal_transcription_requires_gpu(self):
        self.assertIn('set "DOCTOR_MODE=full"', LAUNCHER)
        self.assertIn('set "DOCTOR_GPU_ARG=--require-gpu"', LAUNCHER)
        self.assertIn('set "DOCTOR_MODE=transcribe"', LAUNCHER)
        self.assertIn(
            'pipeline_doctor.py --mode "%DOCTOR_MODE%" %DOCTOR_GPU_ARG%',
            LAUNCHER,
        )

    def test_non_transcription_modes_do_not_require_gpu(self):
        self.assertIn(':doctor_inventory', LAUNCHER)
        self.assertIn('set "DOCTOR_MODE=inventory"', LAUNCHER)
        self.assertIn(':doctor_cleanup', LAUNCHER)
        self.assertIn('set "DOCTOR_MODE=cleanup-only"', LAUNCHER)
        self.assertIn(':doctor_render', LAUNCHER)
        self.assertIn('set "DOCTOR_MODE=render-only"', LAUNCHER)

    def test_source_publication_is_default_with_stripped_opt_out(self):
        self.assertIn(
            'set "PUBLISH_SOURCE_DOCX=--publish-source-docx"', LAUNCHER
        )
        self.assertIn(
            'if /I "%~1"=="--no-publish-source-docx" goto disable_source_publish',
            LAUNCHER,
        )
        archive_call = next(
            line for line in LAUNCHER.splitlines() if "archive_pipeline.py" in line
        )
        self.assertIn("%PUBLISH_SOURCE_DOCX%", archive_call)
        self.assertNotIn("%*", archive_call)
        self.assertNotIn("--no-publish-source-docx", archive_call)

    def test_dry_run_never_requests_publication(self):
        inventory_block = LAUNCHER.split(":doctor_inventory", 1)[1].split(
            ":doctor_render", 1
        )[0]
        self.assertIn('set "PUBLISH_SOURCE_DOCX="', inventory_block)


if __name__ == "__main__":
    unittest.main()
