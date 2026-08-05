import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = (REPO_ROOT / "install_geforce.ps1").read_text(encoding="utf-8")


class GeForceInstallerContractTests(unittest.TestCase):
    def test_is_local_and_does_not_mutate_git_or_preload_models(self):
        lowered = INSTALLER.casefold()
        self.assertNotIn("invoke-restmethod", lowered)
        self.assertNotIn("irm ", lowered)
        self.assertNotIn("winget", lowered)
        self.assertNotIn("git pull", lowered)
        self.assertNotIn("git clone", lowered)
        self.assertNotIn("preload_models.py", lowered)

    def test_pins_python_torch_and_packaging_lane(self):
        self.assertIn('$PinnedPython = "3.12"', INSTALLER)
        self.assertIn('$PinnedTorch = "2.6.0+cu124"', INSTALLER)
        self.assertIn("https://download.pytorch.org/whl/cu124", INSTALLER)
        self.assertIn('"pip==$PinnedPip"', INSTALLER)

    def test_python_selection_falls_back_to_exact_per_user_python312(self):
        launcher_probe = INSTALLER.index('PrefixArguments @("-$PinnedPython")')
        fallback_probe = INSTALLER.index(
            '"Programs\\Python\\Python312\\python.exe"'
        )
        create_venv = INSTALLER.index(
            "Invoke-Native -FilePath $BasePythonFilePath"
        )
        self.assertLess(launcher_probe, fallback_probe)
        self.assertLess(fallback_probe, create_venv)
        self.assertIn("$BasePythonPrefixArguments", INSTALLER)

    def test_installs_torch_before_application_requirements(self):
        torch_step = INSTALLER.index('"torch==$PinnedTorch"')
        requirements_step = INSTALLER.index('"--requirement", $RequirementsPath')
        self.assertLess(torch_step, requirements_step)

    def test_checks_native_exit_codes_and_runs_doctor(self):
        self.assertIn("if ($exitCode -ne 0)", INSTALLER)
        self.assertIn('"-m", "pip", "check"', INSTALLER)
        self.assertIn('"--mode", "transcribe", "--require-gpu"', INSTALLER)

    def test_recreate_guard_rejects_reparse_points(self):
        self.assertIn("[System.IO.FileAttributes]::ReparsePoint", INSTALLER)
        self.assertIn("Remove-LocalVenv", INSTALLER)


if __name__ == "__main__":
    unittest.main()
