import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pipeline_doctor


class FakeCuda:
    def __init__(
        self,
        *,
        available=True,
        capability=(6, 1),
        architectures=None,
    ):
        self.available = available
        self.capability = capability
        self.architectures = architectures or ["sm_61", "sm_75"]

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def get_device_name(self, _index):
        return "NVIDIA GeForce GTX 1070 Ti"

    def get_device_capability(self, _index):
        return self.capability

    def get_device_properties(self, _index):
        return SimpleNamespace(total_memory=8 * 1024**3)

    def get_arch_list(self):
        return self.architectures


def fake_torch(*, available=True, architectures=None, built_cuda="12.4"):
    return SimpleNamespace(
        version=SimpleNamespace(cuda=built_cuda),
        cuda=FakeCuda(available=available, architectures=architectures),
    )


class FakeCTranslate2:
    def __init__(self, *, devices=1, compute_types=None):
        self.devices = devices
        self.compute_types = compute_types or {"float32", "int8", "int8_float32"}

    def get_cuda_device_count(self):
        return self.devices

    def get_supported_compute_types(self, device, index):
        if device != "cuda" or index != 0:
            raise AssertionError("doctor queried the wrong CUDA device")
        return self.compute_types


class PipelineDoctorTests(unittest.TestCase):
    def test_mode_requirements_skip_unrelated_stages(self):
        full = pipeline_doctor.requirements_for_mode("full", cleanup_required=True)
        local = pipeline_doctor.requirements_for_mode(
            "transcribe", cleanup_required=True
        )
        cleanup = pipeline_doctor.requirements_for_mode(
            "cleanup-only", cleanup_required=True
        )
        render = pipeline_doctor.requirements_for_mode(
            "render-only", cleanup_required=True
        )
        inventory = pipeline_doctor.requirements_for_mode(
            "inventory", cleanup_required=True
        )

        self.assertEqual(full, pipeline_doctor.ModeRequirements(True, True, True))
        self.assertEqual(local, pipeline_doctor.ModeRequirements(True, False, True))
        self.assertEqual(cleanup, pipeline_doctor.ModeRequirements(False, True, True))
        self.assertEqual(render, pipeline_doctor.ModeRequirements(False, False, True))
        self.assertEqual(
            inventory, pipeline_doctor.ModeRequirements(False, False, False)
        )

    def test_render_mode_does_not_require_transcription_packages(self):
        checks = {
            check.name: check
            for check in pipeline_doctor.run_checks(
                mode="render-only", cleanup_required=False
            )
        }
        self.assertFalse(checks["PyTorch"].required)
        self.assertFalse(checks["CTranslate2"].required)
        self.assertFalse(checks["FFmpeg"].required)
        self.assertTrue(checks["python-docx"].required)

    def test_ffmpeg_falls_back_to_path_when_bundle_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            discovered = Path(temporary) / "ffmpeg.exe"
            with (
                patch.object(pipeline_doctor, "REPO_ROOT", Path(temporary)),
                patch("pipeline_doctor.shutil.which", return_value=str(discovered)),
            ):
                self.assertEqual(
                    pipeline_doctor.find_ffmpeg(), discovered.resolve()
                )

    def test_settings_require_a_top_level_json_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / ".transcribe_settings.json"
            with patch.object(pipeline_doctor, "REPO_ROOT", root):
                settings.write_text("[]", encoding="utf-8")
                invalid = pipeline_doctor.check_settings(
                    required=True, mode="transcribe"
                )
                settings.write_text('{"model": "large-v3"}', encoding="utf-8")
                valid = pipeline_doctor.check_settings(
                    required=True, mode="transcribe"
                )
            self.assertEqual(invalid.status, "error")
            self.assertIn("JSON object", invalid.detail)
            self.assertEqual(valid.status, "ok")

    def test_cleanup_credentials_are_required_only_when_requested(self):
        old_id = os.environ.pop("CF_ACCESS_CLIENT_ID", None)
        old_secret = os.environ.pop("CF_ACCESS_CLIENT_SECRET", None)
        try:
            with patch("cleanup_client._keyring_credentials", return_value=None):
                required = pipeline_doctor.check_cleanup_credentials(required=True)
                optional = pipeline_doctor.check_cleanup_credentials(required=False)
            self.assertEqual(required.status, "error")
            self.assertEqual(optional.status, "ok")
        finally:
            if old_id is not None:
                os.environ["CF_ACCESS_CLIENT_ID"] = old_id
            if old_secret is not None:
                os.environ["CF_ACCESS_CLIENT_SECRET"] = old_secret

    def test_checks_never_expose_token_values(self):
        old_id = os.environ.get("CF_ACCESS_CLIENT_ID")
        old_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")
        try:
            os.environ["CF_ACCESS_CLIENT_ID"] = "sensitive-client-id"
            os.environ["CF_ACCESS_CLIENT_SECRET"] = "sensitive-secret"
            serialized = repr(
                pipeline_doctor.check_cleanup_credentials(required=True)
            )
            self.assertNotIn("sensitive-client-id", serialized)
            self.assertNotIn("sensitive-secret", serialized)
        finally:
            if old_id is None:
                os.environ.pop("CF_ACCESS_CLIENT_ID", None)
            else:
                os.environ["CF_ACCESS_CLIENT_ID"] = old_id
            if old_secret is None:
                os.environ.pop("CF_ACCESS_CLIENT_SECRET", None)
            else:
                os.environ["CF_ACCESS_CLIENT_SECRET"] = old_secret

    def test_package_check_imports_and_rejects_version_drift(self):
        with (
            patch("pipeline_doctor.importlib.metadata.version", return_value="9.9.9"),
            patch("pipeline_doctor.importlib.import_module", return_value=object()),
        ):
            _module, check = pipeline_doctor.package_check(
                "ctranslate2",
                "ctranslate2",
                "CTranslate2",
                required=True,
                mode="transcribe",
            )
        self.assertEqual(check.status, "error")
        self.assertIn("pinned version is 4.8.1", check.detail)

    def test_package_check_reports_guarded_import_failure(self):
        with patch(
            "pipeline_doctor.importlib.import_module",
            side_effect=OSError("missing native DLL"),
        ):
            module, check = pipeline_doctor.package_check(
                "ctranslate2",
                "ctranslate2",
                "CTranslate2",
                required=True,
                mode="transcribe",
            )
        self.assertIsNone(module)
        self.assertEqual(check.status, "error")
        self.assertIn("missing native DLL", check.detail)

    def test_torch_cuda_accepts_proven_pascal_lane(self):
        check = pipeline_doctor.check_torch_cuda(
            fake_torch(), require_gpu=True
        )
        self.assertEqual(check.status, "ok")
        self.assertIn("compute 6.1", check.detail)
        self.assertIn("sm_61 present=True", check.detail)

    def test_torch_cuda_rejects_pascal_wheel_without_sm61(self):
        check = pipeline_doctor.check_torch_cuda(
            fake_torch(architectures=["sm_75", "sm_80"]), require_gpu=True
        )
        self.assertEqual(check.status, "error")
        self.assertIn("lacks sm_61", check.detail)

    def test_torch_cuda_allows_explicit_cpu_fallback_as_warning(self):
        optional = pipeline_doctor.check_torch_cuda(
            fake_torch(available=False), require_gpu=False
        )
        required = pipeline_doctor.check_torch_cuda(
            fake_torch(available=False), require_gpu=True
        )
        self.assertEqual(optional.status, "warning")
        self.assertEqual(required.status, "error")

    def test_torch_cuda_rejects_wrong_runtime_lane(self):
        check = pipeline_doctor.check_torch_cuda(
            fake_torch(built_cuda="13.0"), require_gpu=True
        )
        self.assertEqual(check.status, "error")
        self.assertIn("pinned runtime is 12.4", check.detail)

    def test_ctranslate2_cuda_requires_pascal_safe_int8(self):
        passing = pipeline_doctor.check_ctranslate2_cuda(
            FakeCTranslate2(), require_gpu=True
        )
        failing = pipeline_doctor.check_ctranslate2_cuda(
            FakeCTranslate2(compute_types={"float32"}), require_gpu=True
        )
        self.assertEqual(passing.status, "ok")
        self.assertEqual(failing.status, "error")
        self.assertIn("lacks Pascal-safe INT8", failing.detail)

    def test_ctranslate2_zero_devices_honours_gpu_policy(self):
        optional = pipeline_doctor.check_ctranslate2_cuda(
            FakeCTranslate2(devices=0), require_gpu=False
        )
        required = pipeline_doctor.check_ctranslate2_cuda(
            FakeCTranslate2(devices=0), require_gpu=True
        )
        self.assertEqual(optional.status, "warning")
        self.assertEqual(required.status, "error")


if __name__ == "__main__":
    unittest.main()
