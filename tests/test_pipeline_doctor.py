import unittest
from unittest.mock import patch

import pipeline_doctor


class PipelineDoctorTests(unittest.TestCase):
    def test_cleanup_credentials_are_required_unless_cleanup_is_disabled(self):
        old_id = pipeline_doctor.os.environ.pop("CF_ACCESS_CLIENT_ID", None)
        old_secret = pipeline_doctor.os.environ.pop("CF_ACCESS_CLIENT_SECRET", None)
        try:
            with patch("cleanup_client._keyring_credentials", return_value=None):
                required = {
                    check.name: check
                    for check in pipeline_doctor.run_checks(cleanup_required=True)
                }
                local_only = {
                    check.name: check
                    for check in pipeline_doctor.run_checks(cleanup_required=False)
                }
            self.assertEqual(required["Cleanup service token"].status, "error")
            self.assertEqual(local_only["Cleanup service token"].status, "ok")
        finally:
            if old_id is not None:
                pipeline_doctor.os.environ["CF_ACCESS_CLIENT_ID"] = old_id
            if old_secret is not None:
                pipeline_doctor.os.environ["CF_ACCESS_CLIENT_SECRET"] = old_secret

    def test_checks_never_expose_token_values(self):
        old_id = pipeline_doctor.os.environ.get("CF_ACCESS_CLIENT_ID")
        old_secret = pipeline_doctor.os.environ.get("CF_ACCESS_CLIENT_SECRET")
        try:
            pipeline_doctor.os.environ["CF_ACCESS_CLIENT_ID"] = "sensitive-client-id"
            pipeline_doctor.os.environ["CF_ACCESS_CLIENT_SECRET"] = "sensitive-secret"
            serialized = repr(pipeline_doctor.run_checks())
            self.assertNotIn("sensitive-client-id", serialized)
            self.assertNotIn("sensitive-secret", serialized)
        finally:
            if old_id is None:
                pipeline_doctor.os.environ.pop("CF_ACCESS_CLIENT_ID", None)
            else:
                pipeline_doctor.os.environ["CF_ACCESS_CLIENT_ID"] = old_id
            if old_secret is None:
                pipeline_doctor.os.environ.pop("CF_ACCESS_CLIENT_SECRET", None)
            else:
                pipeline_doctor.os.environ["CF_ACCESS_CLIENT_SECRET"] = old_secret


if __name__ == "__main__":
    unittest.main()
