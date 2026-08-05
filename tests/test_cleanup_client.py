from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from cleanup_client import (
    ACCESS_CLIENT_ID_HEADER,
    ACCESS_CLIENT_SECRET_HEADER,
    CleanupAccessError,
    CleanupClient,
    CleanupConfigurationError,
    CleanupProtocolError,
    CHECKPOINT_VERSION,
    DEFAULT_CLEANUP_PROFILE,
    HttpResponse,
    chunk_text,
    normalize_access_credential,
)
from pipeline_control import PipelineCancelledError


def response(status: int, body: str, **headers: str) -> HttpResponse:
    return HttpResponse(status, headers, body.encode("utf-8"))


def cleanup_response(
    corrected: str, *, status: str = "passed", model: str = "@cf/zai-org/glm-4.7-flash"
) -> HttpResponse:
    corrected_sha256 = hashlib.sha256(corrected.encode("utf-8")).hexdigest()
    return response(
        200,
        json.dumps(
            {
                "corrected": corrected,
                "model": model,
                "cleanup_profile": DEFAULT_CLEANUP_PROFILE,
                "quality": {
                    "status": status,
                    "reasons": [] if status == "passed" else ["manual check"],
                    "used_original": False,
                },
                "grounding": {
                    "glossary_terms_considered": 2,
                    "term_candidates": 1,
                },
                "repair": {
                    "attempted": True,
                    "profile": DEFAULT_CLEANUP_PROFILE,
                    "model": "@cf/zai-org/glm-4.7-flash",
                    "input_sha256": corrected_sha256,
                    "output_sha256": corrected_sha256,
                    "input_bytes": len(corrected.encode("utf-8")),
                    "output_bytes": len(corrected.encode("utf-8")),
                    "proposal_count": 0,
                    "applied_count": 0,
                    "review_count": 0,
                    "rejected_count": 0,
                    "decisions": [],
                },
                "usage": {"input_tokens": 20, "output_tokens": 20},
            }
        ),
        **{"Content-Type": "application/json"},
    )


class ScriptedTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout": timeout,
            }
        )
        if not self.replies:
            raise AssertionError("unexpected HTTP call")
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class CredentialInputTests(unittest.TestCase):
    def test_accepts_cloudflare_copied_header_lines(self):
        self.assertEqual(
            "example.access",
            normalize_access_credential(
                "CF-Access-Client-Id: example.access", ACCESS_CLIENT_ID_HEADER
            ),
        )
        self.assertEqual(
            "secret-value",
            normalize_access_credential(
                "cf-access-client-secret: secret-value",
                ACCESS_CLIENT_SECRET_HEADER,
            ),
        )

    def test_preserves_raw_values(self):
        self.assertEqual(
            "example.access",
            normalize_access_credential("  example.access  ", ACCESS_CLIENT_ID_HEADER),
        )


class ChunkingTests(unittest.TestCase):
    def test_short_text_remains_one_chunk(self):
        text = "First sentence. Second sentence. Final sentence."

        chunks = chunk_text(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_chunking_is_lossless_and_respects_both_limits(self):
        paragraphs = []
        for paragraph in range(24):
            paragraphs.append(
                " ".join(f"word{paragraph}_{word}." for word in range(95))
            )
        text = "\n\n".join(paragraphs) + "\n"

        chunks = chunk_text(text, target_words=800, max_chars=18_000)

        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.word_count <= 800 for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 18_000 for chunk in chunks))

    def test_unbroken_token_is_split_without_character_loss(self):
        text = "a" * 40_001
        chunks = chunk_text(text, target_words=800, max_chars=18_000)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertEqual([len(chunk.text) for chunk in chunks], [18_000, 18_000, 4_001])


class CleanupClientTests(unittest.TestCase):
    def test_rejects_plain_http_for_non_local_access_credentials(self):
        with self.assertRaises(CleanupConfigurationError):
            CleanupClient(
                client_id="client-id",
                client_secret="client-secret",
                endpoint="http://example.test/api/tooling/cleanup-chunk",
            )
        with self.assertRaises(CleanupConfigurationError):
            CleanupClient(
                client_id="client-id",
                client_secret="client-secret",
                endpoint="https://example.test/api/tooling/cleanup-chunk",
                terms_endpoint="http://example.test/api/tooling/terms",
            )

    def make_client(self, transport, **kwargs):
        return CleanupClient(
            client_id="client-id",
            client_secret="client-secret",
            endpoint="https://example.test/api/tooling/cleanup-chunk",
            transport=transport,
            sleep=lambda _seconds: None,
            **kwargs,
        )

    def test_pins_glossary_authenticates_and_sends_sequential_context(self):
        transport = ScriptedTransport(
            [
                response(200, "# comment\nGurdjieff\n Fourth Way \n", **{"Content-Type": "text/plain"}),
                cleanup_response("One two three."),
                cleanup_response("Four five six."),
            ]
        )
        client = self.make_client(transport, target_words=3)

        result = client.cleanup_text("one two three four five six")

        self.assertEqual(len(result.chunks), 2)
        self.assertEqual(result.model, "@cf/zai-org/glm-4.7-flash")
        self.assertEqual(result.glossary_count, 2)
        expected_hash = hashlib.sha256(
            json.dumps(
                ("Gurdjieff", "Fourth Way"),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(result.glossary_sha256, expected_hash)
        self.assertFalse(result.needs_review)

        for call in transport.calls:
            self.assertEqual(call["headers"]["CF-Access-Client-Id"], "client-id")
            self.assertEqual(call["headers"]["CF-Access-Client-Secret"], "client-secret")
        first_payload = json.loads(transport.calls[1]["body"])
        second_payload = json.loads(transport.calls[2]["body"])
        for call in transport.calls[1:]:
            self.assertEqual(
                call["headers"]["Idempotency-Key"],
                hashlib.sha256(call["body"]).hexdigest(),
            )
        self.assertEqual(first_payload["terms"], ["Gurdjieff", "Fourth Way"])
        self.assertEqual(second_payload["terms"], first_payload["terms"])
        self.assertEqual(
            first_payload["model"], "@cf/zai-org/glm-4.7-flash"
        )
        self.assertEqual(second_payload["model"], first_payload["model"])
        self.assertEqual(first_payload["cleanupProfile"], DEFAULT_CLEANUP_PROFILE)
        self.assertEqual(second_payload["cleanupProfile"], DEFAULT_CLEANUP_PROFILE)
        self.assertNotIn("glossary", first_payload)
        self.assertEqual(first_payload["precedingContext"], "")
        self.assertEqual(second_payload["precedingContext"], "One two three.")

    def test_checkpoint_resumes_every_chunk_for_same_input_and_glossary(self):
        terms = response(200, "Gurdjieff\nFourth Way\n")
        text = "one two three four five six"
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            first_transport = ScriptedTransport(
                [terms, cleanup_response("One two three."), cleanup_response("Four five six.")]
            )
            first = self.make_client(first_transport, target_words=3).cleanup_text(
                text, checkpoint_dir
            )

            input_directory = checkpoint_dir / first.input_sha256
            self.assertTrue(input_directory.is_dir())
            self.assertEqual(len(list(input_directory.glob("chunk-*.json"))), 2)
            checkpoint_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in input_directory.glob("chunk-*.json")
            )
            self.assertNotIn("client-id", checkpoint_text)
            self.assertNotIn("client-secret", checkpoint_text)
            checkpoint_records = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(input_directory.glob("chunk-*.json"))
            ]
            self.assertEqual(CHECKPOINT_VERSION, 10)
            self.assertTrue(
                all(
                    record["checkpoint_version"] == CHECKPOINT_VERSION
                    and len(record["preceding_context_sha256"]) == 64
                    for record in checkpoint_records
                )
            )

            second_transport = ScriptedTransport([terms])
            second = self.make_client(second_transport, target_words=3).cleanup_text(
                text, checkpoint_dir
            )

            self.assertEqual(len(second_transport.calls), 1)
            self.assertTrue(all(chunk.from_checkpoint for chunk in second.chunks))
            self.assertEqual(second.text, first.text)
            self.assertEqual(second.to_dict()["glossary_sha256"], first.glossary_sha256)

            forced_transport = ScriptedTransport(
                [
                    terms,
                    cleanup_response("Fresh one two three."),
                    cleanup_response("Fresh four five six."),
                ]
            )
            forced = self.make_client(
                forced_transport, target_words=3
            ).cleanup_text(text, checkpoint_dir, reuse_checkpoints=False)
            self.assertEqual(len(forced_transport.calls), 3)
            self.assertTrue(all(not chunk.from_checkpoint for chunk in forced.chunks))
            self.assertNotEqual(forced.text, first.text)

    def test_recomputed_upstream_context_invalidates_downstream_checkpoints(self):
        terms = response(200, "Gurdjieff\nFourth Way\n")
        text = "one two three four five six"
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            initial_transport = ScriptedTransport(
                [
                    terms,
                    cleanup_response("One two "),
                    cleanup_response("three four "),
                    cleanup_response("five six"),
                ]
            )
            first = self.make_client(
                initial_transport, target_words=2
            ).cleanup_text(text, checkpoint_dir)
            run_dir = checkpoint_dir / first.input_sha256
            checkpoints = sorted(run_dir.glob("chunk-*.json"))
            self.assertEqual(len(checkpoints), 3)

            # Force only chunk 1 to be recomputed. Its changed correction must
            # invalidate every downstream checkpoint whose request context was
            # derived from the old upstream output.
            checkpoints[0].unlink()
            resumed_transport = ScriptedTransport(
                [
                    terms,
                    cleanup_response("Changed one two "),
                    cleanup_response("changed three four "),
                    cleanup_response("changed five six"),
                ]
            )
            resumed = self.make_client(
                resumed_transport, target_words=2
            ).cleanup_text(text, checkpoint_dir)

            self.assertEqual(len(resumed_transport.calls), 4)
            self.assertTrue(all(not chunk.from_checkpoint for chunk in resumed.chunks))
            self.assertTrue(
                any("chunk-00002" in warning for warning in resumed.warnings)
            )
            self.assertTrue(
                any("chunk-00003" in warning for warning in resumed.warnings)
            )

    def test_cancellation_after_remote_chunk_stops_later_calls_and_preserves_checkpoint(self):
        text = "one two three four five six"
        cancelled = {"value": False}
        scripted = ScriptedTransport(
            [
                response(200, "Gurdjieff\nFourth Way\n"),
                cleanup_response("One two "),
                cleanup_response("three four "),
                cleanup_response("five six"),
            ]
        )

        def cancel_after_first_post(method, url, headers, body, timeout):
            reply = scripted(method, url, headers, body, timeout)
            if method == "POST":
                cancelled["value"] = True
            return reply

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            client = self.make_client(
                cancel_after_first_post,
                target_words=2,
            )

            with self.assertRaises(PipelineCancelledError):
                client.cleanup_text(
                    text,
                    checkpoint_dir,
                    cancel_check=lambda: cancelled["value"],
                )

            self.assertEqual(len(scripted.calls), 2)  # glossary + first POST only
            run_dir = checkpoint_dir / hashlib.sha256(text.encode("utf-8")).hexdigest()
            checkpoints = sorted(run_dir.glob("chunk-*.json"))
            self.assertEqual(len(checkpoints), 1)
            saved = json.loads(checkpoints[0].read_text(encoding="utf-8"))
            self.assertEqual(saved["chunk_index"], 0)
            self.assertEqual(saved["response"]["corrected"], "One two ")

            resume_transport = ScriptedTransport(
                [
                    response(200, "Gurdjieff\nFourth Way\n"),
                    cleanup_response("three four "),
                    cleanup_response("five six"),
                ]
            )
            resumed = self.make_client(
                resume_transport,
                target_words=2,
            ).cleanup_text(text, checkpoint_dir)

            self.assertEqual(len(resume_transport.calls), 3)
            self.assertTrue(resumed.chunks[0].from_checkpoint)
            self.assertFalse(resumed.chunks[1].from_checkpoint)
            self.assertFalse(resumed.chunks[2].from_checkpoint)
            self.assertEqual(resumed.text, "One two three four five six")

    def test_glossary_retry_backoff_is_cooperatively_interruptible(self):
        cancelled = {"value": False}
        sleeps = []
        transport = ScriptedTransport(
            [
                urllib.error.URLError("temporary glossary failure"),
                response(200, "Gurdjieff\n"),
            ]
        )

        def stop_during_backoff(delay):
            sleeps.append(delay)
            cancelled["value"] = True

        client = CleanupClient(
            client_id="client-id",
            client_secret="client-secret",
            endpoint="https://example.test/api/tooling/cleanup-chunk",
            transport=transport,
            sleep=stop_during_backoff,
            retry_base_delay=30.0,
        )

        with self.assertRaises(PipelineCancelledError):
            client.ensure_glossary(cancel_check=lambda: cancelled["value"])

        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(sleeps, [0.1])

    def test_cleanup_retry_backoff_is_cooperatively_interruptible(self):
        cancelled = {"value": False}
        sleeps = []
        transport = ScriptedTransport(
            [
                response(200, "Gurdjieff\n"),
                urllib.error.URLError("temporary cleanup failure"),
                cleanup_response("Cleaned."),
            ]
        )

        def stop_during_backoff(delay):
            sleeps.append(delay)
            cancelled["value"] = True

        client = CleanupClient(
            client_id="client-id",
            client_secret="client-secret",
            endpoint="https://example.test/api/tooling/cleanup-chunk",
            transport=transport,
            sleep=stop_during_backoff,
            retry_base_delay=30.0,
        )

        with self.assertRaises(PipelineCancelledError):
            client.cleanup_text(
                "cleaned",
                cancel_check=lambda: cancelled["value"],
            )

        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(sleeps, [0.1])

    def test_retries_network_429_and_5xx_and_honours_retry_after(self):
        sleeps = []
        transport = ScriptedTransport(
            [
                response(200, "Gurdjieff\n"),
                urllib.error.URLError("temporary DNS failure"),
                response(429, '{"error":"rate limited"}', **{"Retry-After": "3"}),
                response(503, '{"error":"busy"}'),
                cleanup_response("Cleaned."),
            ]
        )
        client = CleanupClient(
            client_id="client-id",
            client_secret="client-secret",
            endpoint="https://example.test/api/tooling/cleanup-chunk",
            transport=transport,
            sleep=sleeps.append,
            retry_base_delay=1,
            max_attempts=4,
        )

        result = client.cleanup_text("cleaned")

        self.assertEqual(result.text, "Cleaned.")
        self.assertEqual(sleeps, [1, 3.0, 4])
        self.assertEqual(len(transport.calls), 5)

    def test_cloudflare_access_html_has_a_clear_error(self):
        transport = ScriptedTransport(
            [
                response(
                    200,
                    "<!doctype html><title>Cloudflare Access</title>"
                    '<a href="/cdn-cgi/access/login">sign in</a>',
                    **{"Content-Type": "text/html"},
                )
            ]
        )
        client = self.make_client(transport)

        with self.assertRaisesRegex(CleanupAccessError, "service-token policy"):
            client.ensure_glossary()

    def test_cloudflare_access_redirect_has_a_clear_error(self):
        transport = ScriptedTransport(
            [
                HttpResponse(
                    302,
                    {
                        "Location": "https://example.cloudflareaccess.com/cdn-cgi/access/login"
                    },
                    b"",
                )
            ]
        )
        client = self.make_client(transport)
        with self.assertRaises(CleanupAccessError):
            client.ensure_glossary()

    def test_explicit_unpinned_mode_omits_terms(self):
        transport = ScriptedTransport([cleanup_response("Cleaned.")])
        client = self.make_client(transport, pin_glossary=False)

        result = client.cleanup_text("cleaned")

        payload = json.loads(transport.calls[0]["body"])
        self.assertNotIn("terms", payload)
        self.assertIsNone(result.glossary_sha256)
        self.assertEqual(result.glossary_count, 0)
        self.assertIn("glossary pinning was explicitly disabled", result.warnings)

    def test_rejects_unpinned_server_cleanup_profile(self):
        bad = cleanup_response("Cleaned.")
        decoded = json.loads(bad.body.decode("utf-8"))
        decoded["cleanup_profile"] = "different-profile"
        transport = ScriptedTransport(
            [
                response(200, "Gurdjieff\n"),
                response(200, json.dumps(decoded), **{"Content-Type": "application/json"}),
            ]
        )
        client = self.make_client(transport)

        with self.assertRaisesRegex(CleanupProtocolError, "pinned profile"):
            client.cleanup_text("cleaned")

    def test_requires_semantic_repair_provenance(self):
        bad = cleanup_response("Cleaned.")
        decoded = json.loads(bad.body.decode("utf-8"))
        del decoded["repair"]
        transport = ScriptedTransport(
            [
                response(200, "Gurdjieff\n"),
                response(200, json.dumps(decoded), **{"Content-Type": "application/json"}),
            ]
        )
        client = self.make_client(transport)

        with self.assertRaisesRegex(CleanupProtocolError, "no semantic repair metadata"):
            client.cleanup_text("cleaned")

    def test_rejects_semantic_repair_output_hash_mismatch(self):
        bad = cleanup_response("Cleaned.")
        decoded = json.loads(bad.body.decode("utf-8"))
        decoded["repair"]["output_sha256"] = "0" * 64
        transport = ScriptedTransport(
            [
                response(200, "Gurdjieff\n"),
                response(200, json.dumps(decoded), **{"Content-Type": "application/json"}),
            ]
        )
        client = self.make_client(transport)

        with self.assertRaisesRegex(CleanupProtocolError, "output provenance"):
            client.cleanup_text("cleaned")

    def test_repair_review_metadata_fails_safe_even_if_quality_says_passed(self):
        reply = cleanup_response("Cleaned.")
        decoded = json.loads(reply.body.decode("utf-8"))
        decoded["repair"].update(
            {
                "proposal_count": 1,
                "applied_count": 0,
                "review_count": 1,
                "rejected_count": 0,
                "decisions": [{"disposition": "review"}],
            }
        )
        transport = ScriptedTransport(
            [
                response(200, "Gurdjieff\n"),
                response(200, json.dumps(decoded), **{"Content-Type": "application/json"}),
            ]
        )

        result = self.make_client(transport).cleanup_text("cleaned")

        self.assertTrue(result.needs_review)

    def test_tampered_repair_checkpoint_is_ignored_and_refetched(self):
        text = "one two three"
        terms = response(200, "Gurdjieff\n")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            first = self.make_client(
                ScriptedTransport([terms, cleanup_response("One two three.")])
            ).cleanup_text(text, checkpoint_dir)
            checkpoint = next(
                (checkpoint_dir / first.input_sha256).glob("chunk-*.json")
            )
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            record["response"]["repair"]["output_sha256"] = "0" * 64
            checkpoint.write_text(json.dumps(record), encoding="utf-8")

            retry_transport = ScriptedTransport(
                [
                    response(200, "Gurdjieff\n"),
                    cleanup_response("Fresh one two three."),
                ]
            )
            retried = self.make_client(retry_transport).cleanup_text(
                text, checkpoint_dir
            )

        self.assertEqual(retried.text, "Fresh one two three.")
        self.assertFalse(retried.chunks[0].from_checkpoint)
        self.assertEqual(len(retry_transport.calls), 2)
        self.assertTrue(
            any("ignored invalid checkpoint" in warning for warning in retried.warnings)
        )

    def test_from_environment_requires_both_access_credentials(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "cleanup_client._keyring_credentials", return_value=None
        ):
            with self.assertRaises(CleanupConfigurationError):
                CleanupClient.from_environment()

    def test_from_environment_uses_os_credential_store(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "cleanup_client._keyring_credentials",
            return_value=("stored-id", "stored-secret"),
        ):
            client = CleanupClient.from_environment()
        self.assertEqual(client.client_id, "stored-id")
        self.assertEqual(client.client_secret, "stored-secret")

    def test_from_environment_pins_explicit_model(self):
        with patch.dict(
            os.environ,
            {
                "CF_ACCESS_CLIENT_ID": "id",
                "CF_ACCESS_CLIENT_SECRET": "secret",
                "PG_CLEANUP_MODEL": "@cf/example/pinned-model",
            },
            clear=True,
        ):
            client = CleanupClient.from_environment()
        self.assertEqual(client.model, "@cf/example/pinned-model")


if __name__ == "__main__":
    unittest.main()
