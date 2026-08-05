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
    CleanupAccessError,
    CleanupClient,
    CleanupConfigurationError,
    HttpResponse,
    chunk_text,
)


def response(status: int, body: str, **headers: str) -> HttpResponse:
    return HttpResponse(status, headers, body.encode("utf-8"))


def cleanup_response(
    corrected: str, *, status: str = "passed", model: str = "@cf/zai-org/glm-4.7-flash"
) -> HttpResponse:
    return response(
        200,
        json.dumps(
            {
                "corrected": corrected,
                "model": model,
                "quality": {
                    "status": status,
                    "reasons": [] if status == "passed" else ["manual check"],
                    "used_original": False,
                },
                "grounding": {
                    "glossary_terms_considered": 2,
                    "term_candidates": 1,
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


class ChunkingTests(unittest.TestCase):
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
        self.assertEqual(first_payload["terms"], ["Gurdjieff", "Fourth Way"])
        self.assertEqual(second_payload["terms"], first_payload["terms"])
        self.assertEqual(
            first_payload["model"], "@cf/zai-org/glm-4.7-flash"
        )
        self.assertEqual(second_payload["model"], first_payload["model"])
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
