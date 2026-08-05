"""Resumable client for the protected transcript-cleanup service.

The client deliberately uses only the Python standard library.  By default it
downloads and pins the tooling glossary once per run, then sends that immutable
snapshot with every transcript chunk. Cloudflare Access credentials are read
from a complete environment pair or the OS credential store by
``CleanupClient.from_environment``; they are never written to checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_ENDPOINT = "https://pg.objectiveartefacts.com.au/api/tooling/cleanup-chunk"
DEFAULT_MODEL = "@cf/zai-org/glm-4.7-flash"
DEFAULT_TARGET_WORDS = 800
DEFAULT_MAX_CHARS = 18_000
CHECKPOINT_VERSION = 1
CREDENTIAL_SERVICE = "AudioProcessor Cloudflare Access"
CREDENTIAL_CLIENT_ID_KEY = "client-id"
CREDENTIAL_CLIENT_SECRET_KEY = "client-secret"

_NONSPACE_RE = re.compile(r"\S+")
_PREFERRED_BREAK_RE = re.compile(
    r"(?:\r?\n[ \t]*\r?\n|[.!?][\"'\u2019\u201d)\]]*[ \t\r\n]+)"
)
_ACCESS_MARKERS = (
    "cloudflare access",
    "cloudflareaccess.com",
    "/cdn-cgi/access/login",
    "cf-access",
)


class CleanupClientError(RuntimeError):
    """Base class for cleanup client failures."""


class CleanupConfigurationError(CleanupClientError):
    """The cleanup client is not configured safely."""


class CleanupAccessError(CleanupClientError):
    """Cloudflare Access intercepted or rejected the request."""


class CleanupNetworkError(CleanupClientError):
    """A request failed after all network retries."""


class CleanupProtocolError(CleanupClientError):
    """The cleanup service returned an unexpected response."""


class CleanupHTTPError(CleanupClientError):
    """The cleanup service returned an unsuccessful HTTP response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"cleanup service returned HTTP {status}: {message}")
        self.status = status


def _keyring_credentials() -> tuple[str, str] | None:
    """Read the optional OS credential store without exposing values to logs."""
    try:
        import keyring  # type: ignore

        client_id = keyring.get_password(
            CREDENTIAL_SERVICE, CREDENTIAL_CLIENT_ID_KEY
        )
        client_secret = keyring.get_password(
            CREDENTIAL_SERVICE, CREDENTIAL_CLIENT_SECRET_KEY
        )
    except Exception:
        return None
    if client_id and client_secret:
        return client_id, client_secret
    return None


def resolve_access_credentials() -> tuple[str, str, str]:
    """Resolve a complete Access token pair from env or the OS credential store."""
    client_id = os.environ.get("CF_ACCESS_CLIENT_ID", "").strip()
    client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "").strip()
    if client_id or client_secret:
        if not (client_id and client_secret):
            raise CleanupConfigurationError(
                "CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET must be set together"
            )
        return client_id, client_secret, "environment"

    stored = _keyring_credentials()
    if stored is not None:
        return stored[0], stored[1], "credential-manager"
    raise CleanupConfigurationError(
        "Cloudflare Access credentials were not found in the process environment "
        "or Windows Credential Manager"
    )


@dataclass(frozen=True)
class HttpResponse:
    """Small transport-neutral HTTP response used by the test seam."""

    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[
    [str, str, Mapping[str, str], bytes | None, float], HttpResponse
]


@dataclass(frozen=True)
class TextChunk:
    """An exact, contiguous slice of the source transcript."""

    index: int
    text: str
    start: int
    end: int
    word_count: int


@dataclass(frozen=True)
class GlossarySnapshot:
    """The immutable glossary version used throughout one cleanup run."""

    terms: tuple[str, ...]
    sha256: str | None
    count: int
    pinned: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "count": self.count,
            "pinned": self.pinned,
        }


@dataclass(frozen=True)
class CleanupChunkResult:
    """One service response plus the source slice it corresponds to."""

    index: int
    total_chunks: int
    source_text: str
    corrected: str
    model: str | None
    quality: Any
    grounding: Any
    usage: Any
    from_checkpoint: bool = False

    @property
    def source_sha256(self) -> str:
        return _sha256_text(self.source_text)

    @property
    def needs_review(self) -> bool:
        if not isinstance(self.quality, Mapping):
            return True
        return (
            self.quality.get("status") != "passed"
            or bool(self.quality.get("used_original"))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "total_chunks": self.total_chunks,
            "source_sha256": self.source_sha256,
            "source_chars": len(self.source_text),
            "corrected": self.corrected,
            "model": self.model,
            "quality": self.quality,
            "grounding": self.grounding,
            "usage": self.usage,
            "from_checkpoint": self.from_checkpoint,
        }


@dataclass(frozen=True)
class CleanupResult:
    """Aggregate output from a complete cleanup run."""

    text: str
    model: str | None
    glossary_sha256: str | None
    glossary_count: int
    input_sha256: str
    chunks: tuple[CleanupChunkResult, ...]
    needs_review: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "glossary_sha256": self.glossary_sha256,
            "glossary_count": self.glossary_count,
            "input_sha256": self.input_sha256,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "needs_review": self.needs_review,
            "warnings": list(self.warnings),
        }


def chunk_text(
    text: str,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[TextChunk]:
    """Split *text* losslessly near ``target_words`` and below ``max_chars``.

    Boundaries prefer the last paragraph or sentence ending in the final
    quarter of a candidate chunk.  Every character belongs to exactly one
    chunk, so ``''.join(chunk.text for chunk in chunks) == text`` always holds.
    Very long unbroken tokens are split only when the character ceiling makes
    that unavoidable.
    """

    if target_words < 1:
        raise ValueError("target_words must be at least 1")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        candidate = text[start:hard_end]
        words = list(_NONSPACE_RE.finditer(candidate))

        if len(words) > target_words:
            # Place all whitespace before the next word in the current chunk.
            end = start + words[target_words].start()
        else:
            end = hard_end
            # Avoid cutting an ordinary word at the character limit.
            if (
                end < len(text)
                and end > start
                and not text[end - 1].isspace()
                and not text[end].isspace()
            ):
                whitespace = list(re.finditer(r"\s+", text[start:end]))
                if whitespace:
                    end = start + whitespace[-1].end()

        if end <= start:
            end = hard_end

        bounded = text[start:end]
        bounded_words = list(_NONSPACE_RE.finditer(bounded))
        if len(bounded_words) >= 4:
            minimum_word = max(1, int(len(bounded_words) * 0.75))
            minimum_position = bounded_words[minimum_word - 1].end()
            preferred = [
                match.end()
                for match in _PREFERRED_BREAK_RE.finditer(bounded)
                if match.end() >= minimum_position
            ]
            if preferred:
                end = start + preferred[-1]
                bounded = text[start:end]
                bounded_words = list(_NONSPACE_RE.finditer(bounded))

        chunks.append(
            TextChunk(
                index=len(chunks),
                text=bounded,
                start=start,
                end=end,
                word_count=len(bounded_words),
            )
        )
        start = end

    # These are invariants, not best-effort checks: silent source loss here
    # would undermine every later fidelity check.
    if "".join(chunk.text for chunk in chunks) != text:
        raise AssertionError("internal chunking error: source text was not preserved")
    if any(len(chunk.text) > max_chars for chunk in chunks):
        raise AssertionError("internal chunking error: character limit exceeded")
    if any(chunk.word_count > target_words for chunk in chunks):
        raise AssertionError("internal chunking error: word limit exceeded")
    return chunks


class CleanupClient:
    """Client for sequential, fidelity-checked transcript cleanup."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        endpoint: str = DEFAULT_ENDPOINT,
        terms_endpoint: str | None = None,
        model: str = DEFAULT_MODEL,
        pin_glossary: bool = True,
        target_words: int = DEFAULT_TARGET_WORDS,
        max_chars: int = DEFAULT_MAX_CHARS,
        preceding_context_words: int = 100,
        timeout: float = 60.0,
        max_attempts: int = 4,
        retry_base_delay: float = 1.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not client_id or not client_secret:
            raise CleanupConfigurationError(
                "CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET are required"
            )
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CleanupConfigurationError("cleanup endpoint must be an HTTP(S) URL")
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and parsed.hostname not in local_hosts:
            raise CleanupConfigurationError(
                "cleanup endpoint must use HTTPS when sending Cloudflare Access credentials"
            )
        resolved_terms_endpoint = terms_endpoint or _derive_terms_endpoint(endpoint)
        parsed_terms = urllib.parse.urlsplit(resolved_terms_endpoint)
        if parsed_terms.scheme not in {"http", "https"} or not parsed_terms.netloc:
            raise CleanupConfigurationError("glossary endpoint must be an HTTP(S) URL")
        if parsed_terms.scheme != "https" and parsed_terms.hostname not in local_hosts:
            raise CleanupConfigurationError(
                "glossary endpoint must use HTTPS when sending Cloudflare Access credentials"
            )
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if preceding_context_words < 0:
            raise ValueError("preceding_context_words cannot be negative")

        self.client_id = client_id
        self.client_secret = client_secret
        self.endpoint = endpoint
        self.terms_endpoint = resolved_terms_endpoint
        self.model = model or DEFAULT_MODEL
        self.pin_glossary = pin_glossary
        self.target_words = target_words
        self.max_chars = max_chars
        self.preceding_context_words = preceding_context_words
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_base_delay = retry_base_delay
        self._transport = transport or _urllib_transport
        self._sleep = sleep
        self._clock = clock
        self._glossary: GlossarySnapshot | None = None

    @classmethod
    def from_environment(
        cls, endpoint: str | None = None, model: str | None = None
    ) -> "CleanupClient":
        """Build a client from environment variables.

        Optional variables are ``TRANSCRIPT_CLEANUP_ENDPOINT``,
        ``PG_CLEANUP_MODEL`` (with ``TRANSCRIPT_CLEANUP_MODEL`` retained as a
        compatibility alias) and
        ``TRANSCRIPT_CLEANUP_PIN_GLOSSARY`` (defaults to true).
        """

        pin_value = os.environ.get("TRANSCRIPT_CLEANUP_PIN_GLOSSARY", "true")
        pin_glossary = pin_value.strip().lower() not in {"0", "false", "no", "off"}
        client_id, client_secret, _source = resolve_access_credentials()
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            endpoint=(
                endpoint
                or os.environ.get("TRANSCRIPT_CLEANUP_ENDPOINT")
                or DEFAULT_ENDPOINT
            ),
            model=(
                model
                or os.environ.get("PG_CLEANUP_MODEL")
                or os.environ.get("TRANSCRIPT_CLEANUP_MODEL")
                or DEFAULT_MODEL
            ),
            pin_glossary=pin_glossary,
        )

    @property
    def glossary_sha256(self) -> str | None:
        return self._glossary.sha256 if self._glossary else None

    @property
    def glossary_count(self) -> int:
        return self._glossary.count if self._glossary else 0

    @property
    def glossary_terms(self) -> tuple[str, ...]:
        return self._glossary.terms if self._glossary else ()

    def ensure_glossary(self) -> GlossarySnapshot:
        """Fetch and cache the immutable glossary snapshot for this client."""

        if self._glossary is not None:
            return self._glossary
        if not self.pin_glossary:
            self._glossary = GlossarySnapshot((), None, 0, False)
            return self._glossary

        response = self._request("GET", self.terms_endpoint, body=None)
        try:
            raw = response.body.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise CleanupProtocolError("tooling glossary is not valid UTF-8") from exc
        terms = tuple(
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if len(terms) > 5_000:
            raise CleanupProtocolError("tooling glossary contains more than 5,000 terms")
        if any(len(term) > 200 for term in terms):
            raise CleanupProtocolError("tooling glossary contains a term over 200 characters")
        if sum(len(term) for term in terms) > 250_000:
            raise CleanupProtocolError("tooling glossary exceeds 250,000 characters")

        digest_source = json.dumps(
            terms, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self._glossary = GlossarySnapshot(
            terms=terms,
            sha256=hashlib.sha256(digest_source).hexdigest(),
            count=len(terms),
            pinned=True,
        )
        return self._glossary

    def cleanup_text(
        self,
        text: str,
        checkpoint_dir: Path | None = None,
        *,
        reuse_checkpoints: bool = True,
    ) -> CleanupResult:
        """Clean *text* sequentially, optionally resuming valid checkpoints."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        glossary = self.ensure_glossary()
        input_sha256 = _sha256_text(text)
        source_chunks = chunk_text(
            text, target_words=self.target_words, max_chars=self.max_chars
        )
        warnings: list[str] = []
        results: list[CleanupChunkResult] = []
        total_chunks = len(source_chunks)

        run_dir: Path | None = None
        if checkpoint_dir is not None:
            run_dir = Path(checkpoint_dir) / input_sha256
            run_dir.mkdir(parents=True, exist_ok=True)

        preceding_text = ""
        for source_chunk in source_chunks:
            checkpoint_path = (
                run_dir
                / f"chunk-{source_chunk.index + 1:05d}-of-{total_chunks:05d}.json"
                if run_dir is not None
                else None
            )
            result: CleanupChunkResult | None = None
            if checkpoint_path is not None and reuse_checkpoints:
                result, checkpoint_warning = self._load_checkpoint(
                    checkpoint_path,
                    input_sha256=input_sha256,
                    chunk=source_chunk,
                    total_chunks=total_chunks,
                    glossary=glossary,
                )
                if checkpoint_warning:
                    warnings.append(checkpoint_warning)

            if result is None:
                if not source_chunk.text.strip():
                    quality = {
                        "status": "passed",
                        "reasons": [],
                        "used_original": True,
                        "local_passthrough": True,
                    }
                    result = CleanupChunkResult(
                        index=source_chunk.index,
                        total_chunks=total_chunks,
                        source_text=source_chunk.text,
                        corrected=source_chunk.text,
                        model=None,
                        quality=quality,
                        grounding=None,
                        usage=None,
                    )
                else:
                    context = _tail_words(
                        preceding_text, self.preceding_context_words
                    )
                    result = self._clean_chunk(
                        source_chunk,
                        total_chunks=total_chunks,
                        preceding_context=context,
                        glossary=glossary,
                    )
                if checkpoint_path is not None:
                    self._write_checkpoint(
                        checkpoint_path,
                        result,
                        input_sha256=input_sha256,
                        glossary=glossary,
                    )

            results.append(result)
            preceding_text = _merge_corrected(results)

        cleaned_text = _merge_corrected(results)
        review_needed = any(result.needs_review for result in results)
        for result in results:
            quality = result.quality
            if result.needs_review:
                if isinstance(quality, Mapping):
                    reasons = quality.get("reasons")
                    detail = "; ".join(map(str, reasons)) if reasons else "quality check did not pass"
                    warnings.append(f"chunk {result.index + 1}: {detail}")
                else:
                    warnings.append(f"chunk {result.index + 1}: missing quality metadata")

        models = tuple(dict.fromkeys(result.model for result in results if result.model))
        aggregate_model: str | None
        if not models:
            aggregate_model = self.model
        elif len(models) == 1:
            aggregate_model = models[0]
        else:
            aggregate_model = "mixed: " + ", ".join(models)
            warnings.append("cleanup chunks were produced by more than one model")

        if not glossary.pinned:
            warnings.append("glossary pinning was explicitly disabled")

        return CleanupResult(
            text=cleaned_text,
            model=aggregate_model,
            glossary_sha256=glossary.sha256,
            glossary_count=glossary.count,
            input_sha256=input_sha256,
            chunks=tuple(results),
            needs_review=review_needed,
            warnings=tuple(warnings),
        )

    def _clean_chunk(
        self,
        chunk: TextChunk,
        *,
        total_chunks: int,
        preceding_context: str,
        glossary: GlossarySnapshot,
    ) -> CleanupChunkResult:
        payload: dict[str, Any] = {
            "text": chunk.text,
            "chunkIndex": chunk.index,
            "totalChunks": total_chunks,
            "precedingContext": preceding_context,
        }
        if glossary.pinned:
            payload["terms"] = list(glossary.terms)
        # Pin a concrete model just as we pin the glossary.  Relying on the
        # service's mutable default could otherwise change engines mid-archive.
        payload["model"] = self.model

        response = self._request(
            "POST",
            self.endpoint,
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )
        try:
            decoded = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CleanupProtocolError(
                f"cleanup chunk {chunk.index + 1} did not return valid JSON"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise CleanupProtocolError(
                f"cleanup chunk {chunk.index + 1} returned a non-object response"
            )
        corrected = decoded.get("corrected")
        if not isinstance(corrected, str) or (
            chunk.text.strip() and not corrected.strip()
        ):
            raise CleanupProtocolError(
                f"cleanup chunk {chunk.index + 1} returned no corrected text"
            )
        model = decoded.get("model")
        if model is not None and not isinstance(model, str):
            raise CleanupProtocolError(
                f"cleanup chunk {chunk.index + 1} returned an invalid model id"
            )
        if model != self.model:
            raise CleanupProtocolError(
                f"cleanup chunk {chunk.index + 1} used {model!r}, not the pinned "
                f"model {self.model!r}"
            )
        return CleanupChunkResult(
            index=chunk.index,
            total_chunks=total_chunks,
            source_text=chunk.text,
            corrected=corrected,
            model=model,
            quality=decoded.get("quality"),
            grounding=decoded.get("grounding"),
            usage=decoded.get("usage"),
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None,
        content_type: str | None = None,
    ) -> HttpResponse:
        headers = {
            "Accept": "application/json, text/plain;q=0.9",
            "CF-Access-Client-Id": self.client_id,
            "CF-Access-Client-Secret": self.client_secret,
            "User-Agent": "AudioProcessor-CleanupClient/1",
        }
        if content_type:
            headers["Content-Type"] = content_type

        last_network_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self._transport(method, url, headers, body, self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_network_error = exc
                if attempt + 1 >= self.max_attempts:
                    break
                self._wait_before_retry(attempt, None)
                continue

            if _looks_like_access_response(response):
                raise CleanupAccessError(
                    "Cloudflare Access returned an HTML login/interstitial page; "
                    "check CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET and the "
                    "Access service-token policy"
                )
            if 200 <= response.status < 300:
                return response

            retryable = response.status == 429 or 500 <= response.status <= 599
            if retryable and attempt + 1 < self.max_attempts:
                self._wait_before_retry(attempt, _header(response.headers, "Retry-After"))
                continue
            message = _response_error_message(response.body)
            raise CleanupHTTPError(response.status, message)

        assert last_network_error is not None
        raise CleanupNetworkError(
            f"cleanup request failed after {self.max_attempts} attempts: "
            f"{last_network_error}"
        ) from last_network_error

    def _wait_before_retry(self, attempt: int, retry_after: str | None) -> None:
        delay = _parse_retry_after(retry_after, self._clock())
        if delay is None:
            delay = self.retry_base_delay * (2**attempt)
        if delay > 0:
            self._sleep(delay)

    def _load_checkpoint(
        self,
        path: Path,
        *,
        input_sha256: str,
        chunk: TextChunk,
        total_chunks: int,
        glossary: GlossarySnapshot,
    ) -> tuple[CleanupChunkResult | None, str | None]:
        if not path.exists():
            return None, None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "input_sha256": input_sha256,
                "chunk_sha256": _sha256_text(chunk.text),
                "chunk_index": chunk.index,
                "total_chunks": total_chunks,
                "endpoint": self.endpoint,
                "requested_model": self.model,
                "glossary_sha256": glossary.sha256,
                "glossary_count": glossary.count,
            }
            if not isinstance(data, Mapping) or any(
                data.get(key) != value for key, value in expected.items()
            ):
                raise ValueError("checkpoint metadata no longer matches this run")
            response = data.get("response")
            if not isinstance(response, Mapping):
                raise ValueError("checkpoint response is missing")
            corrected = response.get("corrected")
            if not isinstance(corrected, str):
                raise ValueError("checkpoint corrected text is invalid")
            model = response.get("model")
            if model is not None and not isinstance(model, str):
                raise ValueError("checkpoint model id is invalid")
            return (
                CleanupChunkResult(
                    index=chunk.index,
                    total_chunks=total_chunks,
                    source_text=chunk.text,
                    corrected=corrected,
                    model=model,
                    quality=response.get("quality"),
                    grounding=response.get("grounding"),
                    usage=response.get("usage"),
                    from_checkpoint=True,
                ),
                None,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return None, f"ignored invalid checkpoint {path.name}: {exc}"

    def _write_checkpoint(
        self,
        path: Path,
        result: CleanupChunkResult,
        *,
        input_sha256: str,
        glossary: GlossarySnapshot,
    ) -> None:
        data = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "input_sha256": input_sha256,
            "chunk_sha256": result.source_sha256,
            "chunk_index": result.index,
            "total_chunks": result.total_chunks,
            "endpoint": self.endpoint,
            "requested_model": self.model,
            "glossary_sha256": glossary.sha256,
            "glossary_count": glossary.count,
            "response": {
                "corrected": result.corrected,
                "model": result.model,
                "quality": result.quality,
                "grounding": result.grounding,
                "usage": result.usage,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                json.dump(data, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)


def _urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> HttpResponse:
    request = urllib.request.Request(
        url=url, data=body, headers=dict(headers), method=method
    )
    # Do not follow an Access redirect: following can turn a clear 302 into an
    # HTML parse error and may forward authentication headers to another host.
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return HttpResponse(
                status=response.status,
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status=exc.code,
            headers=dict(exc.headers.items()) if exc.headers else {},
            body=exc.read(),
        )


def _derive_terms_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    path = parsed.path
    if path.endswith("/cleanup-chunk"):
        path = path[: -len("cleanup-chunk")] + "terms"
    else:
        path = path.rstrip("/") + "/terms"
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tail_words(text: str, limit: int) -> str:
    if limit <= 0 or not text:
        return ""
    matches = list(_NONSPACE_RE.finditer(text))
    if not matches:
        return ""
    return text[matches[max(0, len(matches) - limit)].start() :].strip()


def _merge_corrected(results: Sequence[CleanupChunkResult]) -> str:
    pieces: list[str] = []
    for result in results:
        source = result.source_text
        if not source or not source.strip():
            pieces.append(source)
            continue
        leading_match = re.match(r"\s*", source)
        trailing_match = re.search(r"\s*$", source)
        leading = leading_match.group(0) if leading_match else ""
        trailing = trailing_match.group(0) if trailing_match else ""
        pieces.append(leading + result.corrected.strip() + trailing)
    return "".join(pieces)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def _looks_like_access_response(response: HttpResponse) -> bool:
    location = (_header(response.headers, "Location") or "").lower()
    if 300 <= response.status < 400 and any(
        marker in location for marker in _ACCESS_MARKERS
    ):
        return True
    content_type = (_header(response.headers, "Content-Type") or "").lower()
    sample = response.body[:32_768].decode("utf-8", errors="ignore").lower()
    looks_html = "text/html" in content_type or "<html" in sample or "<!doctype html" in sample
    if not looks_html:
        return False
    if any(marker in sample for marker in _ACCESS_MARKERS):
        return True
    # Successful HTML can only be an interstitial: both tooling endpoints return
    # text/plain or JSON.  Calling this out as Access is more useful than a JSON
    # parser error and is the common service-token failure mode.
    return 200 <= response.status < 300 or response.status in {401, 403}


def _response_error_message(body: bytes) -> str:
    decoded = body[:2_000].decode("utf-8", errors="replace").strip()
    if not decoded:
        return "empty response"
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        return decoded[:500]
    if isinstance(parsed, Mapping) and parsed.get("error"):
        return str(parsed["error"])[:500]
    return decoded[:500]


def _parse_retry_after(value: str | None, now: float) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, parsed.timestamp() - now)
    except (TypeError, ValueError, OverflowError):
        return None


__all__ = [
    "CleanupAccessError",
    "CleanupChunkResult",
    "CleanupClient",
    "CleanupClientError",
    "CleanupConfigurationError",
    "CleanupHTTPError",
    "CleanupNetworkError",
    "CleanupProtocolError",
    "CleanupResult",
    "CREDENTIAL_CLIENT_ID_KEY",
    "CREDENTIAL_CLIENT_SECRET_KEY",
    "CREDENTIAL_SERVICE",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "GlossarySnapshot",
    "HttpResponse",
    "TextChunk",
    "chunk_text",
    "resolve_access_credentials",
]
