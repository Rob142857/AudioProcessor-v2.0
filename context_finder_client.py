"""Protected GLM boundary client for :mod:`context_finder`.

The service may choose offsets, but this client never accepts replacement prose.
Every response is checked against the immutable local paragraph snapshot before
paragraph-only bounds are applied by ``context_finder.apply_boundary_selection``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
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

from cleanup_client import (
    ACCESS_CLIENT_ID_HEADER,
    ACCESS_CLIENT_SECRET_HEADER,
    HttpResponse,
    resolve_access_credentials,
)
from context_finder import (
    ContextRegion,
    OccurrenceRecord,
    ParagraphSnapshot,
    SearchResult,
    apply_boundary_selection,
)
from pipeline_control import CancelCheck, PipelineCancelledError, raise_if_cancelled


DEFAULT_ENDPOINT = "https://pg.objectiveartefacts.com.au/api/tooling/context-extract"
BOUNDARY_PROFILE = "source-preserving-context-boundaries-v1"
BOUNDARY_MODEL = "@cf/zai-org/glm-4.7-flash"
CHECKPOINT_VERSION = 1
DEFAULT_MAX_WORKERS = 3
MAX_WORKERS = 8
MAX_PARAGRAPHS = 48
MAX_PARAGRAPH_CHARS = 8_000
MAX_CONTEXT_BYTES = 48_000
MAX_REQUEST_BYTES = 64_000
MAX_RESPONSE_BYTES = 128_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACCESS_MARKERS = (
    "cloudflare access",
    "cloudflareaccess.com",
    "/cdn-cgi/access/login",
    "cf-access",
)

Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], HttpResponse]
ProgressCallback = Callable[[int, int, ContextRegion, str], None]


class ContextFinderClientError(RuntimeError):
    """Base class for safe per-region refinement failures."""


class ContextFinderConfigurationError(ContextFinderClientError):
    pass


class ContextFinderAccessError(ContextFinderClientError):
    pass


class ContextFinderProtocolError(ContextFinderClientError):
    pass


class ContextFinderNetworkError(ContextFinderClientError):
    pass


@dataclass(frozen=True, slots=True)
class _PreparedRequest:
    occurrence_id: str
    payload: dict[str, Any]
    body: bytes
    body_sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedReply:
    start_paragraph: int
    end_paragraph: int
    confidence: float
    method: str
    response_sha256: str
    response_metadata: Mapping[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _utf16_offset(text: str, python_offset: int) -> int:
    if python_offset < 0 or python_offset > len(text):
        raise ContextFinderProtocolError("local occurrence offset is outside its paragraph")
    return len(text[:python_offset].encode("utf-16-le")) // 2


def _python_offset(text: str, utf16_offset: int) -> int:
    if not _is_int(utf16_offset) or utf16_offset < 0:
        raise ContextFinderProtocolError("service returned an invalid UTF-16 offset")
    units = 0
    if utf16_offset == 0:
        return 0
    for index, character in enumerate(text):
        units += 2 if ord(character) > 0xFFFF else 1
        if units == utf16_offset:
            return index + 1
        if units > utf16_offset:
            raise ContextFinderProtocolError("service offset splits a UTF-16 surrogate pair")
    if units == utf16_offset:
        return len(text)
    raise ContextFinderProtocolError("service offset is outside its paragraph")


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == wanted), None)


def _looks_like_access_response(response: HttpResponse) -> bool:
    location = (_header(response.headers, "Location") or "").casefold()
    if 300 <= response.status < 400 and any(marker in location for marker in _ACCESS_MARKERS):
        return True
    content_type = (_header(response.headers, "Content-Type") or "").casefold()
    sample = response.body[:32_768].decode("utf-8", errors="ignore").casefold()
    looks_html = "text/html" in content_type or "<html" in sample or "<!doctype html" in sample
    return looks_html and (
        any(marker in sample for marker in _ACCESS_MARKERS)
        or 200 <= response.status < 300
        or response.status in {401, 403}
    )


def _urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> HttpResponse:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return HttpResponse(
                status=response.status,
                headers=dict(response.headers.items()),
                body=response.read(MAX_RESPONSE_BYTES + 1),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status=exc.code,
            headers=dict(exc.headers.items()) if exc.headers else {},
            body=exc.read(MAX_RESPONSE_BYTES + 1),
        )


def _region_fingerprint(region: ContextRegion) -> str:
    value = {
        "region_id": region.region_id,
        "query": region.query,
        "source_relative_path": region.source_relative_path,
        "source_sha256": region.source_sha256,
        "broad_start_paragraph": region.broad_start_paragraph,
        "broad_end_paragraph": region.broad_end_paragraph,
        "paragraphs": [
            {"number": paragraph.number, "text": paragraph.text}
            for paragraph in region.paragraphs
        ],
        "occurrences": [asdict(occurrence) for occurrence in region.occurrences],
    }
    return _sha256_text(_compact_json(value))


def _selection_text(region: ContextRegion, start: int, end: int) -> str:
    return "\n\n".join(
        paragraph.text for paragraph in region.paragraphs if start <= paragraph.number <= end
    )


def _paragraph_window(
    region: ContextRegion, occurrence: OccurrenceRecord
) -> tuple[ParagraphSnapshot, ...]:
    nonempty = tuple(paragraph for paragraph in region.paragraphs if paragraph.text)
    hit_index = next(
        (index for index, item in enumerate(nonempty) if item.number == occurrence.paragraph_number),
        None,
    )
    if hit_index is None:
        raise ContextFinderProtocolError("occurrence paragraph is absent from its region")
    hit = nonempty[hit_index]
    if len(hit.text) > MAX_PARAGRAPH_CHARS:
        raise ContextFinderProtocolError("occurrence paragraph exceeds the service limit")

    chosen = [hit]
    left = hit_index - 1
    right = hit_index + 1
    while len(chosen) < MAX_PARAGRAPHS and (left >= 0 or right < len(nonempty)):
        added = False
        for side in ("left", "right"):
            if len(chosen) >= MAX_PARAGRAPHS:
                break
            index = left if side == "left" else right
            if index < 0 or index >= len(nonempty):
                continue
            candidate = nonempty[index]
            if len(candidate.text) > MAX_PARAGRAPH_CHARS:
                if side == "left":
                    left = -1
                else:
                    right = len(nonempty)
                continue
            proposed = sorted((*chosen, candidate), key=lambda item: item.number)
            source = [{"number": item.number, "text": item.text} for item in proposed]
            if len(_compact_json(source).encode("utf-8")) <= MAX_CONTEXT_BYTES:
                chosen = proposed
                added = True
                if side == "left":
                    left -= 1
                else:
                    right += 1
            elif side == "left":
                left = -1
            else:
                right = len(nonempty)
        if not added:
            break
    return tuple(chosen)


def _prepare_requests(region: ContextRegion) -> tuple[_PreparedRequest, ...]:
    if not region.occurrences:
        raise ContextFinderProtocolError("context region has no exact occurrences")
    paragraph_map = {paragraph.number: paragraph for paragraph in region.paragraphs}
    prepared: list[_PreparedRequest] = []
    for occurrence in region.occurrences:
        paragraph = paragraph_map.get(occurrence.paragraph_number)
        if paragraph is None:
            raise ContextFinderProtocolError("occurrence references an unknown paragraph")
        if not (0 <= occurrence.start < occurrence.end <= len(paragraph.text)):
            raise ContextFinderProtocolError("occurrence coordinates are outside the paragraph")
        matched = paragraph.text[occurrence.start : occurrence.end]
        if matched != occurrence.matched_text:
            raise ContextFinderProtocolError("occurrence text no longer matches its source")
        snapshots = _paragraph_window(region, occurrence)
        payload = {
            # Literal matched text preserves source casing and whitespace; the
            # original user query is still represented by the immutable region.
            "query": matched,
            "paragraphs": [
                {"number": item.number, "text": item.text} for item in snapshots
            ],
            "hit": {
                "paragraphNumber": occurrence.paragraph_number,
                "startOffset": _utf16_offset(paragraph.text, occurrence.start),
                "endOffset": _utf16_offset(paragraph.text, occurrence.end),
            },
        }
        body = _compact_json(payload).encode("utf-8")
        if len(body) > MAX_REQUEST_BYTES:
            raise ContextFinderProtocolError("context request exceeds the service limit")
        prepared.append(
            _PreparedRequest(
                occurrence_id=occurrence.occurrence_id,
                payload=payload,
                body=body,
                body_sha256=_sha256_bytes(body),
            )
        )
    return tuple(prepared)


class ContextFinderClient:
    """Concurrent, hash-validating client for paragraph boundary refinement."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 180.0,
        max_attempts: int = 3,
        retry_base_delay: float = 1.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not client_id or not client_secret:
            raise ContextFinderConfigurationError("Cloudflare Access credentials are required")
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ContextFinderConfigurationError("context endpoint must be an HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ContextFinderConfigurationError(
                "context endpoint must use HTTPS when sending Access credentials"
            )
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.client_id = client_id
        self.client_secret = client_secret
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_base_delay = retry_base_delay
        self._transport = transport or _urllib_transport
        self._sleep = sleep

    @classmethod
    def from_environment(cls, endpoint: str | None = None) -> "ContextFinderClient":
        client_id, client_secret, _source = resolve_access_credentials()
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            endpoint=endpoint or os.environ.get("CONTEXT_FINDER_ENDPOINT") or DEFAULT_ENDPOINT,
        )

    def refine_result(
        self,
        result: SearchResult,
        *,
        checkpoint_dir: Path | str | None = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        retain_detailed_artifacts: bool = False,
        reuse_checkpoints: bool = True,
        cancel_check: CancelCheck | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> SearchResult:
        workers = max(1, min(MAX_WORKERS, int(max_workers)))
        regions = tuple(result.regions)
        if not regions:
            return result
        completed: dict[str, ContextRegion] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="context-glm") as executor:
            futures = {
                executor.submit(
                    self._refine_region_safely,
                    region,
                    checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
                    retain_detailed_artifacts=retain_detailed_artifacts,
                    reuse_checkpoints=reuse_checkpoints,
                    cancel_check=cancel_check,
                ): region
                for region in regions
            }
            done_count = 0
            for future in as_completed(futures):
                raise_if_cancelled(cancel_check, phase="context boundary refinement")
                region, status = future.result()
                completed[region.region_id] = region
                done_count += 1
                if progress_callback is not None:
                    progress_callback(done_count, len(regions), region, status)
        return result.with_regions(tuple(completed[region.region_id] for region in regions))

    def _refine_region_safely(
        self,
        region: ContextRegion,
        *,
        checkpoint_dir: Path | None,
        retain_detailed_artifacts: bool,
        reuse_checkpoints: bool,
        cancel_check: CancelCheck | None,
    ) -> tuple[ContextRegion, str]:
        try:
            return self._refine_region(
                region,
                checkpoint_dir=checkpoint_dir,
                retain_detailed_artifacts=retain_detailed_artifacts,
                reuse_checkpoints=reuse_checkpoints,
                cancel_check=cancel_check,
            )
        except PipelineCancelledError:
            raise
        except Exception as exc:
            fallback = apply_boundary_selection(
                region,
                region.broad_start_paragraph,
                region.broad_end_paragraph,
                method="deterministic_context_window",
                note="GLM boundary refinement unavailable; deterministic source context retained",
            )
            return fallback, f"deterministic fallback ({type(exc).__name__})"

    def _refine_region(
        self,
        region: ContextRegion,
        *,
        checkpoint_dir: Path | None,
        retain_detailed_artifacts: bool,
        reuse_checkpoints: bool,
        cancel_check: CancelCheck | None,
    ) -> tuple[ContextRegion, str]:
        raise_if_cancelled(cancel_check, phase="context boundary preparation")
        requests = _prepare_requests(region)
        fingerprint = _region_fingerprint(region)
        checkpoint_path = (
            checkpoint_dir / BOUNDARY_PROFILE / f"{region.region_id}.json"
            if checkpoint_dir is not None
            else None
        )
        if checkpoint_path is not None and reuse_checkpoints:
            resumed = self._load_checkpoint(
                checkpoint_path,
                region=region,
                fingerprint=fingerprint,
                requests=requests,
            )
            if resumed is not None:
                return resumed, "resumed verified boundary checkpoint"

        replies: list[_ValidatedReply] = []
        for index, prepared in enumerate(requests, start=1):
            raise_if_cancelled(
                cancel_check,
                phase=f"context hit {index} of {len(requests)}",
            )
            decoded, raw_response = self._post(prepared, cancel_check=cancel_check)
            replies.append(self._validate_reply(prepared, decoded, raw_response))

        start = min(reply.start_paragraph for reply in replies)
        end = max(reply.end_paragraph for reply in replies)
        all_glm = all(reply.method == "glm" for reply in replies)
        confidence = min(reply.confidence for reply in replies) if all_glm else None
        refined = apply_boundary_selection(
            region,
            start,
            end,
            method=("glm_boundary_refinement" if all_glm else "deterministic_context_window"),
            model=(BOUNDARY_MODEL if all_glm else None),
            confidence=confidence,
            note=(None if all_glm else "Service used deterministic source boundaries"),
        )
        if checkpoint_path is not None and all_glm:
            self._write_checkpoint(
                checkpoint_path,
                region=refined,
                fingerprint=fingerprint,
                requests=requests,
                replies=replies,
                retain_detailed_artifacts=retain_detailed_artifacts,
            )
        return refined, (
            "GLM boundaries verified" if all_glm else "deterministic service fallback"
        )

    def _post(
        self,
        prepared: _PreparedRequest,
        *,
        cancel_check: CancelCheck | None,
    ) -> tuple[Mapping[str, Any], bytes]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            ACCESS_CLIENT_ID_HEADER: self.client_id,
            ACCESS_CLIENT_SECRET_HEADER: self.client_secret,
            "Idempotency-Key": prepared.body_sha256,
            "User-Agent": "AudioProcessor-ContextFinderClient/1",
        }
        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            raise_if_cancelled(cancel_check, phase="context endpoint request")
            try:
                response = self._transport(
                    "POST", self.endpoint, headers, prepared.body, self.timeout
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    self._sleep(self.retry_base_delay * (2**attempt))
                    continue
                break
            if len(response.body) > MAX_RESPONSE_BYTES:
                raise ContextFinderProtocolError("context service response is too large")
            if _looks_like_access_response(response):
                raise ContextFinderAccessError(
                    "Cloudflare Access rejected the stored service-token credentials"
                )
            if response.status == 429 or 500 <= response.status <= 599:
                if attempt + 1 < self.max_attempts:
                    self._sleep(self.retry_base_delay * (2**attempt))
                    continue
            if not 200 <= response.status < 300:
                raise ContextFinderNetworkError(
                    f"context service returned HTTP {response.status}"
                )
            try:
                decoded = json.loads(response.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContextFinderProtocolError(
                    "context service returned invalid JSON"
                ) from exc
            if not isinstance(decoded, Mapping):
                raise ContextFinderProtocolError("context service response is not an object")
            raise_if_cancelled(cancel_check, phase="context endpoint response")
            return decoded, response.body
        raise ContextFinderNetworkError(
            f"context request failed after {self.max_attempts} attempts"
        ) from last_error

    def _validate_reply(
        self,
        prepared: _PreparedRequest,
        response: Mapping[str, Any],
        raw_response: bytes,
    ) -> _ValidatedReply:
        if response.get("boundary_profile") != BOUNDARY_PROFILE:
            raise ContextFinderProtocolError("service returned an unexpected boundary profile")
        if response.get("model") != BOUNDARY_MODEL:
            raise ContextFinderProtocolError("service returned an unexpected model")
        if response.get("offset_unit") != "utf16_code_unit":
            raise ContextFinderProtocolError("service returned an unexpected offset unit")
        if response.get("hit") != prepared.payload["hit"]:
            raise ContextFinderProtocolError("service returned a different occurrence")

        selection = response.get("selection")
        decision = response.get("decision")
        integrity = response.get("integrity")
        if not all(isinstance(item, Mapping) for item in (selection, decision, integrity)):
            raise ContextFinderProtocolError("service response metadata is incomplete")
        assert isinstance(selection, Mapping)
        assert isinstance(decision, Mapping)
        assert isinstance(integrity, Mapping)
        selection_keys = (
            "startParagraphNumber",
            "startOffset",
            "endParagraphNumber",
            "endOffset",
        )
        if any(not _is_int(selection.get(key)) for key in selection_keys):
            raise ContextFinderProtocolError("service returned non-integer boundaries")
        paragraph_map = {
            item["number"]: item["text"] for item in prepared.payload["paragraphs"]
        }
        numbers = [item["number"] for item in prepared.payload["paragraphs"]]
        start_number = int(selection["startParagraphNumber"])
        end_number = int(selection["endParagraphNumber"])
        if start_number not in paragraph_map or end_number not in paragraph_map:
            raise ContextFinderProtocolError("service selected an unknown paragraph")
        start_index = numbers.index(start_number)
        end_index = numbers.index(end_number)
        hit_number = int(prepared.payload["hit"]["paragraphNumber"])
        hit_index = numbers.index(hit_number)
        if start_index > hit_index or end_index < hit_index or end_index < start_index:
            raise ContextFinderProtocolError("service selection excludes the occurrence")
        start_text = paragraph_map[start_number]
        end_text = paragraph_map[end_number]
        start_offset = _python_offset(start_text, int(selection["startOffset"]))
        end_offset = _python_offset(end_text, int(selection["endOffset"]))
        if start_index == end_index and end_offset <= start_offset:
            raise ContextFinderProtocolError("service returned an empty selection")
        hit_start = int(prepared.payload["hit"]["startOffset"])
        hit_end = int(prepared.payload["hit"]["endOffset"])
        if (start_index == hit_index and int(selection["startOffset"]) > hit_start) or (
            end_index == hit_index and int(selection["endOffset"]) < hit_end
        ):
            raise ContextFinderProtocolError("service offsets exclude the occurrence")

        chosen_parts: list[str] = []
        for index in range(start_index, end_index + 1):
            text = paragraph_map[numbers[index]]
            left = start_offset if index == start_index else 0
            right = end_offset if index == end_index else len(text)
            chosen_parts.append(text[left:right])
        chosen_text = "\n\n".join(chosen_parts)
        source_json = _compact_json(prepared.payload["paragraphs"])
        matched_paragraph = paragraph_map[hit_number]
        matched_start = _python_offset(matched_paragraph, hit_start)
        matched_end = _python_offset(matched_paragraph, hit_end)
        matched_text = matched_paragraph[matched_start:matched_end]
        expected_integrity = {
            "source_sha256": _sha256_text(source_json),
            "source_bytes": len(source_json.encode("utf-8")),
            "match_sha256": _sha256_text(matched_text),
            "selected_sha256": _sha256_text(chosen_text),
            "selected_bytes": len(chosen_text.encode("utf-8")),
        }
        for key, expected in expected_integrity.items():
            if integrity.get(key) != expected:
                raise ContextFinderProtocolError(f"service returned mismatched {key}")
        model_response_hash = integrity.get("model_response_sha256")
        if not isinstance(model_response_hash, str) or not _SHA256_RE.fullmatch(model_response_hash):
            raise ContextFinderProtocolError("service returned invalid model provenance")
        method = decision.get("method")
        confidence = decision.get("confidence")
        if method not in {"glm", "deterministic_fallback"}:
            raise ContextFinderProtocolError("service returned an invalid decision method")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            raise ContextFinderProtocolError("service returned invalid confidence")
        return _ValidatedReply(
            start_paragraph=start_number,
            end_paragraph=end_number,
            confidence=float(confidence),
            method=str(method),
            response_sha256=_sha256_bytes(raw_response),
            response_metadata=response,
        )

    def _load_checkpoint(
        self,
        path: Path,
        *,
        region: ContextRegion,
        fingerprint: str,
        requests: Sequence[_PreparedRequest],
    ) -> ContextRegion | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                return None
            if (
                data.get("checkpoint_version") != CHECKPOINT_VERSION
                or data.get("boundary_profile") != BOUNDARY_PROFILE
                or data.get("model") != BOUNDARY_MODEL
                or data.get("region_id") != region.region_id
                or data.get("region_fingerprint") != fingerprint
                or data.get("occurrence_ids")
                != [request.occurrence_id for request in requests]
                or data.get("request_sha256s")
                != [request.body_sha256 for request in requests]
            ):
                return None
            selection = data.get("selection")
            if not isinstance(selection, Mapping):
                return None
            start = selection.get("start_paragraph")
            end = selection.get("end_paragraph")
            confidence = selection.get("confidence")
            if not _is_int(start) or not _is_int(end):
                return None
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                return None
            resumed = apply_boundary_selection(
                region,
                int(start),
                int(end),
                method="glm_boundary_refinement",
                model=BOUNDARY_MODEL,
                confidence=float(confidence),
            )
            if data.get("selected_sha256") != _sha256_text(
                _selection_text(resumed, int(start), int(end))
            ):
                return None
            return resumed
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _write_checkpoint(
        self,
        path: Path,
        *,
        region: ContextRegion,
        fingerprint: str,
        requests: Sequence[_PreparedRequest],
        replies: Sequence[_ValidatedReply],
        retain_detailed_artifacts: bool,
    ) -> None:
        data: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "boundary_profile": BOUNDARY_PROFILE,
            "model": BOUNDARY_MODEL,
            "region_id": region.region_id,
            "region_fingerprint": fingerprint,
            "occurrence_ids": [request.occurrence_id for request in requests],
            "request_sha256s": [request.body_sha256 for request in requests],
            "response_sha256s": [reply.response_sha256 for reply in replies],
            "selection": {
                "start_paragraph": region.selection.start_paragraph,
                "end_paragraph": region.selection.end_paragraph,
                "confidence": region.selection.confidence,
            },
            "selected_sha256": _sha256_text(
                _selection_text(
                    region,
                    region.selection.start_paragraph,
                    region.selection.end_paragraph,
                )
            ),
        }
        if retain_detailed_artifacts:
            data["responses"] = [reply.response_metadata for reply in replies]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _default_checkpoint_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    return base / "AudioProcessor" / "context-finder-checkpoints"


def _remove_completed_checkpoints(
    checkpoint_dir: Path, regions: Sequence[ContextRegion]
) -> None:
    """Remove only the exact compact checkpoint files for a completed result."""

    profile_dir = checkpoint_dir / BOUNDARY_PROFILE
    for region in regions:
        try:
            (profile_dir / f"{region.region_id}.json").unlink(missing_ok=True)
        except OSError:
            # Retention cleanup must not invalidate a correct in-memory result.
            pass
    try:
        profile_dir.rmdir()
    except OSError:
        pass


def refine_result_with_glm(
    result: SearchResult,
    *,
    cancel_check: CancelCheck | None = None,
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | str | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    retain_detailed_artifacts: bool = False,
    retain_checkpoints: bool = False,
    reuse_checkpoints: bool = True,
    client: ContextFinderClient | None = None,
) -> SearchResult:
    """Refine every region safely; individual failures retain broad source bounds."""

    active_client = client or ContextFinderClient.from_environment()
    resolved_checkpoint_dir = Path(checkpoint_dir or _default_checkpoint_dir())
    refined = active_client.refine_result(
        result,
        checkpoint_dir=resolved_checkpoint_dir,
        max_workers=max_workers,
        retain_detailed_artifacts=retain_detailed_artifacts,
        reuse_checkpoints=reuse_checkpoints,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    if not (retain_checkpoints or retain_detailed_artifacts):
        _remove_completed_checkpoints(resolved_checkpoint_dir, result.regions)
    return refined


__all__ = [
    "BOUNDARY_MODEL",
    "BOUNDARY_PROFILE",
    "ContextFinderAccessError",
    "ContextFinderClient",
    "ContextFinderClientError",
    "ContextFinderConfigurationError",
    "ContextFinderNetworkError",
    "ContextFinderProtocolError",
    "DEFAULT_ENDPOINT",
    "refine_result_with_glm",
]
