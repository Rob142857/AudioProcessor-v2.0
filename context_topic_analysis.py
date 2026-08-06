"""Resumable GLM topic analysis for an exact Context Finder compilation.

The local JSONL/DOCX binding remains the text authority.  This module sends
only bounded, immutable selected-paragraph snapshots to the protected topic
analysis endpoint, validates every structured response, and writes a proposed
analysis JSON.  It never rewrites quotations or produces reader documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from cleanup_client import (
    ACCESS_CLIENT_ID_HEADER,
    ACCESS_CLIENT_SECRET_HEADER,
    HttpResponse,
    resolve_access_credentials,
)
from context_compilation_inventory import (
    BoundContextCompilation,
    BoundContextRegion,
    bind_context_compilation,
)
from context_finder import compile_query_pattern, validate_query
from pipeline_control import CancelCheck, raise_if_cancelled


TOPIC_ANALYSIS_PROFILE = "context-topic-analysis-v1"
TOPIC_ANALYSIS_MODEL = "@cf/zai-org/glm-4.7-flash"
ANALYSIS_SCHEMA_VERSION = "context-topic-analysis-proposed-v1"
CHECKPOINT_VERSION = 1
DEFAULT_ENDPOINT = (
    "https://pg.objectiveartefacts.com.au/api/tooling/context-topic-analysis"
)
MAX_BATCH_REGIONS = 20
MAX_REGION_PAYLOAD_BYTES = 40_000
MAX_REQUEST_BYTES = 64_000
MAX_RESPONSE_BYTES = 384_000
MIN_FAMILIES = 6
MAX_FAMILIES = 10
MIN_TOPICS = 12
MAX_TOPICS = 18
MAX_SECONDARY_TOPICS = 2
MAX_CANDIDATE_DESCRIPTION_CHARS = 360
MAX_CANDIDATE_ALIASES = 8
MAX_REPRESENTATIVE_REGIONS = 8
SYNTHESIS_SUMMARY_BUDGET_BYTES = 39_000
MAX_CANDIDATE_SUMMARIES = 200
DEFAULT_READING_WORDS_PER_PAGE = 450
DEFAULT_MASTER_LAYOUT_PAGES = 490
UNCLASSIFIED_TOPIC_ID = "unclassified_needs_review"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
_ACCESS_MARKERS = (
    "cloudflare access",
    "cloudflareaccess.com",
    "/cdn-cgi/access/login",
    "cf-access",
)
AWAKENING_ROLES = frozenset(
    {
        "definition",
        "process_or_stage",
        "practice_or_instruction",
        "state_or_experience",
        "contrast_with_sleep",
        "transformation_or_rebirth",
        "collective_or_historical",
        "analogy_or_symbol",
        "passing_reference",
        "unclear",
    }
)
AMBIGUITY_CODES = frozenset(
    {
        "mixed_passage",
        "taxonomy_overlap",
        "insufficient_context",
        "boundary_uncertain",
        "taxonomy_gap",
    }
)

Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], HttpResponse]
ProgressCallback = Callable[[int, int, str, str], None]


class ContextTopicAnalysisError(RuntimeError):
    """Base class for topic-analysis failures."""


class ContextTopicAnalysisConfigurationError(ContextTopicAnalysisError):
    pass


class ContextTopicAnalysisInputError(ContextTopicAnalysisError):
    pass


class ContextTopicAnalysisNetworkError(ContextTopicAnalysisError):
    pass


class ContextTopicAnalysisAccessError(ContextTopicAnalysisError):
    pass


class ContextTopicAnalysisProtocolError(ContextTopicAnalysisError):
    pass


class ContextTopicAnalysisResumeError(ContextTopicAnalysisError):
    pass


@dataclass(frozen=True, slots=True)
class TopicAnalysisOutcome:
    output_path: Path
    checkpoint_dir: Path
    region_count: int
    unique_text_count: int
    duplicate_region_count: int
    taxonomy_sha256: str
    family_count: int
    topic_count: int
    classified_count: int
    unclassified_review_count: int
    review_required_count: int
    resumed_batches: int
    network_batches: int


@dataclass(frozen=True, slots=True)
class _UniqueRegion:
    representative: BoundContextRegion
    members: tuple[BoundContextRegion, ...]
    payload: Mapping[str, Any]
    region_input_sha256: str
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class _Batch:
    batch_id: str
    regions: tuple[_UniqueRegion, ...]


def _canonical_json(value: Any) -> str:
    """Cross-language canonical JSON: sorted keys, compact, Unicode preserved."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _wire_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalise_integral_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == wanted), None)


def _looks_like_access_response(response: HttpResponse) -> bool:
    location = (_header(response.headers, "Location") or "").casefold()
    if 300 <= response.status < 400 and any(x in location for x in _ACCESS_MARKERS):
        return True
    content_type = (_header(response.headers, "Content-Type") or "").casefold()
    sample = response.body[:32_768].decode("utf-8", errors="ignore").casefold()
    looks_html = "text/html" in content_type or "<html" in sample or "<!doctype html" in sample
    return looks_html and (
        any(x in sample for x in _ACCESS_MARKERS)
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


class ContextTopicAnalysisClient:
    """Strict Cloudflare Access client for the four topic-analysis operations."""

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
            raise ContextTopicAnalysisConfigurationError(
                "Cloudflare Access credentials are required"
            )
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ContextTopicAnalysisConfigurationError(
                "topic-analysis endpoint must be an HTTP(S) URL"
            )
        if parsed.scheme != "https" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ContextTopicAnalysisConfigurationError(
                "topic-analysis endpoint must use HTTPS when sending Access credentials"
            )
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.client_id = client_id
        self.client_secret = client_secret
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.max_attempts = int(max_attempts)
        self.retry_base_delay = float(retry_base_delay)
        self._transport = transport or _urllib_transport
        self._sleep = sleep

    @classmethod
    def from_environment(
        cls, endpoint: str | None = None
    ) -> "ContextTopicAnalysisClient":
        client_id, client_secret, _source = resolve_access_credentials()
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            endpoint=(
                endpoint
                or os.environ.get("CONTEXT_TOPIC_ANALYSIS_ENDPOINT")
                or DEFAULT_ENDPOINT
            ),
        )

    def request(
        self,
        payload: Mapping[str, Any],
        *,
        cancel_check: CancelCheck | None = None,
    ) -> Mapping[str, Any]:
        body = _wire_json(payload)
        if len(body) > MAX_REQUEST_BYTES:
            raise ContextTopicAnalysisInputError(
                f"topic-analysis request exceeds {MAX_REQUEST_BYTES} bytes"
            )
        request_sha256 = _sha256_bytes(body)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            ACCESS_CLIENT_ID_HEADER: self.client_id,
            ACCESS_CLIENT_SECRET_HEADER: self.client_secret,
            "Idempotency-Key": request_sha256,
            "User-Agent": "AudioProcessor-ContextTopicAnalysis/1",
        }
        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            raise_if_cancelled(cancel_check, phase="topic-analysis endpoint request")
            try:
                response = self._transport(
                    "POST", self.endpoint, headers, body, self.timeout
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    self._sleep(self.retry_base_delay * (2**attempt))
                    continue
                break
            if len(response.body) > MAX_RESPONSE_BYTES:
                raise ContextTopicAnalysisProtocolError(
                    "topic-analysis response is too large"
                )
            if _looks_like_access_response(response):
                raise ContextTopicAnalysisAccessError(
                    "Cloudflare Access rejected the stored service-token credentials"
                )
            if response.status == 429 or 500 <= response.status <= 599:
                if attempt + 1 < self.max_attempts:
                    self._sleep(self.retry_base_delay * (2**attempt))
                    continue
            if not 200 <= response.status < 300:
                raise ContextTopicAnalysisNetworkError(
                    f"topic-analysis service returned HTTP {response.status}"
                )
            try:
                decoded = json.loads(response.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContextTopicAnalysisProtocolError(
                    "topic-analysis service returned invalid JSON"
                ) from exc
            if not isinstance(decoded, Mapping):
                raise ContextTopicAnalysisProtocolError(
                    "topic-analysis response is not an object"
                )
            self._validate_wrapper(
                payload,
                decoded,
                request_sha256=request_sha256,
                request_bytes=len(body),
            )
            return decoded
        raise ContextTopicAnalysisNetworkError(
            f"topic-analysis request failed after {self.max_attempts} attempts"
        ) from last_error

    @staticmethod
    def _validate_wrapper(
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        *,
        request_sha256: str,
        request_bytes: int,
    ) -> None:
        expected = {
            "topic_analysis_profile": TOPIC_ANALYSIS_PROFILE,
            "operation": request["operation"],
            "batch_id": request["batch_id"],
            "source_records_sha256": request["source_records_sha256"],
        }
        for key, value in expected.items():
            if response.get(key) != value:
                raise ContextTopicAnalysisProtocolError(
                    f"service returned a mismatched {key}"
                )
        if response.get("model") != TOPIC_ANALYSIS_MODEL:
            raise ContextTopicAnalysisProtocolError(
                "service returned an unexpected model"
            )
        result = response.get("result")
        integrity = response.get("integrity")
        if not isinstance(result, Mapping) or not isinstance(integrity, Mapping):
            raise ContextTopicAnalysisProtocolError(
                "service response result/integrity is incomplete"
            )
        if result.get("batch_id") != request["batch_id"]:
            raise ContextTopicAnalysisProtocolError(
                "service result returned a mismatched batch_id"
            )
        if integrity.get("request_sha256") != request_sha256:
            raise ContextTopicAnalysisProtocolError(
                "service returned a mismatched request_sha256"
            )
        if integrity.get("request_bytes") != request_bytes:
            raise ContextTopicAnalysisProtocolError(
                "service returned a mismatched request_bytes"
            )
        model_hash = integrity.get("model_response_sha256")
        if not isinstance(model_hash, str) or _SHA256_RE.fullmatch(model_hash) is None:
            raise ContextTopicAnalysisProtocolError(
                "service returned invalid model provenance"
            )


def _region_payload(
    region: BoundContextRegion,
    *,
    query_pattern: re.Pattern[str],
) -> dict[str, Any]:
    paragraphs = [
        {"number": item.number, "text": item.text}
        for item in region.selected_paragraphs
    ]
    hit_numbers: list[int] = []
    match_count = 0
    for item in region.selected_paragraphs:
        matches = list(query_pattern.finditer(item.text))
        match_count += len(matches)
        if matches:
            hit_numbers.append(item.number)
    if match_count != region.occurrence_count:
        raise ContextTopicAnalysisInputError(
            f"bound occurrence count changed for region {region.region_id}"
        )
    method = region.selection_method
    if method not in {"glm_boundary_refinement", "deterministic_context_window"}:
        raise ContextTopicAnalysisInputError(
            f"unsupported boundary method for region {region.region_id}: {method}"
        )
    return {
        "region_id": region.region_id,
        "selection_method": method,
        "boundary_confidence": _normalise_integral_number(
            region.selection_confidence
        ),
        "occurrence_count": region.occurrence_count,
        "hit_paragraph_numbers": hit_numbers,
        "paragraphs": paragraphs,
    }


def _deduplicate_regions(
    inventory: BoundContextCompilation,
) -> tuple[_UniqueRegion, ...]:
    query_pattern = compile_query_pattern(validate_query(inventory.query))
    grouped: MutableMapping[str, list[BoundContextRegion]] = {}
    texts: dict[str, str] = {}
    for region in inventory.regions:
        digest = region.selected_text_sha256
        previous = texts.setdefault(digest, region.selected_text)
        if previous != region.selected_text:
            raise ContextTopicAnalysisInputError(
                "selected-text SHA-256 collision detected"
            )
        grouped.setdefault(digest, []).append(region)

    unique: list[_UniqueRegion] = []
    for members in grouped.values():
        representative = members[0]
        payload = _region_payload(representative, query_pattern=query_pattern)
        payload_bytes = len(_wire_json(payload))
        unique.append(
            _UniqueRegion(
                representative=representative,
                members=tuple(members),
                payload=payload,
                region_input_sha256=_sha256_json(payload),
                payload_bytes=payload_bytes,
            )
        )
    return tuple(unique)


def _make_batches(
    unique: Sequence[_UniqueRegion], stage: str
) -> tuple[_Batch, ...]:
    batches: list[_Batch] = []
    current: list[_UniqueRegion] = []
    current_bytes = 2

    def emit() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        index = len(batches) + 1
        identity = _sha256_json(
            [item.region_input_sha256 for item in current]
        )[:12]
        batches.append(
            _Batch(
                batch_id=f"{stage}-{index:04d}-{identity}",
                regions=tuple(current),
            )
        )
        current = []
        current_bytes = 2

    for item in unique:
        if item.payload_bytes > MAX_REGION_PAYLOAD_BYTES:
            emit()
            continue
        addition = item.payload_bytes + (1 if current else 0)
        if current and (
            len(current) >= MAX_BATCH_REGIONS
            or current_bytes + addition > MAX_REGION_PAYLOAD_BYTES
        ):
            emit()
            addition = item.payload_bytes
        current.append(item)
        current_bytes += addition
    emit()
    return tuple(batches)


def _common_payload(
    inventory: BoundContextCompilation,
    *,
    operation: str,
    batch_id: str,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "profile": TOPIC_ANALYSIS_PROFILE,
        "batch_id": batch_id,
        "query": inventory.query,
        "source_records_sha256": inventory.jsonl_sha256,
    }


def _normalise_label(label: str) -> str:
    value = unicodedata.normalize("NFKC", label).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _require_text(value: Any, field: str, *, maximum: int = 2_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ContextTopicAnalysisProtocolError(f"service returned invalid {field}")
    return value.strip()


def _require_string_list(
    value: Any, field: str, *, maximum_items: int = 32, maximum_chars: int = 500
) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ContextTopicAnalysisProtocolError(f"service returned invalid {field}")
    result: list[str] = []
    for item in value:
        result.append(_require_text(item, field, maximum=maximum_chars))
    if len(set(result)) != len(result):
        raise ContextTopicAnalysisProtocolError(f"service returned duplicate {field}")
    return result


def _validate_evidence_numbers(value: Any, region: _UniqueRegion) -> list[int]:
    if not isinstance(value, list) or not all(_is_int(item) for item in value):
        raise ContextTopicAnalysisProtocolError(
            "service returned invalid evidence paragraph numbers"
        )
    allowed = {item["number"] for item in region.payload["paragraphs"]}
    numbers = [int(item) for item in value]
    if len(numbers) != len(set(numbers)) or any(item not in allowed for item in numbers):
        raise ContextTopicAnalysisProtocolError(
            "service evidence refers to an unknown paragraph"
        )
    return numbers


def _validate_roles(value: Any) -> list[str]:
    roles = _require_string_list(value, "awakening_roles", maximum_items=10)
    if any(item not in AWAKENING_ROLES for item in roles):
        raise ContextTopicAnalysisProtocolError(
            "service returned an unknown awakening role"
        )
    return roles


def _validate_candidate_cards(
    wrapper: Mapping[str, Any], batch: _Batch
) -> list[dict[str, Any]]:
    result = wrapper["result"]
    assert isinstance(result, Mapping)
    cards = result.get("cards")
    if not isinstance(cards, list) or len(cards) != len(batch.regions):
        raise ContextTopicAnalysisProtocolError(
            "candidate response does not cover its batch exactly"
        )
    expected = {item.representative.region_id: item for item in batch.regions}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in cards:
        if not isinstance(raw, Mapping):
            raise ContextTopicAnalysisProtocolError("candidate card is not an object")
        region_id = raw.get("region_id")
        if not isinstance(region_id, str) or region_id not in expected or region_id in seen:
            raise ContextTopicAnalysisProtocolError(
                "candidate response returned an unknown/duplicate region"
            )
        seen.add(region_id)
        region = expected[region_id]
        if raw.get("region_input_sha256") != region.region_input_sha256:
            raise ContextTopicAnalysisProtocolError(
                "candidate response returned mismatched region_input_sha256"
            )
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates or len(candidates) > 8:
            raise ContextTopicAnalysisProtocolError(
                "candidate card must contain one to eight candidates"
            )
        clean_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ContextTopicAnalysisProtocolError("candidate is not an object")
            clean_candidates.append(
                {
                    "label": _require_text(candidate.get("label"), "candidate label", maximum=160),
                    "description": _require_text(
                        candidate.get("description"),
                        "candidate description",
                        maximum=1_000,
                    ),
                    "evidence_paragraph_numbers": _validate_evidence_numbers(
                        candidate.get("evidence_paragraph_numbers"), region
                    ),
                }
            )
        mixed = raw.get("mixed_section")
        if not isinstance(mixed, bool):
            raise ContextTopicAnalysisProtocolError(
                "candidate card mixed_section must be boolean"
            )
        output.append(
            {
                "region_id": region_id,
                "region_input_sha256": region.region_input_sha256,
                "candidates": clean_candidates,
                "awakening_roles": _validate_roles(raw.get("awakening_roles")),
                "mixed_section": mixed,
            }
        )
    return output


def _aggregate_candidate_summaries(
    cards: Sequence[Mapping[str, Any]],
    ordinal_by_region: Mapping[str, int],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for card in cards:
        region_id = str(card["region_id"])
        for raw in card["candidates"]:
            assert isinstance(raw, Mapping)
            label = str(raw["label"])
            normalised = _normalise_label(label)
            if not normalised:
                raise ContextTopicAnalysisProtocolError(
                    "candidate label normalises to an empty value"
                )
            candidate_key = f"candidate_{_sha256_bytes(normalised.encode('utf-8'))[:16]}"
            group = groups.setdefault(
                candidate_key,
                {
                    "candidate_key": candidate_key,
                    "normalised": normalised,
                    "labels": {},
                    "descriptions": {},
                    "regions": set(),
                },
            )
            group["labels"][label] = group["labels"].get(label, 0) + 1
            description = str(raw["description"]).strip()[:MAX_CANDIDATE_DESCRIPTION_CHARS]
            group["descriptions"][description] = group["descriptions"].get(description, 0) + 1
            group["regions"].add(region_id)
    summaries: list[dict[str, Any]] = []
    for group in groups.values():
        labels = sorted(
            group["labels"],
            key=lambda value: (-group["labels"][value], value.casefold(), value),
        )
        descriptions = sorted(
            group["descriptions"],
            key=lambda value: (-group["descriptions"][value], len(value), value),
        )
        regions = sorted(group["regions"], key=lambda value: ordinal_by_region[value])
        summaries.append(
            {
                "candidate_key": group["candidate_key"],
                "label": labels[0],
                "description": descriptions[0],
                "support_count": len(regions),
                "representative_region_ids": regions[:MAX_REPRESENTATIVE_REGIONS],
                "aliases": labels[1 : 1 + MAX_CANDIDATE_ALIASES],
            }
        )
    summaries.sort(key=lambda item: (-item["support_count"], item["candidate_key"]))
    return summaries


def _summaries_for_synthesis(
    inventory: BoundContextCompilation,
    summaries: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for summary in summaries:
        if len(selected) >= MAX_CANDIDATE_SUMMARIES:
            break
        proposed = [*selected, summary]
        probe = _common_payload(
            inventory, operation="taxonomy_synthesis", batch_id="taxonomy-synthesis"
        )
        probe["candidate_summaries"] = proposed
        probe["topic_limits"] = {
            "min_families": MIN_FAMILIES,
            "max_families": MAX_FAMILIES,
            "min_topics": MIN_TOPICS,
            "max_topics": MAX_TOPICS,
        }
        if len(_wire_json(probe)) > SYNTHESIS_SUMMARY_BUDGET_BYTES:
            break
        selected = proposed
    if not selected:
        raise ContextTopicAnalysisInputError(
            "candidate summaries cannot fit the synthesis request"
        )
    return selected


def _validate_taxonomy(wrapper: Mapping[str, Any]) -> tuple[dict[str, Any], str, list[Any]]:
    result = wrapper["result"]
    assert isinstance(result, Mapping)
    raw = result.get("taxonomy")
    migration = result.get("migration")
    if not isinstance(raw, Mapping) or not isinstance(migration, list):
        raise ContextTopicAnalysisProtocolError(
            "taxonomy response is incomplete"
        )
    families = raw.get("families")
    topics = raw.get("topics")
    if not isinstance(families, list) or not MIN_FAMILIES <= len(families) <= MAX_FAMILIES:
        raise ContextTopicAnalysisProtocolError("taxonomy family count is outside limits")
    if not isinstance(topics, list) or not MIN_TOPICS <= len(topics) <= MAX_TOPICS:
        raise ContextTopicAnalysisProtocolError("taxonomy topic count is outside limits")
    clean_families: list[dict[str, Any]] = []
    family_ids: set[str] = set()
    for item in families:
        if not isinstance(item, Mapping):
            raise ContextTopicAnalysisProtocolError("taxonomy family is not an object")
        family_id = _require_text(item.get("family_id"), "family_id", maximum=80)
        if _ID_RE.fullmatch(family_id) is None or family_id in family_ids:
            raise ContextTopicAnalysisProtocolError("taxonomy family_id is invalid/duplicate")
        family_ids.add(family_id)
        clean_families.append(
            {
                "family_id": family_id,
                "label": _require_text(item.get("label"), "family label", maximum=160),
                "definition": _require_text(
                    item.get("definition"), "family definition", maximum=1_500
                ),
            }
        )
    clean_topics: list[dict[str, Any]] = []
    topic_ids: set[str] = set()
    used_families: set[str] = set()
    for item in topics:
        if not isinstance(item, Mapping):
            raise ContextTopicAnalysisProtocolError("taxonomy topic is not an object")
        topic_id = _require_text(item.get("topic_id"), "topic_id", maximum=80)
        family_id = _require_text(item.get("family_id"), "topic family_id", maximum=80)
        if _ID_RE.fullmatch(topic_id) is None or topic_id in topic_ids:
            raise ContextTopicAnalysisProtocolError("taxonomy topic_id is invalid/duplicate")
        if family_id not in family_ids:
            raise ContextTopicAnalysisProtocolError("taxonomy topic references unknown family")
        topic_ids.add(topic_id)
        used_families.add(family_id)
        clean_topics.append(
            {
                "topic_id": topic_id,
                "label": _require_text(item.get("label"), "topic label", maximum=180),
                "family_id": family_id,
                "definition": _require_text(
                    item.get("definition"), "topic definition", maximum=2_000
                ),
                "aliases": _require_string_list(item.get("aliases"), "topic aliases"),
                "include_cues": _require_string_list(
                    item.get("include_cues"), "topic include_cues"
                ),
                "exclude_cues": _require_string_list(
                    item.get("exclude_cues"), "topic exclude_cues"
                ),
            }
        )
    if used_families != family_ids:
        raise ContextTopicAnalysisProtocolError(
            "taxonomy contains a family with no topics"
        )
    taxonomy = {
        "title": _require_text(raw.get("title"), "taxonomy title", maximum=240),
        "scope_note": _require_text(raw.get("scope_note"), "scope note", maximum=2_000),
        "families": clean_families,
        "topics": clean_topics,
    }
    digest = _sha256_json(taxonomy)
    if wrapper.get("taxonomy_sha256") != digest:
        raise ContextTopicAnalysisProtocolError(
            "service returned mismatched taxonomy_sha256"
        )
    return taxonomy, digest, migration


def _validate_classifications(
    wrapper: Mapping[str, Any],
    batch: _Batch,
    taxonomy: Mapping[str, Any],
    taxonomy_sha256: str,
) -> list[dict[str, Any]]:
    if wrapper.get("taxonomy_sha256") != taxonomy_sha256:
        raise ContextTopicAnalysisProtocolError(
            "classification wrapper returned mismatched taxonomy_sha256"
        )
    result = wrapper["result"]
    assert isinstance(result, Mapping)
    rows = result.get("classifications")
    if not isinstance(rows, list) or len(rows) != len(batch.regions):
        raise ContextTopicAnalysisProtocolError(
            "classification response does not cover its batch exactly"
        )
    topic_ids = {item["topic_id"] for item in taxonomy["topics"]}
    expected = {item.representative.region_id: item for item in batch.regions}
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContextTopicAnalysisProtocolError("classification row is not an object")
        region_id = row.get("region_id")
        if not isinstance(region_id, str) or region_id not in expected or region_id in seen:
            raise ContextTopicAnalysisProtocolError(
                "classification returned an unknown/duplicate region"
            )
        seen.add(region_id)
        region = expected[region_id]
        if row.get("region_input_sha256") != region.region_input_sha256:
            raise ContextTopicAnalysisProtocolError(
                "classification returned mismatched region_input_sha256"
            )
        assignments = row.get("assignments")
        if not isinstance(assignments, list) or len(assignments) > 1 + MAX_SECONDARY_TOPICS:
            raise ContextTopicAnalysisProtocolError("classification assignments are invalid")
        clean_assignments: list[dict[str, str]] = []
        primary_count = 0
        assigned_ids: set[str] = set()
        for assignment in assignments:
            if not isinstance(assignment, Mapping):
                raise ContextTopicAnalysisProtocolError("assignment is not an object")
            topic_id = assignment.get("topic_id")
            role = assignment.get("role")
            if topic_id not in topic_ids or topic_id in assigned_ids or role not in {"primary", "secondary"}:
                raise ContextTopicAnalysisProtocolError("assignment topic/role is invalid")
            assigned_ids.add(str(topic_id))
            primary_count += role == "primary"
            clean_assignments.append({"topic_id": str(topic_id), "role": str(role)})
        taxonomy_gap = row.get("taxonomy_gap")
        if not isinstance(taxonomy_gap, bool):
            raise ContextTopicAnalysisProtocolError("taxonomy_gap must be boolean")
        if primary_count != 1 and not taxonomy_gap:
            raise ContextTopicAnalysisProtocolError(
                "classification must have exactly one primary unless taxonomy_gap is true"
            )
        if primary_count > 1:
            raise ContextTopicAnalysisProtocolError("classification has multiple primaries")
        certainty = row.get("certainty")
        if certainty not in {"high", "medium", "low"}:
            raise ContextTopicAnalysisProtocolError("classification certainty is invalid")
        ambiguity = _require_string_list(
            row.get("ambiguity_codes"), "ambiguity_codes", maximum_items=5
        )
        if any(item not in AMBIGUITY_CODES for item in ambiguity):
            raise ContextTopicAnalysisProtocolError("unknown ambiguity code")
        suggested = row.get("suggested_topic_label")
        if suggested is not None:
            suggested = _require_text(suggested, "suggested_topic_label", maximum=180)
        review_status = _require_text(
            row.get("review_status"), "review_status", maximum=120
        )
        if review_status not in {"accepted", "adjudicate", "human_review"}:
            raise ContextTopicAnalysisProtocolError(
                "classification review_status is invalid"
            )
        clean.append(
            {
                "region_id": region_id,
                "region_input_sha256": region.region_input_sha256,
                "assignments": clean_assignments,
                "awakening_roles": _validate_roles(row.get("awakening_roles")),
                "evidence_paragraph_numbers": _validate_evidence_numbers(
                    row.get("evidence_paragraph_numbers"), region
                ),
                "certainty": certainty,
                "ambiguity_codes": ambiguity,
                "taxonomy_gap": taxonomy_gap,
                "suggested_topic_label": suggested,
                "review_status": review_status,
            }
        )
    return clean


def _build_refinement_feedback(
    classifications: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
    ordinal_by_region: Mapping[str, int],
) -> dict[str, Any]:
    gap_groups: dict[str, list[str]] = {}
    confusion_counts: dict[tuple[str, ...], int] = {}
    primary_counts = {item["topic_id"]: 0 for item in taxonomy["topics"]}
    for row in classifications:
        assignments = row["assignments"]
        primary = [item["topic_id"] for item in assignments if item["role"] == "primary"]
        if primary:
            primary_counts[primary[0]] += 1
        if row["taxonomy_gap"]:
            label = row.get("suggested_topic_label") or "Unspecified taxonomy gap"
            gap_groups.setdefault(str(label), []).append(str(row["region_id"]))
        if "taxonomy_overlap" in row["ambiguity_codes"] and len(assignments) > 1:
            ids = tuple(sorted(item["topic_id"] for item in assignments))
            confusion_counts[ids] = confusion_counts.get(ids, 0) + 1
    nonzero = sorted(value for value in primary_counts.values() if value)
    median = nonzero[len(nonzero) // 2] if nonzero else 0
    overbroad = sorted(
        topic_id
        for topic_id, count in primary_counts.items()
        if count > max(10, math.ceil(len(classifications) * 0.25), median * 2)
    )
    underused = sorted(topic_id for topic_id, count in primary_counts.items() if count == 0)
    gaps = []
    for label, ids in sorted(gap_groups.items(), key=lambda item: (-len(item[1]), item[0].casefold())):
        ordered = sorted(ids, key=lambda item: ordinal_by_region[item])
        gaps.append(
            {
                "suggested_label": label,
                "count": len(ids),
                "representative_region_ids": ordered[:MAX_REPRESENTATIVE_REGIONS],
            }
        )
    confusions = [
        {"topic_ids": list(ids), "count": count}
        for ids, count in sorted(
            confusion_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "gaps": gaps,
        "confusions": confusions,
        "overbroad_topic_ids": overbroad,
        "underused_topic_ids": underused,
    }


def _feedback_has_work(feedback: Mapping[str, Any]) -> bool:
    return any(bool(feedback[key]) for key in feedback)


def _map_evidence_to_member(
    representative: BoundContextRegion,
    member: BoundContextRegion,
    evidence_numbers: Sequence[int],
) -> tuple[list[int], bool]:
    rep = [item for item in representative.selected_paragraphs if item.emitted]
    target = [item for item in member.selected_paragraphs if item.emitted]
    if len(rep) != len(target) or any(a.text != b.text for a, b in zip(rep, target)):
        return [], False
    by_number = {item.number: index for index, item in enumerate(rep)}
    mapped: list[int] = []
    for number in evidence_numbers:
        index = by_number.get(number)
        if index is None:
            return [], False
        mapped.append(target[index].number)
    return mapped, True


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
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
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _default_output_path(docx_path: Path) -> Path:
    return docx_path.with_name(f"{docx_path.stem} - Proposed Topic Analysis.json")


def _default_checkpoint_dir(output_path: Path) -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    return base / "AudioProcessor" / "context-topic-analysis" / output_path.stem


def _new_state(
    inventory: BoundContextCompilation,
    unique: Sequence[_UniqueRegion],
    *,
    endpoint: str,
    master_layout_pages: int | None,
    reading_words_per_page: int,
) -> dict[str, Any]:
    duplicate_groups = [
        {
            "selected_text_sha256": item.representative.selected_text_sha256,
            "representative_region_id": item.representative.region_id,
            "region_ids": [member.region_id for member in item.members],
        }
        for item in unique
        if len(item.members) > 1
    ]
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "in_progress",
        "topic_analysis_profile": TOPIC_ANALYSIS_PROFILE,
        "analysis_profile": TOPIC_ANALYSIS_PROFILE,
        "model": TOPIC_ANALYSIS_MODEL,
        "query": inventory.query,
        "recommendation": (
            "Retain the exact-source master compilation as the complete authority, "
            "and use the proposed subtopics as smaller reading and review volumes."
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "inventory_pair_fingerprint": inventory.pair_fingerprint,
            "master_docx_path": str(inventory.docx_path),
            "master_jsonl_path": str(inventory.jsonl_path),
            "master_docx_sha256": inventory.docx_sha256,
            "source_records_sha256": inventory.jsonl_sha256,
            "ordered_regions_sha256": inventory.ordered_regions_sha256,
            "query": inventory.query,
            "region_count": inventory.region_count,
            "occurrence_count": inventory.occurrence_count,
            "source_count": inventory.source_count,
            "master_layout_pages": master_layout_pages,
            "reading_words_per_page": reading_words_per_page,
        },
        "deduplication": {
            "method": "exact selected_text_sha256",
            "region_count": inventory.region_count,
            "unique_text_count": len(unique),
            "duplicate_region_count": inventory.region_count - len(unique),
            "duplicate_groups": duplicate_groups,
        },
        "endpoint": endpoint,
        "candidate_summaries": [],
        "candidate_summary_count_used_for_synthesis": 0,
        "taxonomy": None,
        "taxonomy_sha256": None,
        "taxonomy_migration": [],
        "regions": [],
        "overlaps": [],
        "boundary_reviews": [],
        "coverage": {},
        "resume": {
            "candidate_cards": {},
            "classification": {},
            "taxonomy_synthesis_complete": False,
            "taxonomy_refinement_complete": False,
        },
    }


def _load_state(
    path: Path,
    inventory: BoundContextCompilation,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextTopicAnalysisResumeError(
            "existing proposed-analysis JSON is unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise ContextTopicAnalysisResumeError(
            "existing proposed-analysis JSON is not an object"
        )
    corpus = value.get("corpus")
    if (
        value.get("schema_version") != ANALYSIS_SCHEMA_VERSION
        or value.get("topic_analysis_profile") != TOPIC_ANALYSIS_PROFILE
        or not isinstance(corpus, Mapping)
        or corpus.get("inventory_pair_fingerprint") != inventory.pair_fingerprint
        or corpus.get("source_records_sha256") != inventory.jsonl_sha256
    ):
        raise ContextTopicAnalysisResumeError(
            "existing proposed analysis belongs to a different compilation"
        )
    return value


def _checkpoint_path(checkpoint_dir: Path, stage: str, batch_id: str) -> Path:
    return checkpoint_dir / stage / f"{batch_id}.json"


def _write_checkpoint(
    checkpoint_dir: Path,
    *,
    inventory: BoundContextCompilation,
    stage: str,
    batch_id: str,
    request_payload: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    item_ids: Sequence[str],
) -> None:
    # Deliberately contains no credentials, raw source text, descriptions, or
    # raw/model response prose.  The proposed-analysis JSON owns structured state.
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "topic_analysis_profile": TOPIC_ANALYSIS_PROFILE,
        "inventory_pair_fingerprint": inventory.pair_fingerprint,
        "source_records_sha256": inventory.jsonl_sha256,
        "stage": stage,
        "batch_id": batch_id,
        "request_sha256": wrapper["integrity"]["request_sha256"],
        "request_bytes": wrapper["integrity"]["request_bytes"],
        "response_sha256": _sha256_json(wrapper),
        "item_ids": list(item_ids),
    }
    _atomic_write_json(_checkpoint_path(checkpoint_dir, stage, batch_id), checkpoint)


def _progress(
    callback: ProgressCallback | None,
    completed: int,
    total: int,
    phase: str,
    detail: str,
) -> None:
    if callback is not None:
        callback(completed, total, phase, detail)


def analyse_context_topics(
    docx_path: Path | str,
    jsonl_path: Path | str | None = None,
    *,
    output_path: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    endpoint: str | None = None,
    client: ContextTopicAnalysisClient | Any | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    reuse_existing: bool = True,
    refine_taxonomy: bool = True,
    master_layout_pages: int | None = DEFAULT_MASTER_LAYOUT_PAGES,
    reading_words_per_page: int = DEFAULT_READING_WORDS_PER_PAGE,
) -> TopicAnalysisOutcome:
    """Analyse one bound master pair and atomically write a proposed taxonomy.

    The supplied ``client`` may be a test double exposing ``request(payload,
    cancel_check=...)``.  Without a client, stored Cloudflare Access credentials
    are resolved lazily after the compilation pair has passed exact validation.
    """

    if master_layout_pages is not None and (
        not _is_int(master_layout_pages) or master_layout_pages < 1
    ):
        raise ValueError("master_layout_pages must be a positive integer or None")
    if not _is_int(reading_words_per_page) or reading_words_per_page < 100:
        raise ValueError("reading_words_per_page must be an integer of at least 100")
    inventory = bind_context_compilation(docx_path, jsonl_path)
    unique = _deduplicate_regions(inventory)
    output = Path(output_path).expanduser().resolve() if output_path else _default_output_path(inventory.docx_path)
    checkpoints = (
        Path(checkpoint_dir).expanduser().resolve()
        if checkpoint_dir
        else _default_checkpoint_dir(output)
    )
    active_endpoint = (
        endpoint
        or (getattr(client, "endpoint", None) if client is not None else None)
        or os.environ.get("CONTEXT_TOPIC_ANALYSIS_ENDPOINT")
        or DEFAULT_ENDPOINT
    )
    state = _load_state(output, inventory) if reuse_existing else None
    if state is None:
        state = _new_state(
            inventory,
            unique,
            endpoint=active_endpoint,
            master_layout_pages=master_layout_pages,
            reading_words_per_page=reading_words_per_page,
        )
        _atomic_write_json(output, state)
    elif state.get("status") == "complete":
        return _outcome_from_state(output, checkpoints, state, resumed_batches=0, network_batches=0)
    active_client = client or ContextTopicAnalysisClient.from_environment(endpoint)

    ordinal_by_region = {item.region_id: item.ordinal for item in inventory.regions}
    by_representative = {item.representative.region_id: item for item in unique}
    candidate_batches = _make_batches(unique, "candidate")
    classification_batches = _make_batches(unique, "classification")
    oversize = {
        item.representative.region_id: item
        for item in unique
        if item.payload_bytes > MAX_REGION_PAYLOAD_BYTES
    }
    resumed_batches = 0
    network_batches = 0
    total_steps = len(candidate_batches) + len(classification_batches) + 1
    completed_steps = 0

    candidate_cards: dict[str, Mapping[str, Any]] = {
        key: value
        for key, value in state["resume"]["candidate_cards"].items()
        if key in by_representative
    }
    for batch in candidate_batches:
        raise_if_cancelled(cancel_check, phase="candidate-card analysis")
        if all(item.representative.region_id in candidate_cards for item in batch.regions):
            resumed_batches += 1
            completed_steps += 1
            _progress(progress_callback, completed_steps, total_steps, "candidate_cards", f"resumed {batch.batch_id}")
            continue
        payload = _common_payload(inventory, operation="candidate_cards", batch_id=batch.batch_id)
        payload["regions"] = [item.payload for item in batch.regions]
        wrapper = active_client.request(payload, cancel_check=cancel_check)
        cards = _validate_candidate_cards(wrapper, batch)
        for card in cards:
            candidate_cards[card["region_id"]] = card
        state["resume"]["candidate_cards"] = candidate_cards
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(output, state)
        _write_checkpoint(
            checkpoints,
            inventory=inventory,
            stage="candidate_cards",
            batch_id=batch.batch_id,
            request_payload=payload,
            wrapper=wrapper,
            item_ids=[item.representative.region_id for item in batch.regions],
        )
        network_batches += 1
        completed_steps += 1
        _progress(progress_callback, completed_steps, total_steps, "candidate_cards", batch.batch_id)

    summaries = _aggregate_candidate_summaries(
        list(candidate_cards.values()), ordinal_by_region
    )
    synthesis_summaries = _summaries_for_synthesis(inventory, summaries)
    state["candidate_summaries"] = summaries
    state["candidate_summary_count_used_for_synthesis"] = len(synthesis_summaries)

    taxonomy = state.get("taxonomy")
    taxonomy_sha256 = state.get("taxonomy_sha256")
    if not state["resume"].get("taxonomy_synthesis_complete"):
        batch_id = f"taxonomy-synthesis-{_sha256_json(synthesis_summaries)[:12]}"
        payload = _common_payload(
            inventory, operation="taxonomy_synthesis", batch_id=batch_id
        )
        payload["candidate_summaries"] = synthesis_summaries
        payload["topic_limits"] = {
            "min_families": MIN_FAMILIES,
            "max_families": MAX_FAMILIES,
            "min_topics": MIN_TOPICS,
            "max_topics": MAX_TOPICS,
        }
        wrapper = active_client.request(payload, cancel_check=cancel_check)
        taxonomy, taxonomy_sha256, migration = _validate_taxonomy(wrapper)
        state["taxonomy"] = taxonomy
        state["taxonomy_sha256"] = taxonomy_sha256
        state["taxonomy_migration"] = migration
        state["resume"]["taxonomy_synthesis_complete"] = True
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(output, state)
        _write_checkpoint(
            checkpoints,
            inventory=inventory,
            stage="taxonomy_synthesis",
            batch_id=batch_id,
            request_payload=payload,
            wrapper=wrapper,
            item_ids=[item["candidate_key"] for item in synthesis_summaries],
        )
        network_batches += 1
    else:
        if not isinstance(taxonomy, Mapping) or not isinstance(taxonomy_sha256, str) or _sha256_json(taxonomy) != taxonomy_sha256:
            raise ContextTopicAnalysisResumeError("resumed taxonomy is invalid")
        resumed_batches += 1
    completed_steps += 1
    _progress(progress_callback, completed_steps, total_steps, "taxonomy_synthesis", "taxonomy ready")

    assert isinstance(taxonomy, Mapping)
    assert isinstance(taxonomy_sha256, str)
    representative_classifications, resumed, called = _classify_unique(
        inventory,
        classification_batches,
        taxonomy,
        taxonomy_sha256,
        state,
        output,
        checkpoints,
        active_client,
        progress_callback,
        cancel_check,
        completed_steps,
        total_steps,
    )
    resumed_batches += resumed
    network_batches += called

    feedback = _build_refinement_feedback(
        list(representative_classifications.values()), taxonomy, ordinal_by_region
    )
    if refine_taxonomy and _feedback_has_work(feedback) and not state["resume"].get("taxonomy_refinement_complete"):
        batch_id = f"taxonomy-refinement-{_sha256_json(feedback)[:12]}"
        payload = _common_payload(
            inventory, operation="taxonomy_refinement", batch_id=batch_id
        )
        payload.update(
            {
                "current_taxonomy": taxonomy,
                "current_taxonomy_sha256": taxonomy_sha256,
                "feedback": feedback,
                "topic_limits": {
                    "min_families": MIN_FAMILIES,
                    "max_families": MAX_FAMILIES,
                    "min_topics": MIN_TOPICS,
                    "max_topics": MAX_TOPICS,
                },
            }
        )
        wrapper = active_client.request(payload, cancel_check=cancel_check)
        taxonomy, taxonomy_sha256, migration = _validate_taxonomy(wrapper)
        state["taxonomy"] = taxonomy
        state["taxonomy_sha256"] = taxonomy_sha256
        state["taxonomy_migration"] = migration
        state["resume"]["taxonomy_refinement_complete"] = True
        state["resume"]["classification"] = {}
        state["regions"] = []
        _atomic_write_json(output, state)
        _write_checkpoint(
            checkpoints,
            inventory=inventory,
            stage="taxonomy_refinement",
            batch_id=batch_id,
            request_payload=payload,
            wrapper=wrapper,
            item_ids=[item["topic_id"] for item in taxonomy["topics"]],
        )
        network_batches += 1
        representative_classifications, resumed, called = _classify_unique(
            inventory,
            classification_batches,
            taxonomy,
            taxonomy_sha256,
            state,
            output,
            checkpoints,
            active_client,
            progress_callback,
            cancel_check,
            completed_steps,
            total_steps,
        )
        resumed_batches += resumed
        network_batches += called

    final_regions = _fan_out_classifications(
        inventory,
        unique,
        representative_classifications,
        oversize,
    )
    if len(final_regions) != inventory.region_count or [item["region_id"] for item in final_regions] != [item.region_id for item in inventory.regions]:
        raise ContextTopicAnalysisProtocolError(
            "final analysis does not preserve exact canonical region coverage/order"
        )
    classified = sum(item["status"] == "classified" for item in final_regions)
    unclassified = len(final_regions) - classified
    review_count = sum(bool(item["review_required"]) for item in final_regions)
    state["taxonomy"] = taxonomy
    state["taxonomy_sha256"] = taxonomy_sha256
    state["regions"] = final_regions
    state["overlaps"] = _build_overlap_summaries(final_regions)
    state["boundary_reviews"] = _build_boundary_summaries(final_regions)
    state["coverage"] = {
        "expected_region_count": inventory.region_count,
        "output_region_count": len(final_regions),
        "classified_count": classified,
        "unclassified_needs_review_count": unclassified,
        "review_required_count": review_count,
        "exact_coverage": len(final_regions) == inventory.region_count,
        "one_primary_or_explicit_unclassified": all(
            (
                item["status"] == "taxonomy_gap"
                and item["primary_topic_id"] is None
            )
            or (
                item["status"] == "classified"
                and isinstance(item["primary_topic_id"], str)
                and bool(item["primary_topic_id"])
            )
            for item in final_regions
        ),
    }
    rebound = bind_context_compilation(inventory.docx_path, inventory.jsonl_path)
    if rebound.pair_fingerprint != inventory.pair_fingerprint:
        raise ContextTopicAnalysisInputError(
            "compilation inputs changed during topic analysis"
        )
    state["status"] = "complete"
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(output, state)
    return _outcome_from_state(
        output,
        checkpoints,
        state,
        resumed_batches=resumed_batches,
        network_batches=network_batches,
    )


def _classify_unique(
    inventory: BoundContextCompilation,
    batches: Sequence[_Batch],
    taxonomy: Mapping[str, Any],
    taxonomy_sha256: str,
    state: MutableMapping[str, Any],
    output: Path,
    checkpoints: Path,
    client: Any,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
    completed_offset: int,
    total_steps: int,
) -> tuple[dict[str, Mapping[str, Any]], int, int]:
    stored = state["resume"].get("classification", {})
    if not isinstance(stored, dict):
        stored = {}
    complete: dict[str, Mapping[str, Any]] = dict(stored)
    resumed = 0
    called = 0
    completed = completed_offset
    for batch in batches:
        raise_if_cancelled(cancel_check, phase="topic classification")
        if all(item.representative.region_id in complete for item in batch.regions):
            resumed += 1
            completed += 1
            _progress(progress_callback, completed, total_steps, "classification", f"resumed {batch.batch_id}")
            continue
        payload = _common_payload(
            inventory, operation="classification", batch_id=batch.batch_id
        )
        payload.update(
            {
                "taxonomy": taxonomy,
                "taxonomy_sha256": taxonomy_sha256,
                "regions": [item.payload for item in batch.regions],
            }
        )
        wrapper = client.request(payload, cancel_check=cancel_check)
        rows = _validate_classifications(wrapper, batch, taxonomy, taxonomy_sha256)
        for row in rows:
            complete[row["region_id"]] = row
        state["resume"]["classification"] = complete
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(output, state)
        _write_checkpoint(
            checkpoints,
            inventory=inventory,
            stage="classification",
            batch_id=batch.batch_id,
            request_payload=payload,
            wrapper=wrapper,
            item_ids=[item.representative.region_id for item in batch.regions],
        )
        called += 1
        completed += 1
        _progress(progress_callback, completed, total_steps, "classification", batch.batch_id)
    return complete, resumed, called


def _fan_out_classifications(
    inventory: BoundContextCompilation,
    unique: Sequence[_UniqueRegion],
    representative_rows: Mapping[str, Mapping[str, Any]],
    oversize: Mapping[str, _UniqueRegion],
) -> list[dict[str, Any]]:
    output_by_id: dict[str, dict[str, Any]] = {}
    for item in unique:
        representative = item.representative
        row = representative_rows.get(representative.region_id)
        for member in item.members:
            review_reasons: list[str] = []
            if member.selection_method == "deterministic_context_window":
                review_reasons.append("deterministic_boundary_fallback")
            if member.selection_confidence is not None and member.selection_confidence < 0.7:
                review_reasons.append("boundary_confidence_below_0.7")
            if representative.region_id in oversize:
                row = None
                review_reasons.append("region_payload_exceeds_40KB")
            if row is None:
                output_by_id[member.region_id] = {
                    "region_id": member.region_id,
                    "ordinal": member.ordinal,
                    "source_relative_path": member.source_relative_path,
                    "selected_text_sha256": member.selected_text_sha256,
                    "representative_region_id": representative.region_id,
                    "status": "taxonomy_gap",
                    "classification_key": UNCLASSIFIED_TOPIC_ID,
                    "primary_topic_id": None,
                    "secondary_topic_ids": [],
                    "awakening_roles": ["unclear"],
                    "evidence_paragraph_numbers": [],
                    "certainty": "low",
                    "ambiguity_codes": ["insufficient_context"],
                    "ambiguity": "taxonomy_gap",
                    "taxonomy_gap": True,
                    "suggested_topic_label": None,
                    "model_review_status": "human_review",
                    "review_status": "review_required",
                    "review_required": True,
                    "review_reasons": sorted(set(review_reasons + ["unclassified"])),
                    "model_proposal": None,
                }
                continue
            assignments = row["assignments"]
            primary = [entry["topic_id"] for entry in assignments if entry["role"] == "primary"]
            secondary = [entry["topic_id"] for entry in assignments if entry["role"] == "secondary"]
            explicit_unclassified = row["taxonomy_gap"] or len(primary) != 1
            mapped_evidence, mapped = _map_evidence_to_member(
                representative, member, row["evidence_paragraph_numbers"]
            )
            if not mapped:
                review_reasons.append("duplicate_evidence_mapping_failed")
            if row["certainty"] == "low":
                review_reasons.append("low_model_certainty")
            if row["ambiguity_codes"]:
                review_reasons.extend(f"ambiguity:{item}" for item in row["ambiguity_codes"])
            if row["taxonomy_gap"]:
                review_reasons.append("taxonomy_gap")
            if explicit_unclassified:
                review_reasons.append("unclassified")
            ambiguity = _normalise_ambiguity(
                row["ambiguity_codes"], taxonomy_gap=explicit_unclassified
            )
            required_review = (
                explicit_unclassified
                or member.selection_method == "deterministic_context_window"
                or (
                    member.selection_confidence is not None
                    and member.selection_confidence < 0.7
                )
                or not mapped
                or row["certainty"] == "low"
                or "boundary_uncertain" in row["ambiguity_codes"]
                or "insufficient_context" in row["ambiguity_codes"]
                or str(row["review_status"]).casefold()
                in {"human_review", "needs_review", "review_required"}
            )
            output_by_id[member.region_id] = {
                "region_id": member.region_id,
                "ordinal": member.ordinal,
                "source_relative_path": member.source_relative_path,
                "selected_text_sha256": member.selected_text_sha256,
                "representative_region_id": representative.region_id,
                "status": "taxonomy_gap" if explicit_unclassified else "classified",
                "classification_key": (
                    UNCLASSIFIED_TOPIC_ID if explicit_unclassified else primary[0]
                ),
                "primary_topic_id": None if explicit_unclassified else primary[0],
                "secondary_topic_ids": (
                    [] if explicit_unclassified else secondary[:MAX_SECONDARY_TOPICS]
                ),
                "awakening_roles": list(row["awakening_roles"]),
                "evidence_paragraph_numbers": mapped_evidence,
                "certainty": row["certainty"],
                "ambiguity_codes": list(row["ambiguity_codes"]),
                "ambiguity": ambiguity,
                "taxonomy_gap": bool(row["taxonomy_gap"]),
                "suggested_topic_label": row.get("suggested_topic_label"),
                "model_review_status": row["review_status"],
                "review_status": (
                    "review_required" if required_review else "review_recommended"
                ),
                "review_required": required_review,
                "review_reasons": sorted(set(review_reasons)),
                "model_proposal": {
                    "representative_region_input_sha256": row["region_input_sha256"],
                    "assignments": list(assignments),
                },
            }
    return [output_by_id[region.region_id] for region in inventory.regions]


def _normalise_ambiguity(
    codes: Sequence[str], *, taxonomy_gap: bool
) -> str:
    if taxonomy_gap or "taxonomy_gap" in codes:
        return "taxonomy_gap"
    if "insufficient_context" in codes or "boundary_uncertain" in codes:
        return "insufficient_context"
    if "taxonomy_overlap" in codes or "mixed_passage" in codes:
        return "topic_overlap"
    return "none"


def _build_overlap_summaries(
    classifications: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[str]] = {}
    for item in classifications:
        topic_ids = [
            value
            for value in [
                item.get("primary_topic_id"),
                *list(item.get("secondary_topic_ids") or []),
            ]
            if isinstance(value, str) and value
        ]
        if len(topic_ids) < 2:
            continue
        key = tuple(sorted(set(topic_ids)))
        groups.setdefault(key, []).append(str(item["region_id"]))
    return [
        {
            "topic_ids": list(topic_ids),
            "count": len(region_ids),
            "representative_region_ids": region_ids[:MAX_REPRESENTATIVE_REGIONS],
            "summary": (
                f"{len(region_ids)} region(s) carry the overlapping topic set: "
                + ", ".join(topic_ids)
                + "."
            ),
        }
        for topic_ids, region_ids in sorted(
            groups.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]


def _build_boundary_summaries(
    classifications: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    categories = (
        ("deterministic_boundary_fallback", "deterministic boundary fallback"),
        ("boundary_confidence_below_0.7", "boundary confidence below 0.7"),
        ("duplicate_evidence_mapping_failed", "duplicate evidence mapping failure"),
    )
    summaries: list[dict[str, Any]] = []
    for reason, label in categories:
        ids = [
            str(item["region_id"])
            for item in classifications
            if reason in item.get("review_reasons", [])
        ]
        if ids:
            summaries.append(
                {
                    "reason": reason,
                    "count": len(ids),
                    "representative_region_ids": ids[:MAX_REPRESENTATIVE_REGIONS],
                    "summary": f"{len(ids)} region(s) require review for {label}.",
                }
            )
    return summaries


def _outcome_from_state(
    output: Path,
    checkpoints: Path,
    state: Mapping[str, Any],
    *,
    resumed_batches: int,
    network_batches: int,
) -> TopicAnalysisOutcome:
    taxonomy = state["taxonomy"]
    coverage = state["coverage"]
    return TopicAnalysisOutcome(
        output_path=output,
        checkpoint_dir=checkpoints,
        region_count=int(state["corpus"]["region_count"]),
        unique_text_count=int(state["deduplication"]["unique_text_count"]),
        duplicate_region_count=int(state["deduplication"]["duplicate_region_count"]),
        taxonomy_sha256=str(state["taxonomy_sha256"]),
        family_count=len(taxonomy["families"]),
        topic_count=len(taxonomy["topics"]),
        classified_count=int(coverage["classified_count"]),
        unclassified_review_count=int(coverage["unclassified_needs_review_count"]),
        review_required_count=int(coverage["review_required_count"]),
        resumed_batches=resumed_batches,
        network_batches=network_batches,
    )


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "ContextTopicAnalysisAccessError",
    "ContextTopicAnalysisClient",
    "ContextTopicAnalysisConfigurationError",
    "ContextTopicAnalysisError",
    "ContextTopicAnalysisInputError",
    "ContextTopicAnalysisNetworkError",
    "ContextTopicAnalysisProtocolError",
    "ContextTopicAnalysisResumeError",
    "DEFAULT_ENDPOINT",
    "TOPIC_ANALYSIS_MODEL",
    "TOPIC_ANALYSIS_PROFILE",
    "TopicAnalysisOutcome",
    "UNCLASSIFIED_TOPIC_ID",
    "analyse_context_topics",
]
