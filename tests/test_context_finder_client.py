from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest

from cleanup_client import HttpResponse
from context_finder import (
    BoundarySelection,
    ContextRegion,
    OccurrenceRecord,
    ParagraphSnapshot,
    QuerySpec,
    SearchOptions,
    SearchResult,
)
from context_finder_client import (
    BOUNDARY_MODEL,
    BOUNDARY_PROFILE,
    ContextFinderClient,
    refine_result_with_glm,
)
from pipeline_control import PipelineCancelledError


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utf16_offset(text: str, index: int) -> int:
    return len(text[:index].encode("utf-16-le")) // 2


def make_region(
    region_id: str,
    paragraphs: list[str],
    query: str,
    occurrence_paragraphs: list[int],
) -> ContextRegion:
    snapshots = tuple(
        ParagraphSnapshot(number=index, text=text)
        for index, text in enumerate(paragraphs, start=1)
    )
    occurrences = []
    phrase_pattern = re.compile(
        r"\s+".join(re.escape(word) for word in query.split()), re.IGNORECASE
    )
    for occurrence_index, paragraph_number in enumerate(occurrence_paragraphs, start=1):
        text = snapshots[paragraph_number - 1].text
        match = phrase_pattern.search(text)
        if match is None:
            raise ValueError(f"query is absent from paragraph {paragraph_number}")
        start, end = match.span()
        occurrences.append(
            OccurrenceRecord(
                occurrence_id=f"occ-{region_id}-{occurrence_index}",
                paragraph_number=paragraph_number,
                start=start,
                end=end,
                matched_text=text[start:end],
            )
        )
    return ContextRegion(
        region_id=region_id,
        query=" ".join(query.split()),
        source_relative_path=f"{region_id}.docx",
        source_absolute_path=str(Path("C:/library") / f"{region_id}.docx"),
        source_sha256=digest("\n".join(paragraphs)),
        source_suffix=".docx",
        broad_start_paragraph=1,
        broad_end_paragraph=len(paragraphs),
        paragraphs=snapshots,
        occurrences=tuple(occurrences),
        selection=BoundarySelection(1, len(paragraphs)),
    )


def make_result(region: ContextRegion) -> SearchResult:
    words = region.query.split()
    return SearchResult(
        schema_version="context-finder-v1",
        root="C:/library",
        query=QuerySpec(region.query, region.query.casefold(), len(words)),
        options=SearchOptions(),
        scanned_files=1,
        ignored_generated_files=0,
        regions=(region,),
    )


def response_for(payload: dict, *, tamper_hash: bool = False) -> HttpResponse:
    paragraphs = payload["paragraphs"]
    hit = payload["hit"]
    paragraph = next(item for item in paragraphs if item["number"] == hit["paragraphNumber"])
    text = paragraph["text"]
    # Select the complete hit paragraph; this is exact source, not generated prose.
    selection = {
        "startParagraphNumber": paragraph["number"],
        "startOffset": 0,
        "endParagraphNumber": paragraph["number"],
        "endOffset": utf16_offset(text, len(text)),
    }
    source_json = compact(paragraphs)
    match_start = 0
    units = 0
    for index, character in enumerate(text):
        if units == hit["startOffset"]:
            match_start = index
            break
        units += 2 if ord(character) > 0xFFFF else 1
    else:
        match_start = len(text)
    match_end = match_start
    units = hit["startOffset"]
    for index in range(match_start, len(text)):
        if units == hit["endOffset"]:
            match_end = index
            break
        units += 2 if ord(text[index]) > 0xFFFF else 1
    else:
        match_end = len(text)
    selected_hash = digest(text)
    if tamper_hash:
        selected_hash = "0" * 64
    body = {
        "boundary_profile": BOUNDARY_PROFILE,
        "model": BOUNDARY_MODEL,
        "offset_unit": "utf16_code_unit",
        "hit": hit,
        "selection": selection,
        "decision": {
            "method": "glm",
            "scope": "whole_paragraphs",
            "confidence": 0.98,
            "reason_codes": ["adjacent_context_required"],
        },
        "integrity": {
            "source_sha256": digest(source_json),
            "source_bytes": len(source_json.encode("utf-8")),
            "match_sha256": digest(text[match_start:match_end]),
            "selected_sha256": selected_hash,
            "selected_bytes": len(text.encode("utf-8")),
            "model_response_sha256": "a" * 64,
        },
    }
    return HttpResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=compact(body).encode("utf-8"),
    )


class ContextFinderClientTests(unittest.TestCase):
    def test_verified_response_applies_only_bounds_and_resumes_compact_checkpoint(self):
        region = make_region(
            "single",
            ["Before context.", "Awakening must occur.", "A new topic."],
            "awakening",
            [2],
        )
        calls: list[tuple[dict, dict]] = []

        def transport(method, url, headers, body, timeout):
            payload = json.loads(body)
            calls.append((dict(headers), payload))
            self.assertEqual(hashlib.sha256(body).hexdigest(), headers["Idempotency-Key"])
            self.assertEqual("secret-id", headers["CF-Access-Client-Id"])
            self.assertEqual("secret-value", headers["CF-Access-Client-Secret"])
            return response_for(payload)

        client = ContextFinderClient(
            client_id="secret-id", client_secret="secret-value", transport=transport
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            refined = client.refine_result(
                make_result(region), checkpoint_dir=checkpoint_dir
            )
            self.assertEqual((2, 2), (
                refined.regions[0].selection.start_paragraph,
                refined.regions[0].selection.end_paragraph,
            ))
            self.assertEqual("glm_boundary_refinement", refined.regions[0].selection.method)
            checkpoints = list(checkpoint_dir.rglob("*.json"))
            self.assertEqual(1, len(checkpoints))
            saved = checkpoints[0].read_text(encoding="utf-8")
            self.assertNotIn("Awakening must occur", saved)
            self.assertNotIn("secret-id", saved)
            self.assertNotIn("secret-value", saved)
            self.assertNotIn('"responses"', saved)

            resumed = client.refine_result(
                make_result(region), checkpoint_dir=checkpoint_dir
            )
            self.assertEqual(1, len(calls))
            self.assertEqual("glm_boundary_refinement", resumed.regions[0].selection.method)

    def test_multiple_hits_use_literal_whitespace_and_utf16_then_union_paragraphs(self):
        first = "😀 We waited at the bus   stop before dawn."
        second = "Connecting thought."
        third = "Later the BUS STOP appeared again."
        region = make_region("multi", [first, second, third], "bus   stop", [1, 3])
        payloads: list[dict] = []

        def transport(_method, _url, _headers, body, _timeout):
            payload = json.loads(body)
            payloads.append(payload)
            return response_for(payload)

        client = ContextFinderClient(
            client_id="id", client_secret="secret", transport=transport
        )
        refined = client.refine_result(make_result(region), checkpoint_dir=None)
        self.assertEqual(2, len(payloads))
        self.assertEqual("bus   stop", payloads[0]["query"])
        self.assertEqual("BUS STOP", payloads[1]["query"])
        first_occurrence = region.occurrences[0]
        self.assertEqual(
            first_occurrence.start + 1,
            payloads[0]["hit"]["startOffset"],
            "the leading non-BMP character must consume two UTF-16 units",
        )
        self.assertEqual((1, 3), (
            refined.regions[0].selection.start_paragraph,
            refined.regions[0].selection.end_paragraph,
        ))

    def test_one_bad_hit_falls_back_the_entire_region_without_checkpoint(self):
        region = make_region(
            "bad",
            ["Awakening begins.", "Bridge.", "Another awakening occurs."],
            "awakening",
            [1, 3],
        )
        call_count = 0

        def transport(_method, _url, _headers, body, _timeout):
            nonlocal call_count
            call_count += 1
            return response_for(json.loads(body), tamper_hash=call_count == 2)

        client = ContextFinderClient(
            client_id="id", client_secret="secret", transport=transport
        )
        with tempfile.TemporaryDirectory() as directory:
            refined = client.refine_result(
                make_result(region), checkpoint_dir=Path(directory)
            )
            selection = refined.regions[0].selection
            self.assertEqual("deterministic_context_window", selection.method)
            self.assertEqual((1, 3), (selection.start_paragraph, selection.end_paragraph))
            self.assertEqual([], list(Path(directory).rglob("*.json")))

    def test_concurrency_is_bounded(self):
        regions = tuple(
            make_region(
                f"region-{index}",
                ["Before.", "Awakening occurs.", "After."],
                "awakening",
                [2],
            )
            for index in range(5)
        )
        base = make_result(regions[0])
        result = SearchResult(
            schema_version=base.schema_version,
            root=base.root,
            query=base.query,
            options=base.options,
            scanned_files=5,
            ignored_generated_files=0,
            regions=regions,
        )
        lock = threading.Lock()
        active = 0
        maximum = 0

        def transport(_method, _url, _headers, body, _timeout):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            try:
                return response_for(json.loads(body))
            finally:
                with lock:
                    active -= 1

        client = ContextFinderClient(
            client_id="id", client_secret="secret", transport=transport
        )
        refined = client.refine_result(result, max_workers=2, checkpoint_dir=None)
        self.assertEqual(5, len(refined.regions))
        self.assertGreaterEqual(maximum, 2)
        self.assertLessEqual(maximum, 2)

    def test_cancellation_propagates_instead_of_becoming_a_fallback(self):
        region = make_region(
            "cancel",
            ["Before.", "Awakening occurs.", "After."],
            "awakening",
            [2],
        )
        client = ContextFinderClient(
            client_id="id",
            client_secret="secret",
            transport=lambda *_args: self.fail("transport should not be called"),
        )
        with self.assertRaises(PipelineCancelledError):
            client.refine_result(make_result(region), cancel_check=lambda: True)

    def test_wrapper_removes_compact_checkpoints_unless_retention_is_requested(self):
        region = make_region(
            "retention",
            ["Before.", "Awakening occurs.", "After."],
            "awakening",
            [2],
        )

        def transport(_method, _url, _headers, body, _timeout):
            return response_for(json.loads(body))

        client = ContextFinderClient(
            client_id="id", client_secret="secret", transport=transport
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            refine_result_with_glm(
                make_result(region),
                client=client,
                checkpoint_dir=checkpoint_dir,
            )
            self.assertEqual([], list(checkpoint_dir.rglob("*.json")))

            refine_result_with_glm(
                make_result(region),
                client=client,
                checkpoint_dir=checkpoint_dir,
                retain_checkpoints=True,
            )
            checkpoints = list(checkpoint_dir.rglob("*.json"))
            self.assertEqual(1, len(checkpoints))
            self.assertNotIn("Awakening occurs", checkpoints[0].read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
