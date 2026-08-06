from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from docx import Document

from context_finder import (
    SearchOptions,
    apply_boundary_selection,
    create_compilation_docx,
    find_contexts,
    write_result_records,
)
from context_topic_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    ContextTopicAnalysisProtocolError,
    TOPIC_ANALYSIS_MODEL,
    TOPIC_ANALYSIS_PROFILE,
    UNCLASSIFIED_TOPIC_ID,
    _make_batches,
    _deduplicate_regions,
    _summaries_for_synthesis,
    _sha256_json,
    _wire_json,
    analyse_context_topics,
)
from context_compilation_inventory import bind_context_compilation


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    document = Document()
    for text in paragraphs:
        document.add_paragraph(text)
    document.save(path)


def _make_pair(root: Path, *, oversize: bool = False) -> tuple[Path, Path]:
    library = root / "library"
    library.mkdir()
    _write_docx(library / "A.docx", ["Awakening must occur."])
    _write_docx(library / "B.docx", ["Awakening must occur."])
    _write_docx(library / "C.docx", ["Awakening grows through daily practice."])
    if oversize:
        _write_docx(
            library / "D.docx",
            ["Awakening " + ("extended context " * 3_500)],
        )
    result = find_contexts(
        library,
        "awakening",
        options=SearchOptions(context_words_each_side=1_000),
    )
    updated = []
    for index, region in enumerate(result.regions):
        if index == 1:
            updated.append(
                apply_boundary_selection(
                    region,
                    region.broad_start_paragraph,
                    region.broad_end_paragraph,
                    method="deterministic_context_window",
                    note="fixture fallback",
                )
            )
        else:
            updated.append(
                apply_boundary_selection(
                    region,
                    region.broad_start_paragraph,
                    region.broad_end_paragraph,
                    method="glm_boundary_refinement",
                    model=TOPIC_ANALYSIS_MODEL,
                    confidence=(0.6 if index == 2 else 1.0),
                )
            )
    result = result.with_regions(tuple(updated))
    docx = root / "Awakening.docx"
    jsonl = root / "Awakening.jsonl"
    write_result_records(result, jsonl)
    create_compilation_docx(result, docx)
    return docx, jsonl


def _taxonomy(version: int = 1) -> dict:
    families = [
        {
            "family_id": f"family_{index}",
            "label": f"Family {index}",
            "definition": f"Definition for family {index}.",
        }
        for index in range(1, 7)
    ]
    topics = [
        {
            "topic_id": f"topic_{version}_{index}",
            "label": f"Topic {version}.{index}",
            "family_id": f"family_{((index - 1) % 6) + 1}",
            "definition": f"Definition for topic {version}.{index}.",
            "aliases": [],
            "include_cues": ["awakening"],
            "exclude_cues": [],
        }
        for index in range(1, 13)
    ]
    return {
        "title": f"Awakening taxonomy {version}",
        "scope_note": "A proposed research taxonomy.",
        "families": families,
        "topics": topics,
    }


class FakeClient:
    endpoint = "https://example.test/topic-analysis"

    def __init__(self, *, bad_region_hash: bool = False, trigger_refinement: bool = False):
        self.bad_region_hash = bad_region_hash
        self.trigger_refinement = trigger_refinement
        self.operations: list[str] = []
        self.payloads: list[dict] = []
        self.secret = "SHOULD_NEVER_BE_WRITTEN"
        self.classification_pass = 0

    def _wrapper(self, payload: dict, result: dict, *, taxonomy=None) -> dict:
        body = _wire_json(payload)
        wrapper = {
            "topic_analysis_profile": TOPIC_ANALYSIS_PROFILE,
            "operation": payload["operation"],
            "model": TOPIC_ANALYSIS_MODEL,
            "batch_id": payload["batch_id"],
            "source_records_sha256": payload["source_records_sha256"],
            "result": result,
            "integrity": {
                "request_sha256": __import__("hashlib").sha256(body).hexdigest(),
                "request_bytes": len(body),
                "model_response_sha256": "a" * 64,
            },
        }
        if taxonomy is not None:
            wrapper["taxonomy_sha256"] = _sha256_json(taxonomy)
        return wrapper

    def request(self, payload, *, cancel_check=None):
        payload = dict(payload)
        self.operations.append(payload["operation"])
        self.payloads.append(payload)
        operation = payload["operation"]
        if operation == "candidate_cards":
            cards = []
            for index, region in enumerate(payload["regions"]):
                digest = _sha256_json(region)
                if self.bad_region_hash and index == 0:
                    digest = "0" * 64
                cards.append(
                    {
                        "region_id": region["region_id"],
                        "region_input_sha256": digest,
                        "candidates": [
                            {
                                "label": "Inner Work",
                                "description": "Candidate prose must not enter checkpoints.",
                                "evidence_paragraph_numbers": [
                                    region["hit_paragraph_numbers"][0]
                                ],
                            }
                        ],
                        "awakening_roles": ["practice_or_instruction"],
                        "mixed_section": False,
                    }
                )
            return self._wrapper(
                payload,
                {"batch_id": payload["batch_id"], "cards": cards},
            )
        if operation in {"taxonomy_synthesis", "taxonomy_refinement"}:
            taxonomy = _taxonomy(2 if operation == "taxonomy_refinement" else 1)
            return self._wrapper(
                payload,
                {
                    "batch_id": payload["batch_id"],
                    "taxonomy": taxonomy,
                    "migration": (
                        [{"from": "topic_1_1", "to": "topic_2_1"}]
                        if operation == "taxonomy_refinement"
                        else []
                    ),
                },
                taxonomy=taxonomy,
            )
        if operation == "classification":
            self.classification_pass += 1
            taxonomy = payload["taxonomy"]
            rows = []
            for index, region in enumerate(payload["regions"]):
                gap = (
                    self.trigger_refinement
                    and taxonomy["topics"][0]["topic_id"].startswith("topic_1_")
                    and index == 0
                )
                rows.append(
                    {
                        "region_id": region["region_id"],
                        "region_input_sha256": _sha256_json(region),
                        "assignments": (
                            []
                            if gap
                            else [
                                {
                                    "topic_id": taxonomy["topics"][0]["topic_id"],
                                    "role": "primary",
                                }
                            ]
                        ),
                        "awakening_roles": ["practice_or_instruction"],
                        "evidence_paragraph_numbers": [
                            region["hit_paragraph_numbers"][0]
                        ],
                        "certainty": "low" if gap else "high",
                        "ambiguity_codes": ["taxonomy_gap"] if gap else [],
                        "taxonomy_gap": gap,
                        "suggested_topic_label": "Direct awakening" if gap else None,
                        "review_status": "human_review" if gap else "accepted",
                    }
                )
            return self._wrapper(
                payload,
                {"batch_id": payload["batch_id"], "classifications": rows},
                taxonomy=taxonomy,
            )
        raise AssertionError(operation)


class NoNetworkClient:
    endpoint = "https://example.test/topic-analysis"

    def request(self, payload, *, cancel_check=None):
        raise AssertionError("completed analysis should resume without network")


class ContextTopicAnalysisTests(unittest.TestCase):
    def test_synthesis_summaries_cap_count_and_total_request_bytes(self):
        inventory = SimpleNamespace(query="awakening", jsonl_sha256="f" * 64)
        summaries = [
            {
                "candidate_key": f"candidate_{index:04d}",
                "label": f"Candidate {index}",
                "description": "A" * 240,
                "support_count": 1,
                "representative_region_ids": [f"region_{index:04d}"],
                "aliases": [],
            }
            for index in range(300)
        ]

        selected = _summaries_for_synthesis(inventory, summaries)
        probe = {
            "operation": "taxonomy_synthesis",
            "profile": TOPIC_ANALYSIS_PROFILE,
            "batch_id": "taxonomy-synthesis",
            "query": "awakening",
            "source_records_sha256": "f" * 64,
            "candidate_summaries": selected,
            "topic_limits": {
                "min_families": 6,
                "max_families": 10,
                "min_topics": 12,
                "max_topics": 18,
            },
        }

        self.assertLessEqual(len(selected), 200)
        self.assertLessEqual(len(_wire_json(probe)), 39_000)

    def test_hit_paragraph_numbers_are_unique_but_match_count_remains_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "library"
            library.mkdir()
            _write_docx(
                library / "Multiple.docx",
                ["Awakening is discussed, and awakening is repeated."],
            )
            result = find_contexts(library, "awakening")
            region = result.regions[0]
            region = apply_boundary_selection(
                region,
                region.broad_start_paragraph,
                region.broad_end_paragraph,
                method="glm_boundary_refinement",
                model=TOPIC_ANALYSIS_MODEL,
                confidence=1.0,
            )
            result = result.with_regions((region,))
            docx = root / "Awakening.docx"
            jsonl = root / "Awakening.jsonl"
            write_result_records(result, jsonl)
            create_compilation_docx(result, docx)

            unique = _deduplicate_regions(bind_context_compilation(docx, jsonl))

            self.assertEqual(2, unique[0].payload["occurrence_count"])
            self.assertEqual([1], unique[0].payload["hit_paragraph_numbers"])

    def test_complete_analysis_deduplicates_fans_out_and_flags_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docx, jsonl = _make_pair(root)
            output = root / "analysis.json"
            checkpoints = root / "checkpoints"
            progress = []
            client = FakeClient()
            outcome = analyse_context_topics(
                docx,
                jsonl,
                output_path=output,
                checkpoint_dir=checkpoints,
                client=client,
                refine_taxonomy=False,
                master_layout_pages=490,
                reading_words_per_page=450,
                progress_callback=lambda done, total, phase, detail: progress.append(
                    (done, total, phase, detail)
                ),
            )
            data = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(ANALYSIS_SCHEMA_VERSION, data["schema_version"])
            self.assertEqual("complete", data["status"])
            self.assertEqual(3, outcome.region_count)
            self.assertEqual(2, outcome.unique_text_count)
            self.assertEqual(1, outcome.duplicate_region_count)
            self.assertEqual(3, len(data["regions"]))
            self.assertEqual(490, data["corpus"]["master_layout_pages"])
            self.assertEqual(450, data["corpus"]["reading_words_per_page"])
            self.assertTrue(data["corpus"]["inventory_pair_fingerprint"])
            self.assertEqual(1, data["deduplication"]["duplicate_region_count"])
            self.assertTrue(data["coverage"]["exact_coverage"])
            self.assertTrue(data["coverage"]["one_primary_or_explicit_unclassified"])
            self.assertEqual("topic_1_1", data["regions"][0]["primary_topic_id"])
            self.assertEqual("topic_1_1", data["regions"][1]["primary_topic_id"])
            self.assertIn(
                "deterministic_boundary_fallback",
                data["regions"][1]["review_reasons"],
            )
            self.assertIn(
                "boundary_confidence_below_0.7",
                data["regions"][2]["review_reasons"],
            )
            candidate_payload = next(
                item for item in client.payloads if item["operation"] == "candidate_cards"
            )
            self.assertEqual(2, len(candidate_payload["regions"]))
            self.assertEqual(1, candidate_payload["regions"][0]["boundary_confidence"])
            self.assertTrue(progress)

    def test_rejects_mismatched_region_input_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docx, jsonl = _make_pair(root)
            with self.assertRaises(ContextTopicAnalysisProtocolError):
                analyse_context_topics(
                    docx,
                    jsonl,
                    output_path=root / "analysis.json",
                    checkpoint_dir=root / "checkpoints",
                    client=FakeClient(bad_region_hash=True),
                    refine_taxonomy=False,
                )

    def test_completed_analysis_resumes_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docx, jsonl = _make_pair(root)
            output = root / "analysis.json"
            analyse_context_topics(
                docx,
                jsonl,
                output_path=output,
                checkpoint_dir=root / "checkpoints",
                client=FakeClient(),
                refine_taxonomy=False,
            )
            resumed = analyse_context_topics(
                docx,
                jsonl,
                output_path=output,
                checkpoint_dir=root / "checkpoints",
                client=NoNetworkClient(),
                refine_taxonomy=False,
            )
            self.assertEqual(0, resumed.network_batches)

    def test_taxonomy_gap_triggers_refinement_and_reclassification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docx, jsonl = _make_pair(root)
            client = FakeClient(trigger_refinement=True)
            output = root / "analysis.json"
            analyse_context_topics(
                docx,
                jsonl,
                output_path=output,
                checkpoint_dir=root / "checkpoints",
                client=client,
                refine_taxonomy=True,
            )
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("taxonomy_refinement", client.operations)
            self.assertGreaterEqual(client.operations.count("classification"), 2)
            self.assertEqual("Awakening taxonomy 2", data["taxonomy"]["title"])
            self.assertTrue(
                all(
                    item["primary_topic_id"].startswith("topic_2_")
                    for item in data["regions"]
                )
            )

    def test_checkpoints_contain_no_source_credentials_or_model_prose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docx, jsonl = _make_pair(root)
            client = FakeClient()
            checkpoints = root / "checkpoints"
            analyse_context_topics(
                docx,
                jsonl,
                output_path=root / "analysis.json",
                checkpoint_dir=checkpoints,
                client=client,
                refine_taxonomy=False,
            )
            combined = "\n".join(
                path.read_text(encoding="utf-8")
                for path in checkpoints.rglob("*.json")
            )
            self.assertNotIn("Awakening must occur", combined)
            self.assertNotIn(client.secret, combined)
            self.assertNotIn("Candidate prose", combined)

    def test_oversize_unique_text_is_explicitly_unclassified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docx, jsonl = _make_pair(root, oversize=True)
            output = root / "analysis.json"
            analyse_context_topics(
                docx,
                jsonl,
                output_path=output,
                checkpoint_dir=root / "checkpoints",
                client=FakeClient(),
                refine_taxonomy=False,
            )
            data = json.loads(output.read_text(encoding="utf-8"))
            oversize = next(
                item
                for item in data["regions"]
                if "region_payload_exceeds_40KB" in item["review_reasons"]
            )
            self.assertEqual("taxonomy_gap", oversize["status"])
            self.assertIsNone(oversize["primary_topic_id"])
            self.assertEqual(UNCLASSIFIED_TOPIC_ID, oversize["classification_key"])
            self.assertEqual("review_required", oversize["review_status"])
            self.assertTrue(oversize["review_required"])

    def test_batches_are_deterministic_and_cap_twenty_regions(self):
        class Representative:
            def __init__(self, region_id):
                self.region_id = region_id

        class Unique:
            def __init__(self, index):
                self.representative = Representative(f"region_{index:03d}")
                self.region_input_sha256 = f"{index:064x}"[-64:]
                self.payload_bytes = 100

        unique = [Unique(index) for index in range(45)]
        first = _make_batches(unique, "candidate")
        second = _make_batches(unique, "candidate")
        self.assertEqual([20, 20, 5], [len(item.regions) for item in first])
        self.assertEqual(
            [item.batch_id for item in first],
            [item.batch_id for item in second],
        )


if __name__ == "__main__":
    unittest.main()
