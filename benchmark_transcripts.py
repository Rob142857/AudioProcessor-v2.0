"""Score model transcripts against a small hand-corrected archival gold set.

The benchmark deliberately evaluates raw ASR outputs before GLM cleanup.  That
keeps an editor from hiding deletions or hallucinations introduced by the speech
model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return WORD_RE.findall(normalized)


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    """Return minimum substitutions/deletions/insertions and matching words."""
    # Each cell is (total errors, substitutions, deletions, insertions, hits).
    previous = [(index, 0, index, 0, 0) for index in range(len(reference) + 1)]
    for hyp_index, hyp_word in enumerate(hypothesis, 1):
        current = [(hyp_index, 0, 0, hyp_index, 0)]
        for ref_index, ref_word in enumerate(reference, 1):
            if ref_word == hyp_word:
                diagonal = previous[ref_index - 1]
                candidates = [
                    (diagonal[0], diagonal[1], diagonal[2], diagonal[3], diagonal[4] + 1)
                ]
            else:
                diagonal = previous[ref_index - 1]
                candidates = [
                    (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3], diagonal[4])
                ]
            deletion = previous[ref_index]
            candidates.append(
                (deletion[0] + 1, deletion[1], deletion[2], deletion[3] + 1, deletion[4])
            )
            insertion = current[ref_index - 1]
            candidates.append(
                (insertion[0] + 1, insertion[1], insertion[2] + 1, insertion[3], insertion[4])
            )
            current.append(min(candidates, key=lambda item: item[:4]))
        previous = current
    errors, substitutions, deletions, insertions, hits = previous[-1]
    return {
        "errors": errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "hits": hits,
    }


def term_recall(reference_text: str, hypothesis_text: str, terms: list[str]) -> dict[str, Any]:
    reference_words = " ".join(words(reference_text))
    hypothesis_words = " ".join(words(hypothesis_text))
    relevant: list[str] = []
    found: list[str] = []
    for term in terms:
        normalized = " ".join(words(term))
        if normalized and normalized in reference_words:
            relevant.append(term)
            if normalized in hypothesis_words:
                found.append(term)
    return {
        "relevant": len(relevant),
        "found": len(found),
        "recall": (len(found) / len(relevant)) if relevant else None,
        "missed": [term for term in relevant if term not in found],
    }


def score(reference_text: str, hypothesis_text: str, terms: list[str]) -> dict[str, Any]:
    reference = words(reference_text)
    hypothesis = words(hypothesis_text)
    counts = edit_counts(reference, hypothesis)
    counts.update(
        {
            "reference_words": len(reference),
            "hypothesis_words": len(hypothesis),
            "wer": counts["errors"] / len(reference) if reference else 0.0,
            "term_recall": term_recall(reference_text, hypothesis_text, terms),
        }
    )
    return counts


def load_text(base: Path, value: str) -> str:
    path = (base / value).resolve()
    return path.read_text(encoding="utf-8")


def run_benchmark(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("benchmark manifest requires a non-empty samples array")
    base = manifest_path.parent
    global_terms = [str(item) for item in manifest.get("terms", [])]
    rows: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        sample_id = str(sample["id"])
        reference_text = load_text(base, str(sample["reference"]))
        terms = global_terms + [str(item) for item in sample.get("terms", [])]
        hypotheses = sample.get("hypotheses", {})
        if not isinstance(hypotheses, dict) or not hypotheses:
            raise ValueError(f"sample {sample_id!r} has no hypotheses")
        for model, hypothesis_path in hypotheses.items():
            result = score(reference_text, load_text(base, str(hypothesis_path)), terms)
            row = {"sample": sample_id, "model": str(model), **result}
            rows.append(row)
            aggregate = aggregates[str(model)]
            for key in (
                "errors",
                "substitutions",
                "deletions",
                "insertions",
                "hits",
                "reference_words",
                "hypothesis_words",
            ):
                aggregate[key] += int(result[key])
            aggregate["term_relevant"] += int(result["term_recall"]["relevant"])
            aggregate["term_found"] += int(result["term_recall"]["found"])

    summary: list[dict[str, Any]] = []
    for model, aggregate in sorted(aggregates.items()):
        reference_count = aggregate["reference_words"]
        term_count = aggregate["term_relevant"]
        summary.append(
            {
                "model": model,
                **dict(aggregate),
                "wer": aggregate["errors"] / reference_count if reference_count else 0.0,
                "term_recall": aggregate["term_found"] / term_count if term_count else None,
            }
        )
    return {"manifest": str(manifest_path), "rows": rows, "summary": summary}


def write_outputs(result: dict[str, Any], output_root: Path) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with output_root.with_suffix(".csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        fields = [
            "model",
            "wer",
            "term_recall",
            "reference_words",
            "hypothesis_words",
            "substitutions",
            "deletions",
            "insertions",
            "hits",
            "term_relevant",
            "term_found",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in result["summary"]:
            writer.writerow({key: row.get(key) for key in fields})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score raw ASR model transcripts")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args(argv)
    result = run_benchmark(args.manifest.resolve())
    write_outputs(result, args.output.resolve())
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
