import json
from pathlib import Path
import tempfile
import unittest

from prepare_docx_cleanup import build_cleanup_plan, quarantine_candidates


class PrepareDocxCleanupTests(unittest.TestCase):
    def _manifest(self, source: Path, final_docx: Path) -> dict:
        return {
            "status": "needs_review",
            "source": {"path": str(source)},
            "stt": {
                "actual_model": "nvidia/parakeet-tdt-0.6b-v3",
                "metadata": {"backend": "nvidia-parakeet"},
            },
            "cleanup": {"output_sha256": "a" * 64},
            "render": {"output_path": str(final_docx), "output_sha256": "b" * 64},
        }

    def test_plan_keeps_only_proven_parakeet_glm_review_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            polished = root / "archive - Polished"
            archive.mkdir()
            polished.mkdir()
            audio = archive / "lecture.mp3"
            raw = archive / "lecture.docx"
            review = archive / "lecture - GLM Review.docx"
            old = archive / "nested" / "old.docx"
            for path in (audio, raw, review, old):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            final = polished / "lecture__mp3" / "final.docx"
            final.parent.mkdir(parents=True)
            final.write_bytes(b"fixture-final")
            manifest = self._manifest(audio, final)
            (final.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            plan = build_cleanup_plan(archive, polished)

            self.assertEqual(plan.kept_final_glm_docx, (review.resolve(),))
            self.assertEqual(plan.delete_candidates, (raw.resolve(), old.resolve()))

    def test_apply_requires_exact_reviewed_count_and_uses_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            polished = root / "archive - Polished"
            archive.mkdir()
            polished.mkdir()
            audio = archive / "lecture.mp3"
            raw = archive / "lecture.docx"
            review = archive / "lecture - GLM Review.docx"
            for path in (audio, raw, review):
                path.write_bytes(b"fixture")
            final = polished / "lecture__mp3" / "final.docx"
            final.parent.mkdir(parents=True)
            final.write_bytes(b"fixture-final")
            (final.parent / "manifest.json").write_text(
                json.dumps(self._manifest(audio, final)), encoding="utf-8"
            )
            plan = build_cleanup_plan(archive, polished)

            with self.assertRaisesRegex(ValueError, "reviewed count"):
                quarantine_candidates(plan, expected_count=2)
            moved = quarantine_candidates(plan, expected_count=1)

            self.assertFalse(raw.exists())
            self.assertTrue(review.exists())
            self.assertEqual(len(moved), 1)
            self.assertTrue(moved[0].is_file())
            self.assertTrue(moved[0].resolve().is_relative_to(plan.polished_root))
