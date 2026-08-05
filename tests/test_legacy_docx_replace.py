from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import legacy_docx_replace as replacement
from stt_coverage import assess_stt_coverage


def write_docx(path: Path, marker: str) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", "<Types/>")
        package.writestr("word/document.xml", f"<document>{marker}</document>")
    return path.read_bytes()


def stt_manifest_fields(job: Path) -> tuple[dict, dict]:
    segments = [
        {"start": 0.0, "end": 45.0, "text": "A complete timestamped lecture"},
        {"start": 45.0, "end": 98.0, "text": "continues to the tape ending"},
    ]
    segment_path = job / "raw.segments.json"
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(segments).encode("utf-8")
    segment_path.write_bytes(encoded)
    coverage = assess_stt_coverage(segments, 100.0)
    return (
        {
            "metadata": {"audio_duration_seconds": 100.0},
            "segments_sha256": hashlib.sha256(encoded).hexdigest(),
        },
        {
            "segments": str(segment_path),
            "coverage": coverage,
        },
    )


def add_job(
    generated_root: Path,
    legacy_root: Path,
    relative_source: str,
    marker: str,
    *,
    status: str = "verified",
) -> tuple[Path, Path, bytes]:
    source_relative = Path(relative_source)
    source = legacy_root / source_relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(f"audio-{marker}".encode("ascii"))
    job = generated_root / source_relative.parent / f"{source_relative.stem}__{source_relative.suffix[1:]}"
    generated = job / "final.docx"
    write_docx(generated, f"new-{marker}")
    target = legacy_root / source_relative.with_suffix(".docx")
    original = write_docx(target, f"old-{marker}")
    stt, stt_fixture = stt_manifest_fields(job)
    manifest = {
        "status": status,
        "qa": {
            "status": "passed" if status == "verified" else "needs_review",
            "stt_coverage": stt_fixture["coverage"],
        },
        "stt": stt,
        "cleanup": {
            "enabled": True,
            "needs_review": False,
            "model": "@cf/zai-org/glm-4.7-flash",
            "glossary_sha256": "a" * 64,
            "glossary_count": 1635,
            "grounding_glossary_terms_min": 1635,
            "grounding_glossary_terms_max": 1635,
        },
        "source": {"relative_path": source_relative.as_posix()},
        "artifacts": {
            "docx": str(generated),
            "segments": stt_fixture["segments"],
        },
    }
    job.mkdir(parents=True, exist_ok=True)
    (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return generated, target, original


class LegacyReplacementTests(unittest.TestCase):
    def roots(self, temporary: str):
        root = Path(temporary)
        generated = root / "generated"
        legacy = root / "dedicated-input"
        generated.mkdir()
        legacy.mkdir()
        return root, generated, legacy

    def test_plan_is_read_only_and_confined_to_verified_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            _new, target, original = add_job(generated, legacy, "1985 MW/0122 Topic.mp3", "one")

            plan = replacement.plan_legacy_docx_replacements(generated, legacy)

            self.assertEqual(len(plan.items), 1)
            self.assertEqual(plan.items[0].target, target.resolve())
            self.assertEqual(plan.items[0].operation, "replace")
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(len(plan.plan_sha256), 64)

    def test_non_verified_transcript_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            add_job(generated, legacy, "1985 MW/0122 Topic.mp3", "one", status="needs_review")
            with self.assertRaisesRegex(replacement.ReplacementError, "non-verified"):
                replacement.plan_legacy_docx_replacements(generated, legacy)

    def test_cleanup_must_prove_full_glm_glossary_grounding(self):
        invalid_updates = {
            "missing_cleanup": None,
            "disabled": {"enabled": False},
            "needs_review": {"needs_review": True},
            "missing_model": {"model": ""},
            "invalid_hash": {"glossary_sha256": "abc123"},
            "empty_glossary": {"glossary_count": 0},
            "missing_grounding": {
                "grounding_glossary_terms_min": None,
                "grounding_glossary_terms_max": None,
            },
            "missing_grounding_min": {"grounding_glossary_terms_min": None},
            "missing_grounding_max": {"grounding_glossary_terms_max": None},
            "noninteger_grounding": {"grounding_glossary_terms_min": 1635.0},
            "partial_grounding": {"grounding_glossary_terms_min": 1634},
            "partial_grounding_max": {"grounding_glossary_terms_max": 1634},
        }
        for label, updates in invalid_updates.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                _root, generated, legacy = self.roots(temporary)
                add_job(generated, legacy, "1985 MW/0122 Topic.mp3", label)
                manifest_path = next(generated.rglob("manifest.json"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if updates is None:
                    manifest.pop("cleanup")
                else:
                    manifest["cleanup"].update(updates)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                with self.assertRaisesRegex(replacement.ReplacementError, "cleanup"):
                    replacement.plan_legacy_docx_replacements(generated, legacy)

    def test_publication_rechecks_nonempty_hashed_stt_coverage(self):
        for label in ("empty_segments", "early_termination", "missing_duration"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                _root, generated, legacy = self.roots(temporary)
                add_job(generated, legacy, "1985 MW/0122 Topic.mp3", label)
                manifest_path = next(generated.rglob("manifest.json"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                segment_path = Path(manifest["artifacts"]["segments"])

                if label == "empty_segments":
                    segments = []
                    manifest["qa"]["stt_coverage"] = assess_stt_coverage(
                        segments, 100.0
                    )
                elif label == "early_termination":
                    segments = [
                        {"start": 0.0, "end": 20.0, "text": "Only the beginning"}
                    ]
                else:
                    segments = json.loads(segment_path.read_text(encoding="utf-8"))
                    manifest["stt"]["metadata"].pop("audio_duration_seconds")

                encoded = json.dumps(segments).encode("utf-8")
                segment_path.write_bytes(encoded)
                manifest["stt"]["segments_sha256"] = hashlib.sha256(encoded).hexdigest()
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                with self.assertRaisesRegex(replacement.ReplacementError, "STT"):
                    replacement.plan_legacy_docx_replacements(generated, legacy)

    def test_same_stem_multiple_formats_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            add_job(generated, legacy, "1987 MW/0302 Symbols.mp3", "mp3")
            # Both formats intentionally map to the same legacy Symbols.docx.
            second_source = Path("1987 MW/0302 Symbols.3gp")
            (legacy / second_source).write_bytes(b"alternate audio")
            job = generated / second_source.parent / "0302 Symbols__3gp"
            final = job / "final.docx"
            write_docx(final, "3gp")
            stt, stt_fixture = stt_manifest_fields(job)
            (job / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "verified",
                        "qa": {
                            "status": "passed",
                            "stt_coverage": stt_fixture["coverage"],
                        },
                        "stt": stt,
                        "cleanup": {
                            "enabled": True,
                            "needs_review": False,
                            "model": "@cf/zai-org/glm-4.7-flash",
                            "glossary_sha256": "b" * 64,
                            "glossary_count": 1635,
                            "grounding_glossary_terms_min": 1635,
                            "grounding_glossary_terms_max": 1635,
                        },
                        "source": {"relative_path": second_source.as_posix()},
                        "artifacts": {
                            "docx": str(final),
                            "segments": stt_fixture["segments"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(replacement.ReplacementError, "multiple source formats"):
                replacement.plan_legacy_docx_replacements(generated, legacy)

    def test_apply_requires_confirmation_exact_scope_and_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, generated, legacy = self.roots(temporary)
            add_job(generated, legacy, "1985 MW/0122 Topic.mp3", "one")
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            with self.assertRaisesRegex(replacement.ReplacementError, "confirm"):
                replacement.apply_legacy_docx_replacements(
                    plan, expected_scope_root=legacy, backup_root=root / "backup"
                )
            with self.assertRaisesRegex(replacement.ReplacementError, "expected_count"):
                replacement.apply_legacy_docx_replacements(
                    plan,
                    expected_scope_root=legacy,
                    backup_root=root / "backup",
                    confirm=True,
                    expected_count=2,
                )

    def test_success_replaces_atomically_and_retains_original_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, generated, legacy = self.roots(temporary)
            new, target, original = add_job(generated, legacy, "1985 MW/0122 Topic.mp3", "one")
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            backup_root = root / "backup"

            replaced = replacement.apply_legacy_docx_replacements(
                plan,
                expected_scope_root=legacy,
                backup_root=backup_root,
                confirm=True,
                expected_count=1,
            )

            self.assertEqual(replaced, (target.resolve(),))
            self.assertEqual(target.read_bytes(), new.read_bytes())
            self.assertEqual((backup_root / "1985 MW/0122 Topic.docx").read_bytes(), original)

    def test_success_creates_source_adjacent_docx_when_none_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, generated, legacy = self.roots(temporary)
            new, target, _original = add_job(
                generated, legacy, "1985 MW/0129 New Topic.mp3", "new"
            )
            target.unlink()
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            backup_root = root / "backup"

            self.assertEqual(plan.items[0].operation, "create")
            self.assertIsNone(plan.items[0].original_sha256)
            published = replacement.apply_legacy_docx_replacements(
                plan,
                expected_scope_root=legacy,
                backup_root=backup_root,
                confirm=True,
                expected_count=1,
            )

            self.assertEqual(published, (target.resolve(),))
            self.assertEqual(target.read_bytes(), new.read_bytes())
            self.assertFalse((backup_root / "1985 MW/0129 New Topic.docx").exists())

    def test_changed_generated_file_aborts_before_backup_or_target_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, generated, legacy = self.roots(temporary)
            new, target, original = add_job(generated, legacy, "1985 MW/0122 Topic.mp3", "one")
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            write_docx(new, "changed-after-plan")
            backup_root = root / "backup"

            with self.assertRaisesRegex(replacement.ReplacementError, "changed after planning"):
                replacement.apply_legacy_docx_replacements(
                    plan,
                    expected_scope_root=legacy,
                    backup_root=backup_root,
                    confirm=True,
                    expected_count=1,
                )

            self.assertEqual(target.read_bytes(), original)
            self.assertFalse(backup_root.exists())

    def test_mid_commit_failure_rolls_back_all_targets_and_keeps_backups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, generated, legacy = self.roots(temporary)
            _new1, target1, original1 = add_job(generated, legacy, "1985 MW/0122 One.mp3", "one")
            _new2, target2, original2 = add_job(generated, legacy, "1985 MW/0129 Two.mp3", "two")
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            backup_root = root / "backup"
            real_commit = replacement._commit_stage
            calls = 0

            def fail_second(staged, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise PermissionError("simulated Word lock")
                real_commit(staged, target)

            with mock.patch.object(replacement, "_commit_stage", side_effect=fail_second):
                with self.assertRaisesRegex(replacement.ReplacementError, "restored"):
                    replacement.apply_legacy_docx_replacements(
                        plan,
                        expected_scope_root=legacy,
                        backup_root=backup_root,
                        confirm=True,
                        expected_count=2,
                    )

            self.assertEqual(target1.read_bytes(), original1)
            self.assertEqual(target2.read_bytes(), original2)
            self.assertEqual((backup_root / "1985 MW/0122 One.docx").read_bytes(), original1)
            self.assertEqual((backup_root / "1985 MW/0129 Two.docx").read_bytes(), original2)

    def test_failure_after_create_restores_existing_and_removes_new_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, generated, legacy = self.roots(temporary)
            _new1, target1, original1 = add_job(
                generated, legacy, "1985 MW/0122 One.mp3", "one"
            )
            _new2, target2, _original2 = add_job(
                generated, legacy, "1985 MW/0129 Two.mp3", "two"
            )
            target2.unlink()
            _new3, target3, original3 = add_job(
                generated, legacy, "1985 MW/0205 Three.mp3", "three"
            )
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            backup_root = root / "backup"
            real_commit = replacement._commit_stage
            calls = 0

            def fail_third(staged, target):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise PermissionError("simulated Word lock")
                real_commit(staged, target)

            with mock.patch.object(replacement, "_commit_stage", side_effect=fail_third):
                with self.assertRaisesRegex(replacement.ReplacementError, "restored"):
                    replacement.apply_legacy_docx_replacements(
                        plan,
                        expected_scope_root=legacy,
                        backup_root=backup_root,
                        confirm=True,
                        expected_count=3,
                    )

            self.assertEqual(target1.read_bytes(), original1)
            self.assertFalse(target2.exists())
            self.assertEqual(target3.read_bytes(), original3)
            self.assertEqual((backup_root / "1985 MW/0122 One.docx").read_bytes(), original1)
            self.assertFalse((backup_root / "1985 MW/0129 Two.docx").exists())
            self.assertEqual((backup_root / "1985 MW/0205 Three.docx").read_bytes(), original3)


if __name__ == "__main__":
    unittest.main()
