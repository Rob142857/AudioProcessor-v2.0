from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
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
) -> tuple[Path, Path, Path, Path, bytes]:
    source_relative = Path(relative_source)
    source = legacy_root / source_relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(f"audio-{marker}".encode("ascii"))
    job = generated_root / source_relative.parent / f"{source_relative.stem}__{source_relative.suffix[1:]}"
    generated = job / "final.docx"
    write_docx(generated, f"new-{marker}")
    whisper_generated = job / "whisper.docx"
    write_docx(whisper_generated, f"raw-new-{marker}")
    whisper_target = legacy_root / source_relative.with_suffix(".docx")
    original = write_docx(whisper_target, f"raw-old-{marker}")
    review_target = legacy_root / source_relative.with_name(
        f"{source_relative.stem} - GLM Review.docx"
    )
    stt, stt_fixture = stt_manifest_fields(job)
    manifest = {
        "status": status,
        "approval_state": "pending_human_review",
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
        "render": {"output_sha256": hashlib.sha256(generated.read_bytes()).hexdigest()},
        "artifacts": {
            "docx": str(generated),
            "whisper_docx": str(whisper_generated),
            "segments": stt_fixture["segments"],
        },
    }
    manifest["stt"]["whisper_docx_sha256"] = hashlib.sha256(
        whisper_generated.read_bytes()
    ).hexdigest()
    job.mkdir(parents=True, exist_ok=True)
    (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return generated, whisper_generated, whisper_target, review_target, original


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
            _new, _raw_new, raw_target, review_target, original = add_job(
                generated, legacy, "1985 MW/0122 Topic.mp3", "one"
            )

            plan = replacement.plan_legacy_docx_replacements(generated, legacy)

            self.assertEqual(len(plan.items), 2)
            self.assertEqual(
                {item.target for item in plan.items},
                {raw_target.resolve(), review_target.resolve()},
            )
            self.assertEqual(
                {item.operation for item in plan.items}, {"create", "replace"}
            )
            self.assertEqual(raw_target.read_bytes(), original)
            self.assertFalse(review_target.exists())
            self.assertEqual(len(plan.plan_sha256), 64)

    def test_default_publishes_only_glm_review_target(self):
        # Owner's directive: publish_raw_transcript_sibling defaults to False
        # via PipelineConfig, and the pipeline always passes that value
        # through explicitly. plan_legacy_docx_replacements itself defaults
        # the keyword to True so unrelated existing callers keep the old
        # two-document behaviour unless they opt out; this test exercises
        # the opted-out (new normal) path directly.
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            _new, _raw_new, raw_target, review_target, original = add_job(
                generated, legacy, "1985 MW/0122 Topic.mp3", "one"
            )

            plan = replacement.plan_legacy_docx_replacements(
                generated, legacy, publish_raw_transcript_sibling=False
            )

            self.assertEqual(len(plan.items), 1)
            self.assertEqual(plan.items[0].target, review_target.resolve())
            self.assertEqual(plan.items[0].operation, "create")
            # The plain raw-transcript sibling is left completely alone: no
            # plan item, and the file on disk is untouched.
            self.assertEqual(raw_target.read_bytes(), original)

    def test_opt_in_restores_two_document_publication(self):
        # publish_raw_transcript_sibling=True restores the old behaviour of
        # publishing both the plain "<stem>.docx" Whisper sibling and the
        # "<stem> - GLM Review.docx" target.
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            _new, _raw_new, raw_target, review_target, _original = add_job(
                generated, legacy, "1985 MW/0122 Topic.mp3", "one"
            )

            plan = replacement.plan_legacy_docx_replacements(
                generated, legacy, publish_raw_transcript_sibling=True
            )

            self.assertEqual(len(plan.items), 2)
            self.assertEqual(
                {item.target for item in plan.items},
                {raw_target.resolve(), review_target.resolve()},
            )

    def test_needs_review_transcript_is_publishable_for_human_checking(self):
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            add_job(generated, legacy, "1985 MW/0122 Topic.mp3", "one", status="needs_review")
            manifest_path = next(generated.rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cleanup"]["needs_review"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            plan = replacement.plan_legacy_docx_replacements(generated, legacy)

            self.assertEqual(len(plan.items), 2)

    def test_cleanup_must_prove_full_glm_glossary_grounding(self):
        invalid_updates = {
            "missing_cleanup": None,
            "disabled": {"enabled": False},
            "invalid_review_state": {"needs_review": None},
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

    def test_music_tail_with_verified_clip_evidence_is_publishable(self):
        # A genuine music tail: the last text-bearing segment ends far short
        # of the audio duration (well beyond the trailing-silence tolerance),
        # but stt.metadata carries proof that every clip covering the gap was
        # verified to have returned a (silent/no-text) result rather than
        # being silently dropped. assess_stt_coverage downgrades the
        # over-tolerance gap to a "notes" entry instead of a blocking reason,
        # and publication must proceed.
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            add_job(generated, legacy, "1985 MW/0122 Topic.mp3", "music")
            manifest_path = next(generated.rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            segment_path = Path(manifest["artifacts"]["segments"])

            duration = 100.0
            segments = [
                {"start": 0.0, "end": 45.0, "text": "A complete timestamped lecture"},
                {"start": 45.0, "end": 50.0, "text": "continues until the music starts"},
            ]
            encoded = json.dumps(segments).encode("utf-8")
            segment_path.write_bytes(encoded)

            metadata = {
                "audio_duration_seconds": duration,
                "clip_seconds": 20.0,
                "clip_count": 5,
                "clip_results_verified": True,
                "empty_clip_count": 2,
            }
            manifest["stt"]["metadata"] = metadata
            manifest["stt"]["segments_sha256"] = hashlib.sha256(encoded).hexdigest()
            manifest["qa"]["stt_coverage"] = assess_stt_coverage(
                segments, duration, stt_metadata=metadata
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            plan = replacement.plan_legacy_docx_replacements(generated, legacy)

            self.assertEqual(len(plan.items), 2)

    def test_music_tail_without_verified_clip_evidence_is_rejected(self):
        # Same over-tolerance gap as above, but neither the metadata nor the
        # persisted coverage record carries verified-clip-grid evidence, so
        # the transcript looks indistinguishable from a truncated one and
        # must keep failing exactly as it did before the evidence contract
        # existed.
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            add_job(generated, legacy, "1985 MW/0122 Topic.mp3", "music")
            manifest_path = next(generated.rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            segment_path = Path(manifest["artifacts"]["segments"])

            duration = 100.0
            segments = [
                {"start": 0.0, "end": 45.0, "text": "A complete timestamped lecture"},
                {"start": 45.0, "end": 50.0, "text": "continues until the music starts"},
            ]
            encoded = json.dumps(segments).encode("utf-8")
            segment_path.write_bytes(encoded)

            metadata = {
                "audio_duration_seconds": duration,
                "clip_seconds": 20.0,
                "clip_count": 5,
                "empty_clip_count": 2,
            }
            manifest["stt"]["metadata"] = metadata
            manifest["stt"]["segments_sha256"] = hashlib.sha256(encoded).hexdigest()
            coverage = assess_stt_coverage(segments, duration, stt_metadata=metadata)
            coverage.pop("clip_results_verified", None)
            coverage.pop("clip_seconds", None)
            coverage.pop("clip_count", None)
            manifest["qa"]["stt_coverage"] = coverage
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                replacement.ReplacementError, "lacks passing STT coverage evidence"
            ):
                replacement.plan_legacy_docx_replacements(generated, legacy)

    def test_music_tail_evidence_tamper_mismatch_is_rejected(self):
        # The persisted "passed" coverage record claims verified-clip-grid
        # evidence, but the metadata actually on disk no longer backs that
        # claim (e.g. a stale record left over after metadata was edited or
        # regenerated). The publish-time recheck recomputes coverage from the
        # *current* metadata rather than trusting the stale recorded record,
        # so this must still be rejected.
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            add_job(generated, legacy, "1985 MW/0122 Topic.mp3", "music")
            manifest_path = next(generated.rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            segment_path = Path(manifest["artifacts"]["segments"])

            duration = 100.0
            segments = [
                {"start": 0.0, "end": 45.0, "text": "A complete timestamped lecture"},
                {"start": 45.0, "end": 50.0, "text": "continues until the music starts"},
            ]
            encoded = json.dumps(segments).encode("utf-8")
            segment_path.write_bytes(encoded)
            manifest["stt"]["segments_sha256"] = hashlib.sha256(encoded).hexdigest()

            valid_metadata = {
                "audio_duration_seconds": duration,
                "clip_seconds": 20.0,
                "clip_count": 5,
                "clip_results_verified": True,
                "empty_clip_count": 2,
            }
            # The recorded coverage record was produced while the metadata
            # still carried valid clip evidence...
            manifest["qa"]["stt_coverage"] = assess_stt_coverage(
                segments, duration, stt_metadata=valid_metadata
            )
            # ...but the metadata now on disk has lost that evidence.
            weakened_metadata = dict(valid_metadata)
            weakened_metadata.pop("clip_results_verified")
            manifest["stt"]["metadata"] = weakened_metadata
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
            whisper = job / "whisper.docx"
            write_docx(whisper, "raw-3gp")
            stt, stt_fixture = stt_manifest_fields(job)
            stt["whisper_docx_sha256"] = hashlib.sha256(
                whisper.read_bytes()
            ).hexdigest()
            (job / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "verified",
                        "approval_state": "pending_human_review",
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
                        "render": {
                            "output_sha256": hashlib.sha256(
                                final.read_bytes()
                            ).hexdigest()
                        },
                        "artifacts": {
                            "docx": str(final),
                            "whisper_docx": str(whisper),
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
                    plan,
                    expected_scope_root=legacy,
                    backup_root=generated / "backups" / "run-1",
                )
            with self.assertRaisesRegex(replacement.ReplacementError, "expected_count"):
                replacement.apply_legacy_docx_replacements(
                    plan,
                    expected_scope_root=legacy,
                    backup_root=generated / "backups" / "run-2",
                    confirm=True,
                    expected_count=99,
                )

    def test_success_replaces_atomically_and_retains_original_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            new_review, new_raw, raw_target, review_target, original = add_job(
                generated, legacy, "1985 MW/0122 Topic.mp3", "one"
            )
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            backup_root = generated / "publication-backups" / "run"

            replaced = replacement.apply_legacy_docx_replacements(
                plan,
                expected_scope_root=legacy,
                backup_root=backup_root,
                confirm=True,
                expected_count=2,
            )

            self.assertEqual(
                set(replaced), {raw_target.resolve(), review_target.resolve()}
            )
            self.assertEqual(review_target.read_bytes(), new_review.read_bytes())
            self.assertEqual(raw_target.read_bytes(), new_raw.read_bytes())
            self.assertEqual((backup_root / "1985 MW/0122 Topic.docx").read_bytes(), original)
            self.assertFalse(
                (backup_root / "1985 MW/0122 Topic - GLM Review.docx").exists()
            )

    @unittest.skipUnless(os.name == "nt", "extended-path handling is Windows-specific")
    def test_success_survives_a_backup_path_beyond_max_path(self):
        # Regression test for a live production failure: `_copy_to_new_path`
        # called `destination.parent.mkdir(...)` and `os.replace(...)` with
        # plain paths. The backup directory itself (a run-timestamped folder
        # under `publication-backups/`, nested under the source's own
        # subfolder structure) can be under 260 chars while the temp file
        # `mkstemp` creates inside it -- which echoes the destination's full
        # original filename -- is not, so mkdir silently succeeded and the
        # temp file write failed with a bare "[Errno 2] No such file or
        # directory". Build a relative source long enough that the backup
        # temp file path clears 260 chars regardless of the platform temp
        # dir's length, then assert the replacement still completes.
        temporary = tempfile.mkdtemp()
        try:
            _root, generated, legacy = self.roots(temporary)
            backup_root = generated / "publication-backups" / "20260809-000000-000000"
            base_relative = Path(
                "Tapes From Joe (MW, RL, et al)/public lectures/Lecture.mp3"
            )
            # Pad so the *fixture's own* job directory under `generated`
            # (shorter -- no publication-backups/<run-id> prefix) stays safely
            # under MAX_PATH while still being able to hold its longest inner
            # filename, but the backup copy's temp file -- nested deeper,
            # under backup_root, with mkstemp's own overhead added on top --
            # clears 260 regardless of this machine's temp dir length.
            innermost_fixture_filename = len("raw.segments.json") + 1
            job_dir_overhead = len(
                str(generated / base_relative.parent / f"{base_relative.stem}__mp3")
            )
            padding = "x" * max(10, 235 - innermost_fixture_filename - job_dir_overhead)
            relative_source = base_relative.with_name(
                f"{base_relative.stem}{padding}{base_relative.suffix}"
            )

            job_dir = (
                generated
                / relative_source.parent
                / f"{relative_source.stem}__mp3"
            )
            self.assertLess(
                len(str(job_dir)) + innermost_fixture_filename,
                260,
                "test setup itself must stay under MAX_PATH",
            )

            new_review, new_raw, raw_target, review_target, original = add_job(
                generated, legacy, relative_source.as_posix(), "long"
            )
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            backup_file = backup_root / relative_source.with_suffix(".docx")
            temp_file_overhead = 14  # ".", destination name repeated, ".", 8 random chars, ".tmp"
            self.assertGreater(
                len(str(backup_file)) + temp_file_overhead,
                260,
                "test setup must actually exercise a path beyond MAX_PATH",
            )

            replaced = replacement.apply_legacy_docx_replacements(
                plan,
                expected_scope_root=legacy,
                backup_root=backup_root,
                confirm=True,
                expected_count=2,
            )

            self.assertEqual(
                set(replaced), {raw_target.resolve(), review_target.resolve()}
            )
            self.assertEqual(raw_target.read_bytes(), new_raw.read_bytes())
            self.assertEqual(review_target.read_bytes(), new_review.read_bytes())
            self.assertEqual(
                Path(replacement._windows_extended_path(backup_file)).read_bytes(),
                original,
            )
        finally:
            # A plain shutil.rmtree (as used by tempfile.TemporaryDirectory's
            # own __exit__) hits the exact same MAX_PATH limit on cleanup that
            # this test exists to exercise, so it must be torn down through
            # the same extended-path helper under test.
            shutil.rmtree(replacement._windows_extended_path(temporary), ignore_errors=True)

    def test_success_creates_source_adjacent_docx_when_none_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            new_review, new_raw, raw_target, review_target, _original = add_job(
                generated, legacy, "1985 MW/0129 New Topic.mp3", "new"
            )
            raw_target.unlink()
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            backup_root = generated / "publication-backups" / "run"

            self.assertEqual({item.operation for item in plan.items}, {"create"})
            published = replacement.apply_legacy_docx_replacements(
                plan,
                expected_scope_root=legacy,
                backup_root=backup_root,
                confirm=True,
                expected_count=2,
            )

            self.assertEqual(set(published), {raw_target.resolve(), review_target.resolve()})
            self.assertEqual(raw_target.read_bytes(), new_raw.read_bytes())
            self.assertEqual(review_target.read_bytes(), new_review.read_bytes())
            self.assertFalse((backup_root / "1985 MW/0129 New Topic.docx").exists())

    def test_changed_generated_file_aborts_before_backup_or_target_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            new, _raw_new, raw_target, review_target, original = add_job(
                generated, legacy, "1985 MW/0122 Topic.mp3", "one"
            )
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            write_docx(new, "changed-after-plan")
            backup_root = generated / "publication-backups" / "run"

            with self.assertRaisesRegex(replacement.ReplacementError, "changed after planning"):
                replacement.apply_legacy_docx_replacements(
                    plan,
                    expected_scope_root=legacy,
                    backup_root=backup_root,
                    confirm=True,
                    expected_count=2,
                )

            self.assertEqual(raw_target.read_bytes(), original)
            self.assertFalse(review_target.exists())
            self.assertFalse(backup_root.exists())

    def test_mid_commit_failure_rolls_back_all_targets_and_keeps_backups(self):
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            _new1, _raw1, target1, review1, original1 = add_job(
                generated, legacy, "1985 MW/0122 One.mp3", "one"
            )
            _new2, _raw2, target2, review2, original2 = add_job(
                generated, legacy, "1985 MW/0129 Two.mp3", "two"
            )
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            backup_root = generated / "publication-backups" / "run"
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
                        expected_count=4,
                    )

            self.assertEqual(target1.read_bytes(), original1)
            self.assertEqual(target2.read_bytes(), original2)
            self.assertFalse(review1.exists())
            self.assertFalse(review2.exists())
            self.assertEqual((backup_root / "1985 MW/0122 One.docx").read_bytes(), original1)
            self.assertEqual((backup_root / "1985 MW/0129 Two.docx").read_bytes(), original2)

    def test_failure_after_create_restores_existing_and_removes_new_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            _root, generated, legacy = self.roots(temporary)
            _new1, _raw1, target1, review1, original1 = add_job(
                generated, legacy, "1985 MW/0122 One.mp3", "one"
            )
            _new2, _raw2, target2, review2, _original2 = add_job(
                generated, legacy, "1985 MW/0129 Two.mp3", "two"
            )
            target2.unlink()
            _new3, _raw3, target3, review3, original3 = add_job(
                generated, legacy, "1985 MW/0205 Three.mp3", "three"
            )
            plan = replacement.plan_legacy_docx_replacements(generated, legacy)
            backup_root = generated / "publication-backups" / "run"
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
                        expected_count=6,
                    )

            self.assertEqual(target1.read_bytes(), original1)
            self.assertFalse(target2.exists())
            self.assertEqual(target3.read_bytes(), original3)
            self.assertFalse(review1.exists())
            self.assertFalse(review2.exists())
            self.assertFalse(review3.exists())
            self.assertEqual((backup_root / "1985 MW/0122 One.docx").read_bytes(), original1)
            self.assertFalse((backup_root / "1985 MW/0129 Two.docx").exists())
            self.assertEqual((backup_root / "1985 MW/0205 Three.docx").read_bytes(), original3)


if __name__ == "__main__":
    unittest.main()
