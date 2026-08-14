from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import archive_doctor as doctor


class ArchiveDoctorCliTests(unittest.TestCase):
    def test_every_subcommand_defaults_to_dry_run(self):
        parser = doctor.build_parser()
        for command, positional in (
            ("requeue", ["out"]),
            ("quarantine", ["source"]),
            ("replace-docx", ["generated", "legacy", "backups"]),
            ("prepare-cleanup", ["archive"]),
        ):
            args = parser.parse_args([command, *positional])
            self.assertFalse(args.confirm, command)

    def test_confirm_requires_exact_expected_count(self):
        with self.assertRaises(SystemExit):
            doctor.main(["quarantine", "source", "--confirm"])

    def test_quarantine_dry_run_never_calls_mutation(self):
        with mock.patch.object(doctor.older_transcripts, "plan_moves", return_value=()) as planned, mock.patch.object(
            doctor.older_transcripts, "apply_moves"
        ) as applied:
            self.assertEqual(doctor.main(["quarantine", "source"]), 0)
        planned.assert_called_once_with(Path("source"))
        applied.assert_not_called()
    def test_quarantine_confirm_passes_reviewed_count_and_option(self):
        move = SimpleNamespace(source=Path("old.docx"), destination=Path("safe/old.docx"))
        with mock.patch.object(doctor.older_transcripts, "plan_moves", return_value=(move,)), mock.patch.object(
            doctor.older_transcripts, "apply_moves", return_value=(move.destination,)
        ) as applied:
            self.assertEqual(
                doctor.main([
                    "quarantine", "source", "--confirm", "--expect", "1", "--replace-identical-destination"
                ]),
                0,
            )
        self.assertEqual(applied.call_args.kwargs["expected_count"], 1)
        self.assertTrue(applied.call_args.kwargs["replace_identical_destination"])

    def test_requeue_dry_run_never_calls_reset(self):
        with mock.patch.object(doctor.corrupted_transcripts, "find_candidates", return_value=()) as found, mock.patch.object(
            doctor.corrupted_transcripts, "apply_reset"
        ) as applied:
            self.assertEqual(doctor.main(["requeue", "out", "--min-gap-seconds", "45"]), 0)
        found.assert_called_once_with(Path("out"), min_gap_seconds=45.0)
        applied.assert_not_called()

    def test_replace_docx_confirm_delegates_to_transactional_implementation(self):
        plan = SimpleNamespace(items=(object(), object()), to_dict=lambda: {"items": 2})
        with mock.patch.object(doctor.legacy_replacement, "plan_legacy_docx_replacements", return_value=plan), mock.patch.object(
            doctor.legacy_replacement, "apply_legacy_docx_replacements", return_value=(Path("a.docx"), Path("b.docx"))
        ) as applied:
            self.assertEqual(doctor.main(["replace-docx", "generated", "legacy", "backups", "--confirm", "--expect", "2"]), 0)
        self.assertTrue(applied.call_args.kwargs["confirm"])
        self.assertEqual(applied.call_args.kwargs["expected_count"], 2)
        self.assertEqual(applied.call_args.kwargs["expected_scope_root"], Path("legacy"))

    def test_prepare_cleanup_dry_run_never_quarantines(self):
        plan = SimpleNamespace(manifest_count=0, kept_final_glm_docx=(), delete_candidates=())
        with mock.patch.object(doctor.docx_cleanup, "build_cleanup_plan", return_value=plan), mock.patch.object(
            doctor.docx_cleanup, "quarantine_candidates"
        ) as applied:
            self.assertEqual(doctor.main(["prepare-cleanup", "archive", "--polished-output", "polished"]), 0)
        applied.assert_not_called()
