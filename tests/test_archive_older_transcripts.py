from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import archive_older_transcripts as tool


class ArchiveOlderTranscriptsTests(unittest.TestCase):
    def roots(self, temporary: str) -> Path:
        root = Path(temporary) / "source"
        root.mkdir()
        return root

    def test_recording_without_glm_review_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"old transcript")

            moves = tool.plan_moves(root)

            self.assertEqual(moves, ())
            self.assertTrue((root / "0122 Topic.docx").exists())

    def test_old_transcript_moves_once_glm_review_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"old transcript")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")

            moves = tool.plan_moves(root)

            self.assertEqual(len(moves), 1)
            self.assertEqual(moves[0].source, (root / "0122 Topic.docx").resolve())
            self.assertEqual(
                moves[0].destination,
                (
                    root.resolve().parent
                    / "source - Older transcripts for review"
                    / "0122 Topic.docx"
                ),
            )

    def test_glm_review_file_itself_is_never_planned_to_move(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"old transcript")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")

            moves = tool.plan_moves(root)

            self.assertNotIn(root / "0122 Topic - GLM Review.docx", {m.source for m in moves})

    def test_unrelated_docx_in_the_same_folder_is_not_matched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")
            (root / "Unrelated Notes.docx").write_bytes(b"something else entirely")

            moves = tool.plan_moves(root)

            self.assertEqual(moves, ())

    def test_preserves_relative_directory_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            nested = root / "1985 MW" / "sub"
            nested.mkdir(parents=True)
            (nested / "0122 Topic.mp3").write_bytes(b"audio")
            (nested / "0122 Topic.docx").write_bytes(b"old transcript")
            (nested / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")

            moves = tool.plan_moves(root)

            self.assertEqual(
                moves[0].destination,
                root.resolve().parent
                / "source - Older transcripts for review"
                / "1985 MW"
                / "sub"
                / "0122 Topic.docx",
            )

    def test_same_docx_matched_via_two_audio_siblings_is_a_harmless_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio-mp3")
            (root / "0122 Topic.flac").write_bytes(b"audio-flac")
            (root / "0122 Topic.docx").write_bytes(b"old transcript")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")

            moves = tool.plan_moves(root)

            self.assertEqual(len(moves), 1)
            self.assertEqual(moves[0].source.name, "0122 Topic.docx")

    def test_apply_moves_requires_exact_expected_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"old transcript")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")
            moves = tool.plan_moves(root)

            with self.assertRaisesRegex(ValueError, "expected_count"):
                tool.apply_moves(moves, confirm=True, expected_count=len(moves) + 1)
            self.assertTrue((root / "0122 Topic.docx").exists())

    def test_apply_moves_refuses_dry_run_without_confirm(self):
        with self.assertRaisesRegex(ValueError, "dry-run only"):
            tool.apply_moves((), confirm=False, expected_count=0)

    def test_apply_moves_actually_moves_and_preserves_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"old transcript content")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")
            moves = tool.plan_moves(root)

            moved = tool.apply_moves(moves, confirm=True, expected_count=len(moves))

            self.assertFalse((root / "0122 Topic.docx").exists())
            self.assertEqual(len(moved), 1)
            self.assertEqual(moved[0].read_bytes(), b"old transcript content")
            # The GLM Review copy and the source audio are untouched.
            self.assertTrue((root / "0122 Topic - GLM Review.docx").exists())
            self.assertTrue((root / "0122 Topic.mp3").exists())

    def test_apply_moves_refuses_to_overwrite_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"old transcript")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")
            moves = tool.plan_moves(root)
            dest_root = root.resolve().parent / "source - Older transcripts for review"
            dest_root.mkdir(parents=True)
            (dest_root / "0122 Topic.docx").write_bytes(b"already something here")

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                tool.apply_moves(moves, confirm=True, expected_count=len(moves))
            self.assertTrue((root / "0122 Topic.docx").exists())

    def test_same_stem_variant_docx_other_than_glm_review_is_also_moved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"old transcript")
            (root / "0122 Topic - Draft.docx").write_bytes(b"a draft variant")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")

            moves = tool.plan_moves(root)

            self.assertEqual(
                {m.source.name for m in moves},
                {"0122 Topic.docx", "0122 Topic - Draft.docx"},
            )


if __name__ == "__main__":
    unittest.main()
