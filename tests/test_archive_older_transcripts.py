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

    def test_replace_identical_destination_is_lossless(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"old transcript")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")
            moves = tool.plan_moves(root)
            dest_root = root.resolve().parent / "source - Older transcripts for review"
            dest_root.mkdir(parents=True)
            # Same bytes as the source docx -- e.g. the pipeline re-published
            # an unchanged job and byte-copied the same whisper.docx sibling.
            (dest_root / "0122 Topic.docx").write_bytes(b"old transcript")

            moved = tool.apply_moves(
                moves,
                confirm=True,
                expected_count=len(moves),
                replace_identical_destination=True,
            )

            self.assertFalse((root / "0122 Topic.docx").exists())
            self.assertEqual(moved, (dest_root / "0122 Topic.docx",))
            self.assertEqual((dest_root / "0122 Topic.docx").read_bytes(), b"old transcript")
            # No conflict file was created -- exactly one docx at the destination.
            self.assertEqual([p.name for p in dest_root.glob("*.docx")], ["0122 Topic.docx"])

    def test_replace_identical_destination_parks_differing_content_as_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"new source content")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")
            moves = tool.plan_moves(root)
            dest_root = root.resolve().parent / "source - Older transcripts for review"
            dest_root.mkdir(parents=True)
            (dest_root / "0122 Topic.docx").write_bytes(b"different archived content")

            moved = tool.apply_moves(
                moves,
                confirm=True,
                expected_count=len(moves),
                replace_identical_destination=True,
            )

            self.assertFalse((root / "0122 Topic.docx").exists())
            conflict_path = dest_root / "0122 Topic - conflict.docx"
            self.assertEqual(moved, (conflict_path,))
            # Both versions survive intact.
            self.assertEqual(
                (dest_root / "0122 Topic.docx").read_bytes(), b"different archived content"
            )
            self.assertEqual(conflict_path.read_bytes(), b"new source content")

    def test_default_mode_still_refuses_even_when_destination_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"old transcript")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")
            moves = tool.plan_moves(root)
            dest_root = root.resolve().parent / "source - Older transcripts for review"
            dest_root.mkdir(parents=True)
            (dest_root / "0122 Topic.docx").write_bytes(b"old transcript")

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                tool.apply_moves(moves, confirm=True, expected_count=len(moves))
            self.assertTrue((root / "0122 Topic.docx").exists())
            self.assertEqual((dest_root / "0122 Topic.docx").read_bytes(), b"old transcript")

    def test_other_recordings_glm_review_keeper_is_never_swept_by_prefix_collision(self):
        # "0122 Topic" and "0122 Topic - clean no music" are two distinct
        # recordings sharing a folder, where one's stem is a strict prefix of
        # the other's. Each has its own audio, its own plain transcript, and
        # its own GLM Review keeper. Processing "0122 Topic" must not sweep
        # up "0122 Topic - clean no music - GLM Review.docx" just because its
        # name starts with "0122 Topic - ".
        with tempfile.TemporaryDirectory() as temporary:
            root = self.roots(temporary)
            (root / "0122 Topic.mp3").write_bytes(b"audio")
            (root / "0122 Topic.docx").write_bytes(b"old transcript")
            (root / "0122 Topic - GLM Review.docx").write_bytes(b"new transcript")
            (root / "0122 Topic - clean no music.mp3").write_bytes(b"audio-clean")
            (root / "0122 Topic - clean no music.docx").write_bytes(b"old transcript-clean")
            (root / "0122 Topic - clean no music - GLM Review.docx").write_bytes(
                b"new transcript-clean"
            )

            moves = tool.plan_moves(root)

            self.assertEqual(
                {m.source.name for m in moves},
                {"0122 Topic.docx", "0122 Topic - clean no music.docx"},
            )

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
