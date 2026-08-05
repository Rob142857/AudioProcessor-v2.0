from __future__ import annotations

from datetime import date
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from publication_metadata import (
    clean_publication_title,
    infer_publication_metadata,
    read_embedded_audio_metadata,
)


def _synchsafe(value: int) -> bytes:
    return bytes(
        ((value >> 21) & 0x7F, (value >> 14) & 0x7F, (value >> 7) & 0x7F, value & 0x7F)
    )


def write_id3(path: Path, **values: str) -> None:
    names = {
        "title": "TIT2",
        "album": "TALB",
        "artist": "TPE1",
        "date": "TDRC",
        "genre": "TCON",
    }
    frames = bytearray()
    for key, value in values.items():
        payload = b"\x03" + value.encode("utf-8")
        frames.extend(names[key].encode("ascii"))
        frames.extend(len(payload).to_bytes(4, "big"))
        frames.extend(b"\x00\x00")
        frames.extend(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ID3\x03\x00\x00" + _synchsafe(len(frames)) + frames + b"audio")


class PublicationInferenceTests(unittest.TestCase):
    def test_embedded_album_and_year_drive_mw_filename_date(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "unhelpful folder" / "0122.mp3"
            write_id3(
                source,
                artist="Dr Philip Groves",
                album="1985 MW",
                date="1985",
                genre="Spiritual Exercises",
            )

            embedded = read_embedded_audio_metadata(source)
            publication = infer_publication_metadata(
                source, Path("different path") / "0122.mp3"
            )

        self.assertEqual(embedded["album"], "1985 MW")
        self.assertEqual(publication.source_type, "MW")
        self.assertEqual(publication.lecture_date, date(1985, 1, 22))
        self.assertEqual(publication.artist, "Dr Philip Groves")
        self.assertEqual(publication.title, "Spiritual Exercise - 22 January 1985")
        self.assertEqual(
            publication.postscript,
            "Processed by speech-to-text from a digitised tape recording "
            "originally recorded in person by MW on 22 January 1985.",
        )

    def test_full_embedded_tags_override_conflicting_path_and_filename(self):
        publication = infer_publication_metadata(
            Path("1991 RS") / "0203 Wrong Topic.mp3",
            Path("1991 RS") / "0203 Wrong Topic.mp3",
            embedded_metadata={
                "title": "The Enneagram",
                "album": "1985 MW",
                "date": "1985-01-29",
                "artist": "Dr Philip Groves",
            },
        )
        self.assertEqual(publication.title, "The Enneagram")
        self.assertEqual(publication.source_type, "MW")
        self.assertEqual(publication.lecture_date, date(1985, 1, 29))

    def test_single_file_relative_name_falls_back_to_absolute_parent_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = (
                Path(temporary)
                / "1985 MW"
                / "0129 The Enneagram.mp3"
            )
            publication = infer_publication_metadata(
                source,
                Path(source.name),
                embedded_metadata={},
            )

        self.assertEqual(publication.source_type, "MW")
        self.assertEqual(publication.lecture_date, date(1985, 1, 29))
        self.assertEqual(publication.title, "The Enneagram")

    def test_relative_path_and_filename_cover_gp_rs_and_rl(self):
        gp = infer_publication_metadata(
            Path("21-88 War Against Time 31 May 1988.mp3"),
            Path("1988 GP") / "21-88 War Against Time 31 May 1988.mp3",
            embedded_metadata={},
        )
        rs = infer_publication_metadata(
            Path("0813 Gurdjieff music.mp3"),
            Path("1991 RS") / "0813 Gurdjieff music.mp3",
            embedded_metadata={},
        )
        rl = infer_publication_metadata(
            Path("0102 Hidden Dimensions.mp3"),
            Path("1990 RL") / "0102 Hidden Dimensions.mp3",
            embedded_metadata={},
        )
        self.assertEqual((gp.source_type, gp.lecture_date, gp.title), ("GP", date(1988, 5, 31), "War Against Time"))
        self.assertEqual((rs.source_type, rs.lecture_date, rs.title), ("RS", date(1991, 8, 13), "Gurdjieff Music"))
        self.assertEqual((rl.source_type, rl.lecture_date, rl.title), ("RL", date(1990, 1, 2), "Hidden Dimensions"))

    def test_process_and_tape_labels_are_not_publishable_title_text(self):
        self.assertEqual(
            clean_publication_title(
                "L0193 The Inner Wisdom Dynamics of Arcane Christianity 20.1.1993_mixdown_Mono.flac"
            ),
            "The Inner Wisdom Dynamics of Arcane Christianity",
        )
        self.assertEqual(
            clean_publication_title("0122 The Enneagram - cleaned up.mp3"),
            "The Enneagram",
        )
        self.assertEqual(
            clean_publication_title(
                "0129 Esoteric Psychology (incomplete) - final transcript.mp3"
            ),
            "Esoteric Psychology (Incomplete)",
        )
        self.assertEqual(
            clean_publication_title("0129 Esoteric Psychology [poor quality].mp3"),
            "Esoteric Psychology (Poor Quality)",
        )
        self.assertEqual(
            clean_publication_title("0129 visualization exercise (incomplete).mp3"),
            "Visualization Exercise (Incomplete)",
        )
        self.assertEqual(
            clean_publication_title("0129 Tibetan Book of the Dead 2.mp3"),
            "Tibetan Book of the Dead 2",
        )


class _Colour:
    rgb = None


class _Font:
    def __init__(self):
        self.name = None
        self.size = None
        self.italic = False
        self.color = _Colour()


class _Run:
    def __init__(self, text=""):
        self.text = text
        self.font = _Font()
        self.bold = False


class _ParagraphFormat:
    alignment = None
    space_before = None
    space_after = None
    line_spacing = None
    keep_with_next = False


class _Paragraph:
    def __init__(self, text=""):
        self.text = text
        self.alignment = None
        self.paragraph_format = _ParagraphFormat()
        self.runs = [_Run(text)] if text else []

    def add_run(self, text):
        self.text += text
        run = _Run(text)
        self.runs.append(run)
        return run


class _Style:
    def __init__(self):
        self.font = _Font()
        self.paragraph_format = _ParagraphFormat()


class _Section:
    pass


class _CoreProperties:
    title = None
    author = None
    subject = None


class _FakeDocument:
    instances = []

    def __init__(self, path=None):
        self.paragraphs = []
        self.sections = [_Section()]
        self.styles = {name: _Style() for name in ("Normal", "Heading 1", "Heading 2", "Heading 3")}
        self.core_properties = _CoreProperties()
        if path is not None and Path(path).read_bytes() != b"valid-docx":
            raise ValueError("invalid DOCX")
        self.instances.append(self)

    def add_paragraph(self, text=""):
        paragraph = _Paragraph(text)
        self.paragraphs.append(paragraph)
        return paragraph

    def save(self, path):
        Path(path).write_bytes(b"valid-docx")


def load_renderer():
    _FakeDocument.instances.clear()
    docx = types.ModuleType("docx")
    docx.Document = _FakeDocument
    docx_enum = types.ModuleType("docx.enum")
    docx_enum_text = types.ModuleType("docx.enum.text")
    docx_enum_text.WD_ALIGN_PARAGRAPH = types.SimpleNamespace(CENTER="center", JUSTIFY="justify")
    docx_shared = types.ModuleType("docx.shared")
    docx_shared.RGBColor = lambda *values: values
    module_name = "_publication_txt_to_docx_test"
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).parents[1] / "txt_to_docx.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"docx": docx, "docx.enum": docx_enum, "docx.enum.text": docx_enum_text, "docx.shared": docx_shared},
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class PublicationRendererTests(unittest.TestCase):
    def test_trailing_cleanup_timestamp_without_colon_is_not_published(self):
        renderer = load_renderer()
        self.assertEqual(
            renderer.strip_workflow_metadata(
                "The lecture concludes here.\n\nCleaned up at 2026-08-05 10:30"
            ),
            "The lecture concludes here.",
        )

    def test_old_generated_provenance_is_replaced_not_duplicated(self):
        renderer = load_renderer()
        self.assertEqual(
            renderer.strip_workflow_metadata(
                "The lecture concludes here.\n\n"
                "Processed by speech-to-text from a digitised tape recording "
                "originally taken from MW on 22 January 1985.\n"
                "Cleaned up at: 2026-08-05 10:30"
            ),
            "The lecture concludes here.",
        )

    def test_renderer_uses_explicit_publication_styles_and_only_provenance_note(self):
        renderer = load_renderer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "1985 MW" / "0122 The Enneagram.mp3"
            write_id3(source, artist="Dr Philip Groves", album="1985 MW", date="1985")
            output = root / "published.docx"
            body = (
                "First body paragraph.\n\nSecond body paragraph.\n\n"
                "Transcription Information\nModel: hidden\nDevice: hidden\n"
                "Processing Time: hidden\nCleaned up at: hidden"
            )
            with mock.patch("builtins.print"):
                renderer.convert_txt_to_docx_from_text(
                    body,
                    source,
                    metadata={"model": "must not appear", "device": "must not appear"},
                    use_australian_spelling=False,
                    output_path=output,
                )

        document = _FakeDocument.instances[0]
        texts = [paragraph.text for paragraph in document.paragraphs]
        joined = "\n".join(texts)
        self.assertEqual(texts[0], "The Enneagram")
        self.assertEqual(texts[1], "Dr Philip Groves | 22 January 1985 | MW")
        self.assertNotIn("Transcript:", joined)
        self.assertNotIn("Transcription Information", joined)
        self.assertNotIn("must not appear", joined)
        self.assertNotIn("Cleaned up", joined)
        self.assertEqual(document.paragraphs[2].alignment, "justify")
        self.assertEqual(document.paragraphs[2].paragraph_format.space_after, 8 * 12_700)
        self.assertEqual(document.paragraphs[2].paragraph_format.line_spacing, 1.333)
        self.assertEqual(document.paragraphs[0].runs[0].font.size, 30 * 12_700)
        self.assertEqual(document.paragraphs[-1].runs[0].font.size, 9 * 12_700)
        self.assertTrue(document.paragraphs[-1].runs[0].font.italic)
        self.assertEqual(
            texts[-1],
            "Processed by speech-to-text from a digitised tape recording "
            "originally recorded in person by MW on 22 January 1985.",
        )
        self.assertEqual(document.sections[0].page_width, int(8.5 * 914_400))
        self.assertEqual(document.sections[0].left_margin, 914_400)
        self.assertEqual(document.styles["Normal"].font.name, "Calibri")
        self.assertEqual(document.styles["Normal"].font.size, 11 * 12_700)

    def test_renderer_never_invents_length_based_paragraphs(self):
        renderer = load_renderer()
        sentences = [
            f"Sentence {index} contains enough ordinary spoken words to test publication paragraph flow."
            for index in range(1, 31)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "1985 MW" / "0129 Long Lecture.mp3"
            write_id3(source, artist="Dr Philip Groves", album="1985 MW", date="1985")
            with mock.patch("builtins.print"):
                renderer.convert_txt_to_docx_from_text(
                    " ".join(sentences),
                    source,
                    use_australian_spelling=False,
                    output_path=Path(temporary) / "published.docx",
                )

        document = _FakeDocument.instances[0]
        body = document.paragraphs[2:-1]
        self.assertEqual(len(body), 1)
        self.assertTrue(all(paragraph.alignment == "justify" for paragraph in body))
        self.assertEqual(body[0].text, " ".join(sentences))


if __name__ == "__main__":
    unittest.main()
