"""Publication metadata inference for the Dr Philip Groves tape archive.

Inference is deliberately conservative and dependency-free.  Embedded MP3 ID3
tags take precedence, followed by the source-relative path and finally the
filename.  The module does not write tags or modify source recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Mapping, Optional


SOURCE_TYPES = frozenset({"MW", "GP", "RS", "RL"})

_MONTH_ALIASES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "febuary": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_PATTERN = "|".join(
    sorted((re.escape(value) for value in _MONTH_ALIASES), key=len, reverse=True)
)
_TEXT_DATE_RE = re.compile(
    rf"(?<!\d)(?P<day>[0-3]?\d)\s*(?P<month>{_MONTH_PATTERN})"
    r"\s*,?\s*(?P<year>(?:19|20)?\d{2})?(?!\d)",
    re.IGNORECASE,
)
_SOURCE_TOKEN_RE = re.compile(r"(?<![A-Za-z])(MW|GP|RS|RL)(?![A-Za-z])", re.I)

_TITLE_MINOR_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "but",
        "by",
        "for",
        "from",
        "in",
        "nor",
        "of",
        "on",
        "or",
        "per",
        "the",
        "to",
        "via",
    }
)

_ID3_FRAME_NAMES = {
    "TIT2": "title",
    "TALB": "album",
    "TPE1": "artist",
    "TDRC": "date",
    "TYER": "date",
    "TCON": "genre",
    "TT2": "title",
    "TAL": "album",
    "TP1": "artist",
    "TDA": "date",
    "TYE": "date",
    "TCO": "genre",
}


@dataclass(frozen=True)
class PublicationMetadata:
    """Human-facing metadata for one published transcript."""

    title: str
    lecture_date: Optional[date]
    source_type: Optional[str]
    artist: str
    album: Optional[str] = None
    genre: Optional[str] = None

    @property
    def source_label(self) -> str:
        return self.source_type or "an unidentified source"

    @property
    def date_label(self) -> str:
        return format_publication_date(self.lecture_date)

    @property
    def postscript(self) -> str:
        return (
            "Processed by speech-to-text from a digitised tape recording "
            f"originally recorded in person by {self.source_label} on {self.date_label}."
        )

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "title": self.title,
            "date": self.lecture_date.isoformat() if self.lecture_date else None,
            "source_type": self.source_type,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
            "postscript": self.postscript,
        }


def format_publication_date(value: Optional[date]) -> str:
    if value is None:
        return "an undetermined date"
    return f"{value.day} {value.strftime('%B')} {value.year}"


def _synchsafe(value: bytes) -> int:
    result = 0
    for byte in value:
        result = (result << 7) | (byte & 0x7F)
    return result


def _decode_id3_text(payload: bytes) -> str:
    if not payload:
        return ""
    encoding = payload[0]
    content = payload[1:]
    codecs = {0: "latin-1", 1: "utf-16", 2: "utf-16-be", 3: "utf-8"}
    try:
        decoded = content.decode(codecs.get(encoding, "latin-1"), errors="replace")
    except (LookupError, UnicodeError):
        decoded = content.decode("latin-1", errors="replace")
    values = [part.strip() for part in decoded.replace("\ufeff", "").split("\x00")]
    return "; ".join(part for part in values if part).strip()


def _read_id3v2(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("rb") as stream:
        header = stream.read(10)
        if len(header) != 10 or header[:3] != b"ID3":
            return values
        version = header[3]
        if version not in {2, 3, 4}:
            return values
        tag_size = _synchsafe(header[6:10])
        if tag_size <= 0 or tag_size > 32 * 1024 * 1024:
            return values
        data = stream.read(tag_size)

    if header[5] & 0x80:
        data = data.replace(b"\xff\x00", b"\xff")
    offset = 0
    if header[5] & 0x40 and len(data) >= 4:
        if version == 3:
            offset = min(len(data), 4 + int.from_bytes(data[:4], "big"))
        elif version == 4:
            offset = min(len(data), _synchsafe(data[:4]))

    while offset < len(data):
        if version == 2:
            if offset + 6 > len(data):
                break
            frame_id = data[offset : offset + 3].decode("ascii", errors="ignore")
            frame_size = int.from_bytes(data[offset + 3 : offset + 6], "big")
            header_size = 6
        else:
            if offset + 10 > len(data):
                break
            frame_id = data[offset : offset + 4].decode("ascii", errors="ignore")
            size_bytes = data[offset + 4 : offset + 8]
            frame_size = _synchsafe(size_bytes) if version == 4 else int.from_bytes(size_bytes, "big")
            header_size = 10
        if not frame_id.strip("\x00") or frame_size <= 0:
            break
        frame_start = offset + header_size
        frame_end = frame_start + frame_size
        if frame_end > len(data):
            break
        key = _ID3_FRAME_NAMES.get(frame_id)
        if key and key not in values:
            decoded = _decode_id3_text(data[frame_start:frame_end])
            if decoded:
                values[key] = decoded
        offset = frame_end
    return values


def _decode_id3v1_field(value: bytes) -> str:
    return value.rstrip(b"\x00 ").decode("latin-1", errors="replace").strip()


def _read_id3v1(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.stat().st_size < 128:
        return values
    with path.open("rb") as stream:
        stream.seek(-128, 2)
        tag = stream.read(128)
    if len(tag) != 128 or tag[:3] != b"TAG":
        return values
    for key, start, end in (
        ("title", 3, 33),
        ("artist", 33, 63),
        ("album", 63, 93),
        ("date", 93, 97),
    ):
        decoded = _decode_id3v1_field(tag[start:end])
        if decoded:
            values[key] = decoded
    return values


def read_embedded_audio_metadata(path: Path) -> dict[str, str]:
    """Read useful ID3 metadata without invoking a process or changing the file."""

    path = Path(path)
    if not path.is_file() or path.suffix.casefold() not in {".mp3", ".mp2", ".mpeg"}:
        return {}
    try:
        values = _read_id3v2(path)
        for key, value in _read_id3v1(path).items():
            values.setdefault(key, value)
        return values
    except OSError:
        return {}


def _year_from_text(value: str) -> Optional[int]:
    match = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", value)
    if match:
        return int(match.group(1))
    return None


def _normalise_year(value: int) -> int:
    if value < 100:
        return 1900 + value if value >= 50 else 2000 + value
    return value


def _full_date_from_tag(value: str) -> Optional[date]:
    value = value.strip()
    match = re.search(
        r"(?<!\d)(?P<year>19\d{2}|20\d{2})[-/.]?(?P<month>0?[1-9]|1[0-2])[-/.]?(?P<day>0?[1-9]|[12]\d|3[01])(?!\d)",
        value,
    )
    if not match:
        return None
    try:
        return date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None


def _date_from_filename(filename: str, year_hint: Optional[int]) -> Optional[date]:
    stem = Path(filename).stem
    textual = list(_TEXT_DATE_RE.finditer(stem))
    if textual:
        match = textual[-1]
        year_text = match.group("year")
        chosen_year = year_hint
        if chosen_year is None and year_text:
            chosen_year = _normalise_year(int(year_text))
        if chosen_year is not None:
            try:
                return date(
                    chosen_year,
                    _MONTH_ALIASES[match.group("month").casefold()],
                    int(match.group("day")),
                )
            except ValueError:
                pass

    numeric = re.search(
        r"(?<!\d)(?P<day>[0-3]?\d)[.\-/](?P<month>[01]?\d)[.\-/](?P<year>(?:19|20)?\d{2})(?!\d)",
        stem,
    )
    if numeric:
        chosen_year = year_hint or _normalise_year(int(numeric.group("year")))
        try:
            return date(chosen_year, int(numeric.group("month")), int(numeric.group("day")))
        except ValueError:
            pass

    compact_six = re.match(r"\s*(?P<value>\d{6})(?:\D|$)", stem)
    if compact_six:
        value = compact_six.group("value")
        first_year = _normalise_year(int(value[:2]))
        if year_hint is None or first_year == year_hint:
            try:
                return date(year_hint or first_year, int(value[2:4]), int(value[4:6]))
            except ValueError:
                pass
        try:
            return date(year_hint or _normalise_year(int(value[4:])), int(value[2:4]), int(value[:2]))
        except ValueError:
            pass

    # MW/RS recordings commonly begin MMDD, optionally after a four-digit year.
    leading_mmdd = re.match(r"\s*(?:(?:19|20)\d{2}[ _-]+)?(?P<mmdd>\d{4})(?:\D|$)", stem)
    if leading_mmdd and year_hint is not None:
        mmdd = leading_mmdd.group("mmdd")
        try:
            return date(year_hint, int(mmdd[:2]), int(mmdd[2:]))
        except ValueError:
            pass
    return None


def _source_from_text(value: str) -> Optional[str]:
    candidates = {match.group(1).upper() for match in _SOURCE_TOKEN_RE.finditer(value)}
    if len(candidates) == 1:
        return next(iter(candidates))
    if not candidates and re.search(r"(?<![A-Za-z])Group(?![A-Za-z])", value, re.I):
        return "GP"
    return None


def _path_parts(value: Optional[Path]) -> list[str]:
    if value is None:
        return []
    return [part for part in Path(value).parts if part not in {"", "."}]


def clean_publication_title(value: str) -> str:
    """Remove dates, tape IDs and processing labels while preserving topic words."""

    title = Path(value).stem.replace("_", " ")
    title = re.sub(
        r"\s*[\[(](?:cleaned(?: up)?|processed|whisper(?: transcript)?|"
        r"glm(?: transcript)?|transcript|final(?: copy| transcript)?)[^\])]*[\])]\s*$",
        "",
        title,
        flags=re.I,
    )
    title = re.sub(
        r"\s+(?:mixdown\d*|mono|track\s*\d+)(?:\s+.*)?$",
        "",
        title,
        flags=re.I,
    )
    title = re.sub(
        r"\s*[-–—]\s*(?:cleaned(?: up)?|processed|whisper(?: transcript)?|glm(?: transcript)?|final transcript)\s*$",
        "",
        title,
        flags=re.I,
    )

    leading_patterns = (
        r"^\s*(?:19|20)\d{2}[ _-]+\d{4}(?:\D|$)\s*",
        r"^\s*\d{6}(?:\D|$)\s*",
        r"^\s*\d{4}(?:\D|$)\s*",
        r"^\s*\d{1,3}[A-Za-z]?\s*-\s*\d{2}\s*",
        r"^\s*L\d{2,6}\s*",
        r"^\s*Lecture\s+\d+(?:\s+(?:19|20)\d{2})?\s*",
        r"^\s*(?:19|20)\d{2}\s+",
    )
    for pattern in leading_patterns:
        title = re.sub(pattern, "", title, count=1, flags=re.I)

    title = re.sub(
        rf"\s+[0-3]?\d\s*(?:{_MONTH_PATTERN})\s*,?\s*(?:(?:19|20)?\d{{2}})?\s*$",
        "",
        title,
        flags=re.I,
    )
    title = re.sub(
        r"\s+[0-3]?\d[.\-/][01]?\d[.\-/](?:19|20)?\d{2}\s*$",
        "",
        title,
    )
    title = re.sub(r"\s+", " ", title).strip(" .,_-–—")
    archival_statuses = {
        "incomplete": "Incomplete",
        "poor quality": "Poor Quality",
        "extract": "Extract",
    }
    for raw_status, display_status in archival_statuses.items():
        title = re.sub(
            rf"\s*[\[(]\s*{re.escape(raw_status)}\s*[\])]\s*$",
            f" ({display_status})",
            title,
            flags=re.I,
        )
        title = re.sub(
            rf"\s*[-–—,]?\s*{re.escape(raw_status)}\s*$",
            f" ({display_status})",
            title,
            flags=re.I,
        )
    return _publication_title_case(title.strip())


def _publication_title_case(value: str) -> str:
    """Apply restrained title case without flattening acronyms or mixed-case names."""

    words = value.split()
    styled: list[str] = []
    for index, word in enumerate(words):
        prefix_match = re.match(r"^(?P<prefix>[^A-Za-z0-9]*)(?P<body>.*)$", word)
        prefix = prefix_match.group("prefix") if prefix_match else ""
        body = prefix_match.group("body") if prefix_match else word
        suffix_match = re.match(r"^(?P<body>.*?)(?P<suffix>[^A-Za-z0-9]*)$", body)
        core = suffix_match.group("body") if suffix_match else body
        suffix = suffix_match.group("suffix") if suffix_match else ""
        if not core:
            styled.append(word)
            continue

        parts = core.split("-")
        rendered_parts: list[str] = []
        for part_index, part in enumerate(parts):
            lowered = part.casefold()
            is_edge = index == 0 or index == len(words) - 1
            if (
                not part
                or (part.isupper() and len(part) > 1)
                or any(character.isupper() for character in part[1:])
                or any(character.isdigit() for character in part)
            ):
                rendered = part
            elif lowered in _TITLE_MINOR_WORDS and not is_edge and part_index == 0:
                rendered = lowered
            else:
                rendered = part[:1].upper() + part[1:].lower()
            rendered_parts.append(rendered)
        styled.append(prefix + "-".join(rendered_parts) + suffix)
    return " ".join(styled)


def _fallback_title(genre: Optional[str], lecture_date: Optional[date]) -> str:
    genre_value = (genre or "").strip()
    if re.search(r"exercise", genre_value, re.I):
        base = "Spiritual Exercise"
    elif re.search(r"teaching", genre_value, re.I):
        base = "Spiritual Teaching"
    else:
        base = "Lecture"
    if lecture_date:
        return f"{base} - {format_publication_date(lecture_date)}"
    return base


def infer_publication_metadata(
    source_path: Path,
    relative_source_path: Optional[Path] = None,
    *,
    embedded_metadata: Optional[Mapping[str, str]] = None,
    year: Optional[int] = None,
    source_type: Optional[str] = None,
    title: Optional[str] = None,
) -> PublicationMetadata:
    """Infer publication fields using tags, then relative path, then filename."""

    source_path = Path(source_path)
    tags = {
        str(key).casefold(): str(value).strip()
        for key, value in (
            embedded_metadata
            if embedded_metadata is not None
            else read_embedded_audio_metadata(source_path)
        ).items()
        if str(value).strip()
    }
    relative_parts = _path_parts(relative_source_path)
    absolute_parts = _path_parts(source_path.parent)

    tag_date = _full_date_from_tag(tags.get("date", ""))
    tag_year = _year_from_text(tags.get("date", "")) or _year_from_text(tags.get("album", ""))
    relative_year = next(
        (
            inferred
            for part in reversed(relative_parts)
            if (inferred := _year_from_text(part)) is not None
        ),
        None,
    )
    absolute_year = next(
        (
            inferred
            for part in reversed(absolute_parts)
            if (inferred := _year_from_text(part)) is not None
        ),
        None,
    )
    path_year = relative_year or absolute_year
    year_hint = year or tag_year or path_year or _year_from_text(source_path.stem)
    lecture_date = tag_date
    if lecture_date and year is not None and lecture_date.year != year:
        try:
            lecture_date = lecture_date.replace(year=year)
        except ValueError:
            lecture_date = None
    if lecture_date is None:
        lecture_date = _date_from_filename(source_path.name, year_hint)

    selected_source = source_type.upper() if source_type else None
    if selected_source not in SOURCE_TYPES:
        selected_source = _source_from_text(tags.get("album", ""))
    if selected_source is None:
        for path_parts in (relative_parts, absolute_parts):
            for part in reversed(path_parts):
                selected_source = _source_from_text(part)
                if selected_source:
                    break
            if selected_source:
                break
    if selected_source is None:
        selected_source = _source_from_text(source_path.stem)

    selected_title = clean_publication_title(title or tags.get("title", ""))
    if not selected_title:
        selected_title = clean_publication_title(source_path.name)
    if not selected_title or selected_title.casefold() in {"lecture", "recording", "audio"}:
        selected_title = _fallback_title(tags.get("genre"), lecture_date)

    return PublicationMetadata(
        title=selected_title,
        lecture_date=lecture_date,
        source_type=selected_source,
        artist=tags.get("artist") or "Dr Philip Groves",
        album=tags.get("album") or None,
        genre=tags.get("genre") or None,
    )


__all__ = [
    "PublicationMetadata",
    "SOURCE_TYPES",
    "clean_publication_title",
    "format_publication_date",
    "infer_publication_metadata",
    "read_embedded_audio_metadata",
]
