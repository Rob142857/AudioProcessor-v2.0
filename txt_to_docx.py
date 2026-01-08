import argparse
import os
import re
from datetime import date
from pathlib import Path
from typing import Optional

from docx import Document  # type: ignore
from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore
from docx.shared import RGBColor  # type: ignore


def infer_year_from_parent(folder_name: str) -> int:
    """Extract a 4-digit year from a folder name, e.g. '1988 MW' -> 1988.

    Raises ValueError if no 4-digit sequence is found.
    """
    m = re.search(r"(19\d{2}|20\d{2})", folder_name)
    if not m:
        raise ValueError(f"Could not find 4-digit year in folder name: {folder_name!r}")
    return int(m.group(1))


def infer_year_from_ancestors(start: Path) -> int:
    """Walk up from a path until we find a folder name containing a 4-digit year.

    This allows layouts like '.../1988 MW/Temp/0202 Fishes.txt' but will also
    work if the year folder is higher up the tree. Raises ValueError if no
    suitable folder is found.
    """
    for folder in [start] + list(start.parents):
        try:
            return infer_year_from_parent(folder.name)
        except ValueError:
            continue
    raise ValueError(
        f"Could not find 4-digit year in any parent folder names starting from {start!r}. "
        "Pass --year explicitly when running this script."
    )


def infer_date_from_filename(filename: str, year: int) -> date:
    """Infer a calendar date from a filename like '0202 Fishes.txt'.

    Assumes the first four digits are MMDD for the given year.
    """
    stem = Path(filename).stem
    m = re.match(r"(\d{4})(?:\D|$)", stem)
    if not m:
        raise ValueError(f"Could not find leading MMDD in filename: {filename!r}")
    mmdd = m.group(1)
    month = int(mmdd[:2])
    day = int(mmdd[2:])
    return date(year, month, day)


def make_title_from_filename(filename: str) -> str:
    """Use the words in the filename (after the leading MMDD) as the title.

    Example: '0202 Fishes.txt' -> 'Fishes'
    """
    stem = Path(filename).stem
    # Drop leading MMDD and optional separator
    m = re.match(r"\d{4}[_\- ]*(.*)", stem)
    title_part = m.group(1) if m and m.group(1) else stem
    return title_part.strip() or stem


def get_source_path_from_header(txt_path: Path) -> Optional[Path]:
    """Read the first 'Source:' line in the txt file and return its path, if any.

    The header is expected to look like:
        Source: C:\\path\\to\\audio.mp3
    """
    text = txt_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("Source:"):
            raw = line[len("Source:") :].strip()
            if raw:
                return Path(raw)
            break
    return None


def load_body_text(txt_path: Path) -> str:
    """Load transcript text, stripping the Source/Output header lines if present."""
    text = txt_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    body_lines = []
    for line in lines:
        if line.startswith("Source:") or line.startswith("Output:"):
            continue
        body_lines.append(line)
    body = "\n".join(body_lines).lstrip("\n")
    return body


def add_paragraphs_from_text(doc: Document, body: str) -> None:
    """Add justified paragraphs to a Document, preserving blank-line paragraph breaks."""
    # Split on blank lines (one or more empty/whitespace-only lines)
    blocks = re.split(r"\n\s*\n", body)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        para = doc.add_paragraph(block)
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def convert_txt_to_docx(txt_path: Path, year: Optional[int] = None) -> Path:
    if not txt_path.is_file():
        raise FileNotFoundError(f"Input file not found: {txt_path}")

    # Infer year from the directory structure if not provided explicitly.
    # Preferred source is the path embedded in the header (first 'Source:'
    # line), which points to the original audio file location, e.g.
    #   Source: .../1988 MW/0202 Fishes.mp3
    # We walk up from that path to find a 4-digit year. If the header is
    # missing or unusable, we fall back to walking up from the txt location.
    if year is None:
        source_path = get_source_path_from_header(txt_path)
        if source_path is not None:
            year = infer_year_from_ancestors(source_path.parent)
        else:
            year = infer_year_from_ancestors(txt_path.parent)

    # Infer date from filename MMDD
    d = infer_date_from_filename(txt_path.name, year)
    weekday = d.strftime("%A")
    month_name = d.strftime("%B")
    date_line = f"{weekday}, {d.day} {month_name} {d.year}"

    # Build title and body
    title = make_title_from_filename(txt_path.name)
    body = load_body_text(txt_path)

    # Create DOCX
    doc = Document()

    # Title
    heading = doc.add_heading(title, level=0)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Date line
    p_date = doc.add_paragraph(date_line)
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Author line
    p_author = doc.add_paragraph("by Dr Philip Groves")
    p_author.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Blank line before body
    doc.add_paragraph("")

    # Body paragraphs
    add_paragraphs_from_text(doc, body)

    # Output path: same folder, .docx extension
    out_path = txt_path.with_suffix(".docx")
    doc.save(out_path)
    return out_path


def convert_txt_to_docx_from_text(body_text: str, source_audio_path: Path, year: Optional[int] = None, metadata: Optional[dict] = None) -> Path:
    """Convert transcript text directly to DOCX, saving next to the source audio file.
    
    Args:
        body_text: The formatted transcript text to include in the document
        source_audio_path: Path to the original audio/video file (DOCX will be saved next to it)
        year: Optional year override for date inference
        metadata: Optional dict with transcription metadata (model, device, time_taken, preprocessing)
    
    Returns:
        Path to the created DOCX file
    """
    # Infer year from the directory structure if not provided explicitly.
    if year is None:
        year = infer_year_from_ancestors(source_audio_path.parent)

    # Infer date from filename MMDD (using source audio filename)
    d = infer_date_from_filename(source_audio_path.name, year)
    weekday = d.strftime("%A")
    month_name = d.strftime("%B")
    date_line = f"{weekday}, {d.day} {month_name} {d.year}"

    # Build title from source audio filename
    title = make_title_from_filename(source_audio_path.name)

    # Create DOCX
    doc = Document()

    # Title
    heading = doc.add_heading(title, level=0)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Date line
    p_date = doc.add_paragraph(date_line)
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Author line
    p_author = doc.add_paragraph("by Dr Philip Groves")
    p_author.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Blank line before body
    doc.add_paragraph("")

    # Body paragraphs
    add_paragraphs_from_text(doc, body_text)
    
    # Add metadata footer if provided
    if metadata:
        # Add spacing before metadata
        doc.add_paragraph("")
        doc.add_paragraph("")
        
        # Add horizontal line separator
        separator = doc.add_paragraph("_" * 80)
        separator.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add metadata section
        meta_header = doc.add_paragraph("Transcription Information")
        meta_header.runs[0].bold = True
        meta_header.runs[0].font.size = 10 * 12700  # 10pt in EMUs
        
        # Add metadata details
        details = [
            f"Model: {metadata.get('model', 'Unknown')}",
            f"Device: {metadata.get('device', 'Unknown')}",
            f"Processing Time: {metadata.get('time_taken', 'Unknown')}",
            f"Audio Preprocessing: {metadata.get('preprocessing', 'None')}"
        ]
        
        for detail in details:
            p = doc.add_paragraph(detail)
            p.runs[0].font.size = 9 * 12700  # 9pt
            p.runs[0].font.italic = True
        
        # Add note about removal
        note = doc.add_paragraph("(This information can be deleted if not needed)")
        note.runs[0].font.size = 8 * 12700  # 8pt
        note.runs[0].font.italic = True
        note.runs[0].font.color.rgb = RGBColor(128, 128, 128)  # Gray color

    # Output path: same folder as source audio, .docx extension
    out_path = source_audio_path.with_suffix(".docx")
    doc.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert transcript .txt file(s) to formatted .docx files ready to print. Can process a single file or all .txt files in a folder.")
    parser.add_argument("input", help="Path to a transcript .txt file or a folder containing .txt files")
    parser.add_argument("--year", type=int, help="Override inferred year (e.g. 1988)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    
    # Determine if input is a file or folder
    if input_path.is_file():
        # Single file
        txt_files = [input_path]
    elif input_path.is_dir():
        # Folder: find all .txt files
        txt_files = sorted(input_path.glob("*.txt"))
        if not txt_files:
            print(f"No .txt files found in {input_path}")
            return
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Convert each file
    for txt_path in txt_files:
        try:
            out_path = convert_txt_to_docx(txt_path, year=args.year)
            print(f"Created DOCX: {out_path}")
        except Exception as e:
            print(f"Error processing {txt_path.name}: {e}")


if __name__ == "__main__":
    main()
