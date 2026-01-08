# Text Processing Improvements

## Recent Enhancements

This document describes the recent improvements made to the transcription system for Dr Philip Groves' lectures.

## 1. Australian/British Spelling Conversion

**File:** `australian_spelling.py`

Automatically converts American spellings to Australian/British spellings:

- `-ize` → `-ise` (realize → realise, recognize → recognise)
- `-or` → `-our` (color → colour, favor → favour, center → centre)
- `-er` → `-re` (center → centre, theater → theatre)
- `-og` → `-ogue` (catalog → catalogue, dialog → dialogue)
- And many more patterns...

**Usage:**
```python
from australian_spelling import normalize_text

# Convert text to Australian spelling and fix number formatting
text = normalize_text(text, use_australian_spelling=True, fix_numbers=True)
```

The conversion is **automatically applied** during DOCX generation.

## 2. Number Formatting

Fixes malformed numbers in transcripts:
- `2, 500` → `2,500`
- `1 , 000` → `1,000`
- Removes extra spaces around commas in numbers

This is integrated into the `normalize_text()` function and applied automatically.

## 3. Enhanced Special Words Dictionary

**File:** `special_words.txt`

Added specialized terminology for amphibians and biology:

```
caecilians
Phyllobates
bufotenine
anurans
urodeles
gymnophiona
```

These words help Whisper correctly recognize technical terms during transcription.

**To add more words:** Simply edit `special_words.txt` and add one term per line.

## 4. Improved Title Format

**File:** `txt_to_docx.py`

DOCX files now use a standardized header format:

```
[Title from filename]

Lecture XX given on DDth of Month YYYY by Dr Philip W Groves

Transcript:

[Content...]
```

**Example:**
```
Frogs and Toads

Lecture 14 given on 14th of January 1992 by Dr Philip W Groves

Transcript:

[Lecture content begins here...]
```

### Features:
- Automatically extracts lecture number from filename
- Formats dates with proper ordinal suffixes (1st, 2nd, 3rd, 21st, etc.)
- Falls back gracefully if date/lecture number cannot be determined
- Works with various filename formats

## 5. Question & Answer Separation

**File:** `separate_qa.py`

A post-processing tool to separate questions from Dr Groves' replies.

**Usage:**

```bash
# Analyze transcript for Q&A patterns
python separate_qa.py transcript.txt --analyze-only

# Process and create formatted output
python separate_qa.py transcript.txt --output transcript_qa.txt

# Auto-generate output filename (adds _qa suffix)
python separate_qa.py transcript.txt
```

**Output format:**
```
Q: Can you explain how caecilians differ from other amphibians?

Dr Groves: Well, caecilians are quite unique. They are limbless amphibians...

Q: What about their habitat?

Dr Groves: They live primarily in tropical regions...
```

**Important Notes:**
- ⚠️ This is an automated attempt using pattern matching
- Manual review is **strongly recommended**
- The script detects common question patterns and marks sections accordingly
- You may need to adjust Q: and Dr Groves: markers for accuracy

### Question Detection Patterns:
- Sentences ending with `?`
- Questions starting with: what, where, when, why, how, who, which
- Common question phrases: can you, could you, is it, are they, etc.

## Integration with Transcription Pipeline

All improvements are automatically integrated:

1. **During transcription:** Special words are loaded from `special_words.txt`
2. **During DOCX generation:** 
   - Australian spelling conversion is applied
   - Number formatting is corrected
   - New title format is used
3. **Post-processing (manual):** Run `separate_qa.py` on finished transcripts

## Configuration

### Enable/Disable Australian Spelling

In `txt_to_docx.py`, the function accepts a parameter:

```python
convert_txt_to_docx_from_text(
    body_text, 
    source_audio_path, 
    use_australian_spelling=True  # Set to False to disable
)
```

### Customize Spelling Rules

Edit `australian_spelling.py` and modify the `SPELLING_MAP` dictionary:

```python
SPELLING_MAP: Dict[str, str] = {
    "realize": "realise",
    "color": "colour",
    # Add your custom mappings here
}
```

## Testing

Test the Australian spelling converter:

```bash
python australian_spelling.py
```

This will run a test with sample text and show the conversions.

## Future Enhancements

Potential improvements for consideration:

1. **AI-powered Q&A separation** using speaker diarization
2. **Custom title mapping** file for specific lecture titles
3. **Terminology glossary** generation from special_words.txt
4. **Batch Q&A processing** for multiple files
5. **Interactive Q&A editor** GUI tool

## Files Modified

- `australian_spelling.py` - NEW: Spelling conversion module
- `separate_qa.py` - NEW: Q&A separation tool
- `special_words.txt` - UPDATED: Added amphibian terminology
- `txt_to_docx.py` - UPDATED: Integrated Australian spelling, new title format
- `TEXT_PROCESSING_IMPROVEMENTS.md` - NEW: This documentation

## Support

For questions or issues with these improvements, check:
1. The relevant Python file's docstrings
2. This documentation
3. The main README.md for general system information
