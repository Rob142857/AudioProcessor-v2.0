# Transcription Quality Fixes

## Issues Identified and Fixed

### Issue 1: Missing Punctuation in Transcriptions

**Problem:** Transcripts were generated without any punctuation, making them difficult to read.

**Root Cause:** The transcription was using `condition_on_previous_text=False` in Whisper, which disables context awareness and can result in poor punctuation generation.

**Fix:** Changed to `condition_on_previous_text=True` in [transcribe_optimised.py](transcribe_optimised.py) line ~1277

**Impact:** Whisper will now use context from previous segments to generate proper punctuation and maintain better sentence flow.

---

### Issue 2: Severe Repetition Loops

**Problem:** Transcriptions contained phrases repeated dozens or hundreds of times, e.g., "telekinesis is the ability to know what is taking place at a distance from oneself" repeated 50+ times.

**Root Cause:** 
1. The `condition_on_previous_text=False` setting was causing Whisper to lose context and fall into repetition loops
2. The repetition detection regex only caught short phrases (1-6 words) and comma-separated patterns
3. Max repeat cap was only 3, too low to catch severe issues

**Fix:** 
1. Enabled `condition_on_previous_text=True` to prevent loops at the source
2. Enhanced `_collapse_repetitions()` function to:
   - Detect longer repeated phrases (up to 20 words)
   - Handle both space-separated and comma-separated repetitions
   - Increased default max_repeats from 3 to 10
   - Increased cap limit from 10 to 20
   - Uses the GUI's max_repeat_cap setting (default 10, range 1-20)

**Code Changes:**
- [transcribe_optimised.py](transcribe_optimised.py) line ~534: Enhanced `_collapse_repetitions()` function
- [transcribe_optimised.py](transcribe_optimised.py) line ~2995: Now uses `TRANSCRIBE_MAX_REPEAT_CAP` environment variable

---

### Issue 3: Output Truncation (Long Files Cut Off)

**Problem:** Longer transcriptions (30+ pages) were being cut off at ~12 pages in the DOCX output.

**Root Cause:** Unknown - potentially an issue with text handling or DOCX generation. Needs investigation with actual files.

**Diagnostic Improvements:**
1. Now saves intermediate `.txt` file next to source audio for debugging
2. Added diagnostic output showing text length at various stages:
   - Input text character/word count
   - After Australian spelling normalization
   - Before DOCX generation

**Files Modified:**
- [transcribe_optimised.py](transcribe_optimised.py) line ~3025: Now saves TXT file before DOCX
- [txt_to_docx.py](txt_to_docx.py) line ~207: Added diagnostic output for text lengths

**Action Required:** 
- Check the intermediate `.txt` files to see if truncation happens before or during DOCX generation
- If TXT file is complete but DOCX is truncated, the issue is in python-docx library
- If TXT file is also truncated, the issue is in earlier processing

---

## Verification Steps

### For Punctuation Issues:
1. Run a new transcription on a file that previously had no punctuation
2. Check that sentences now have proper periods, commas, and capitalization
3. Compare with the intermediate `.txt` file to ensure punctuation is preserved through DOCX generation

### For Repetition Issues:
1. Re-transcribe files that had severe repetition
2. Check the console output for "Removed X music/hallucination pattern(s)" messages
3. Verify max_repeat_cap is set to 10 or higher in the GUI
4. Check that repeated phrases are limited to the configured max (default 10 occurrences)

### For Truncation Issues:
1. Transcribe a long file (>1 hour)
2. Check the console output for text length diagnostics
3. Compare the `.txt` file length with the `.docx` file length
4. If `.txt` is complete but `.docx` is truncated:
   - This is a python-docx library issue
   - Consider splitting very long documents into multiple parts
5. If `.txt` is also truncated:
   - Check for errors during transcription
   - Check available memory (RAM/VRAM)
   - Look for timeout issues

---

## Testing Recommendations

### Test 1: Short File with Punctuation
- Use a 5-10 minute audio file
- Verify proper punctuation throughout
- Check Australian spelling conversions

### Test 2: File with Repetitive Content
- Use a file known to have repetition issues
- Set max_repeat_cap to 10 in GUI
- Verify repetitions are capped properly

### Test 3: Long File (Full Length)
- Use a 60-90 minute file
- Monitor the `.txt` output first
- Check DOCX page count matches expected length
- Review diagnostic output for any anomalies

---

## Configuration

### GUI Settings:
- **Max Repeat Cap:** Default 10, range 1-20 (adjustable in GUI)
- **Model:** faster-whisper-large-v3 (default)
- **Domain Terms:** C:\Users\RobertEvans\Downloads\AudioProcessorAlphaVersion\special_words.txt

### Environment Variables (Set by GUI):
- `TRANSCRIBE_MAX_REPEAT_CAP`: Controls repetition detection threshold
- `TRANSCRIBE_VERBATIM`: Default "1" (enabled)
- `TRANSCRIBE_PARAGRAPH_GAP`: Default "1.8" seconds
- `TRANSCRIBE_QUALITY_MODE`: Set to "1" for higher quality

---

## Files Modified

1. **transcribe_optimised.py**
   - Line ~534: Enhanced `_collapse_repetitions()` function
   - Line ~1277: Changed `condition_on_previous_text` to True
   - Line ~2995: Use max_repeat_cap from environment
   - Line ~3025: Save intermediate TXT file

2. **txt_to_docx.py**
   - Line ~207: Added diagnostic output for text lengths
   - Previously: Added Australian spelling integration
   - Previously: Added new title format with lecture numbers

3. **australian_spelling.py**
   - Tested and verified punctuation preservation

---

## Known Limitations

1. **Q&A Separation:** Still requires manual post-processing with `separate_qa.py`
2. **Very Long Documents:** Python-docx may have performance issues with extremely long documents (>50 pages)
3. **Repetition Detection:** Works for most cases but may not catch all patterns (especially context-dependent repetitions)

---

## Future Improvements

1. Investigate python-docx alternatives for handling very long documents
2. Add automatic detection of transcription length to warn about potential truncation
3. Consider splitting very long transcriptions into chapters/sections
4. Implement better repetition detection using semantic similarity
5. Add automated Q&A detection during transcription (not just post-processing)

---

## Support

If issues persist:
1. Check the intermediate `.txt` file first
2. Review console output for diagnostic messages
3. Check the quality report JSON for statistics
4. Try re-transcribing with different max_repeat_cap values
5. Verify Whisper model is loading correctly

---

Last Updated: January 9, 2026
