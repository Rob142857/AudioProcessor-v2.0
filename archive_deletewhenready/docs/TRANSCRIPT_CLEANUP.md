# Transcript Cleanup Notes

This project sometimes post-processes raw Whisper speech-to-text `.txt` outputs to make them easier to read **without changing any words**.

## Goals

- Preserve every original word from the Whisper transcript
- Fix obvious spacing/punctuation artefacts Whisper leaves behind
- Normalise sentence case where safe
- Leave paragraph breaks (blank lines) as they are in the source file

## Rules Applied

The following automated fixes were applied to files such as `Temp/0202 Fishes.txt`:

- **Decimal spacing**  
  - Fix patterns like `0. 2` → `0.2` (remove spaces after a decimal point inside a number).

- **Thousands separators**  
  - Fix patterns like `22, 000` → `22,000` (remove spaces after a comma when followed by exactly three digits).

- **Spaces before punctuation**  
  - Collapse stray spaces before sentence-ending punctuation:  
    - e.g. `word !` → `word!`, `word ?` → `word?`, `word .` → `word.`

- **Sentence capitalization (within a line)**  
  - When a sentence-ending punctuation mark is followed by a lowercase letter **on the same line**, capitalize it:  
    - e.g. `... end. next` → `... end. Next`  
  - This is line-local only; no words are added or removed.

- **Header path tidy-up**  
  - Clean minor spacing issues in the header lines:  
    - `0202 Fishes. mp3` → `0202 Fishes.mp3`  
    - `0202 Fishes. txt` → `0202 Fishes.txt`

- **Manual micro-fixes for obvious mid-sentence splits**  
  When Whisper inserted a stray full stop mid-sentence, it was removed, keeping the words intact, e.g.:
  - `things that. We have` → `things that we have`  
  - `oh just. Grab that fish` → `oh just grab that fish`

## What Was *Not* Changed

- No words were added, removed, or reordered.
- Paragraph boundaries (blank lines) were left exactly as in the original raw transcript.
- Domain-specific terminology and phrasing were left untouched.

## Reusing This Approach

For other `.txt` transcripts, you can reuse the same strategy:

1. Run a small Python script over `Temp/*.txt` that:
   - Applies the regex-style spacing and capitalization fixes above.
   - Leaves all text content and blank lines unchanged.
2. Optionally add bespoke one-off fixes for any obviously broken `word. Next` splits that Whisper has created.

This keeps the transcripts faithful to the audio while making them much more readable for printing or conversion to `.docx`. 
