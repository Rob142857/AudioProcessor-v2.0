"""
Lightweight Whisper Punctuation Fixes

Fixes common Whisper large-v3 punctuation errors without heavy models.
Targets specific known issues like mid-sentence breaks after short words.
"""

import re


def fix_whisper_punctuation(text: str) -> str:
    """
    Generic pre-processing before BERT refinement.
    Fixes obvious mid-sentence breaks that BERT sometimes creates or misses.
    """
    
    # Pattern 1: Short common words + period + Capitalized word (obvious continuations)
    # Matches: "our. Fingers", "the. Lighter", "a. Result", etc.
    text = re.sub(
        r'\b(a|an|the|our|as|of|to|in|on|at|by|for|with|from|but|and|or)\.\s+([A-Z][a-z]{2,})\b',
        r'\1 \2',
        text,
        flags=re.IGNORECASE
    )
    
    # Pattern 2: "down. Behind", "over. Inside" - preposition + period + preposition/adverb
    text = re.sub(
        r'\b(down|over|under|behind|above|below)\.\s+(Behind|Inside|Outside|Above|Below|Over|Under)\b',
        r'\1 \2',
        text
    )
    
    return text


def fix_whisper_punctuation_safe(text: str, verbose: bool = False) -> str:
    """
    Safe wrapper with error handling and optional reporting.
    """
    try:
        original_text = text
        fixed_text = fix_whisper_punctuation(text)
        
        if verbose:
            # Count changes
            changes = 0
            if original_text != fixed_text:
                # Simple diff count
                changes = abs(len(original_text) - len(fixed_text))
                print(f"✅ Fixed {changes} characters in Whisper punctuation errors")
        
        return fixed_text
        
    except Exception as e:
        print(f"⚠️  Punctuation fix failed: {e}")
        return text


if __name__ == "__main__":
    # Test with the problematic text
    test_text = """that's the way nature beautifies harsh surroundings if you go aqualung. Diving you can sometimes find caves along the coastline if you have adequate illumination you can dive in and the walls are very often coloured very brightly with all. Sorts of anemones, marine growths and horny corals and so on and it's a play of colors like looking at a tapestry."""
    
    print("ORIGINAL:")
    print(test_text)
    print("\n" + "="*80 + "\n")
    
    fixed = fix_whisper_punctuation_safe(test_text, verbose=True)
    
    print("FIXED:")
    print(fixed)
