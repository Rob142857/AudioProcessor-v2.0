"""
Punctuation Comparison Tool

Compares transcription output with and without advanced punctuation restoration.
Useful for evaluating the impact of the new punctuation model.
"""

import sys
import os
import argparse
from typing import Tuple

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compare_punctuation(text: str, show_diff: bool = True) -> Tuple[str, str]:
    """
    Compare text with and without punctuation restoration.
    
    Args:
        text: Original text (from Whisper)
        show_diff: Whether to highlight differences
        
    Returns:
        Tuple of (original, restored)
    """
    try:
        from punctuation_restorer import create_punctuation_restorer
        
        print("🔧 Initializing punctuation restorer...")
        restorer = create_punctuation_restorer()
        
        print("\n🔄 Processing with punctuation restoration...")
        restored = restorer.refine_transcription(text, aggressive=False)
        
        if show_diff:
            print("\n" + "="*80)
            print("COMPARISON RESULTS")
            print("="*80)
            
            print("\n📝 ORIGINAL (Whisper output):")
            print("-"*80)
            print(text)
            
            print("\n✨ RESTORED (With punctuation refinement):")
            print("-"*80)
            print(restored)
            
            print("\n📊 STATISTICS:")
            print("-"*80)
            print(f"Original length:    {len(text)} chars")
            print(f"Restored length:    {len(restored)} chars")
            print(f"Difference:         {len(restored) - len(text):+d} chars")
            
            # Count sentences
            import re
            orig_sentences = len(re.findall(r'[.!?]+', text))
            rest_sentences = len(re.findall(r'[.!?]+', restored))
            print(f"Original sentences: {orig_sentences}")
            print(f"Restored sentences: {rest_sentences}")
            print(f"Difference:         {rest_sentences - orig_sentences:+d}")
            
            # Count paragraphs
            orig_paras = len([p for p in text.split('\n\n') if p.strip()])
            rest_paras = len([p for p in restored.split('\n\n') if p.strip()])
            print(f"Original paragraphs: {orig_paras}")
            print(f"Restored paragraphs: {rest_paras}")
            print(f"Difference:          {rest_paras - orig_paras:+d}")
            
        return text, restored
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return text, text


def compare_from_file(filepath: str, output_file: str = None):
    """
    Compare punctuation for text from a file.
    
    Args:
        filepath: Path to text file
        output_file: Optional path to save restored text
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"📂 Reading from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    original, restored = compare_punctuation(text)
    
    if output_file:
        print(f"\n💾 Saving restored text to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(restored)
        print("✅ Saved!")


def main():
    parser = argparse.ArgumentParser(
        description="Compare transcription with and without punctuation restoration"
    )
    parser.add_argument(
        'input',
        nargs='?',
        help='Input text file to process (if not provided, uses built-in example)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file to save restored text'
    )
    parser.add_argument(
        '--no-diff',
        action='store_true',
        help='Skip showing differences, just process'
    )
    
    args = parser.parse_args()
    
    if args.input:
        compare_from_file(args.input, args.output)
    else:
        # Use example from user's request
        example_text = """What can be said about the difference between a system of development and a way of development? Now people like yourselves who are interested in doing something about your spiritual life, sooner or later will come to contact with a group, with maybe a school, maybe. A teacher, maybe somebody who has some knowledge of these things.
And it's very important in the first few months to try and decide whether that person or group or school is teaching a system or is indicating a way of development. There is a very big difference between them, and the results obtained also are very, very different. We have various systems of yoga. We have, within the sufi framework, particular and very narrow systems. The system arises when there are specific teachings, fixed and sometimes very rigid rules in the group or school that must be obeyed, and the doctrines mustn't be. Departed from.
Now, in such systems, often there is a code of secrecy, that people mustn't disclose the slightest thing that takes place in such groups. And so we find people belonging to a system, being told this is the only way to develop or it's one of the few ways that you have a chance to use in the course of your. Life and if you disclose anything to outside people you'll be booted out and you'll never have a chance of getting back into it. Now these are some of the typical rules that go along with the system of development. The system arises out of a set of beliefs. In the first place the beliefs were possibly quite rational and sound. Maybe they were derived from some useful, even a cosmic source."""
        
        print("📝 Using built-in example text (from your sample)\n")
        compare_punctuation(example_text, show_diff=not args.no_diff)
        
        if args.output:
            original, restored = compare_punctuation(example_text, show_diff=False)
            print(f"\n💾 Saving restored text to: {args.output}")
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(restored)
            print("✅ Saved!")


if __name__ == "__main__":
    main()
