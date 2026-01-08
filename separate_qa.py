"""
Question and Answer Separator for Dr Philip Groves Lectures

This script attempts to separate questions from Dr Groves' replies in lecture transcripts.
It uses simple heuristics to identify question patterns and format them differently.

Usage:
    python separate_qa.py input.txt [--output output.txt]
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple


def detect_question_patterns(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect potential questions in the transcript.
    
    Returns list of (start_pos, end_pos, question_text) tuples.
    """
    questions = []
    
    # Common question patterns
    patterns = [
        r'\b(?:can you|could you|would you|will you|do you|does it|is it|are they|was it|were they)\b[^.?!]*[.?]',
        r'\b(?:what|where|when|why|how|who|which)\b[^.?!]*[.?]',
        r'\b(?:is there|are there|was there|were there)\b[^.?!]*[.?]',
        r'\b[A-Z][^.?!]*\?',  # Any sentence ending with ?
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            questions.append((match.start(), match.end(), match.group(0)))
    
    # Sort by position and remove overlaps
    questions.sort(key=lambda x: x[0])
    
    # Remove overlapping matches (keep longer ones)
    filtered = []
    last_end = -1
    for start, end, q_text in questions:
        if start >= last_end:
            filtered.append((start, end, q_text))
            last_end = end
    
    return filtered


def format_qa_sections(text: str) -> str:
    """
    Format text with Q&A sections clearly marked.
    
    Adds "Q:" prefix for questions and "Dr Groves:" for replies.
    """
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    formatted = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Check if paragraph looks like a question
        is_question = (
            para.endswith('?') or
            re.match(r'\b(?:can you|could you|what|where|when|why|how|who)', para, re.IGNORECASE) or
            re.match(r'\b(?:is it|are they|do you|does it)', para, re.IGNORECASE)
        )
        
        if is_question:
            # Mark as question
            if not para.startswith('Q:'):
                formatted.append(f"Q: {para}")
            else:
                formatted.append(para)
        else:
            # Mark as Dr Groves' reply if previous was a question
            if formatted and formatted[-1].startswith('Q:'):
                if not para.startswith('Dr Groves:'):
                    formatted.append(f"Dr Groves: {para}")
                else:
                    formatted.append(para)
            else:
                # Regular paragraph (lecture content)
                formatted.append(para)
    
    return '\n\n'.join(formatted)


def analyze_speaker_patterns(text: str) -> dict:
    """
    Analyze the text to identify speaker patterns and provide statistics.
    """
    questions = detect_question_patterns(text)
    paragraphs = text.split('\n\n')
    
    stats = {
        'total_paragraphs': len([p for p in paragraphs if p.strip()]),
        'potential_questions': len(questions),
        'question_marks': text.count('?'),
        'question_percentage': 0.0
    }
    
    if stats['total_paragraphs'] > 0:
        stats['question_percentage'] = (stats['potential_questions'] / stats['total_paragraphs']) * 100
    
    return stats


def separate_qa_interactive(input_text: str) -> str:
    """
    Process transcript with interactive Q&A separation.
    
    This is a basic implementation - manual review is recommended.
    """
    print("\n📊 Analyzing transcript for Q&A patterns...")
    
    stats = analyze_speaker_patterns(input_text)
    print(f"   Total paragraphs: {stats['total_paragraphs']}")
    print(f"   Potential questions detected: {stats['potential_questions']}")
    print(f"   Question marks found: {stats['question_marks']}")
    print(f"   Estimated Q&A content: {stats['question_percentage']:.1f}%")
    
    print("\n⚙️  Applying Q&A formatting...")
    formatted = format_qa_sections(input_text)
    
    print("✅ Q&A separation complete")
    print("\n⚠️  Note: This is an automated attempt. Manual review is strongly recommended.")
    print("   Questions are marked with 'Q:' and replies with 'Dr Groves:'")
    
    return formatted


def main():
    parser = argparse.ArgumentParser(
        description="Separate questions and answers in Dr Philip Groves lecture transcripts"
    )
    parser.add_argument("input", help="Input transcript file (.txt)")
    parser.add_argument("--output", help="Output file (default: adds '_qa' suffix)")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze and report statistics, don't create output file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return
    
    # Read input
    print(f"📖 Reading transcript: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if args.analyze_only:
        # Just analyze and report
        stats = analyze_speaker_patterns(text)
        print("\n📊 Transcript Analysis:")
        print(f"   Total paragraphs: {stats['total_paragraphs']}")
        print(f"   Potential questions: {stats['potential_questions']}")
        print(f"   Question marks: {stats['question_marks']}")
        print(f"   Estimated Q&A content: {stats['question_percentage']:.1f}%")
        return
    
    # Process Q&A separation
    formatted = separate_qa_interactive(text)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_qa")
    
    # Write output
    print(f"\n💾 Saving formatted transcript: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted)
    
    print(f"\n✅ Complete! Output saved to: {output_path}")
    print("\n💡 Tip: Review the output manually to ensure accuracy.")
    print("   You may need to adjust Q: and Dr Groves: markers as needed.")


if __name__ == "__main__":
    main()
