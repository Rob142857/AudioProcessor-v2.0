"""
Prosody-based Sentence Merger

Uses pitch contours and intonation patterns to intelligently merge sentences
that Whisper incorrectly split at mid-sentence pauses.

Approach:
- Extract pitch at each sentence boundary timestamp
- If pitch is sustained/rising (not falling), it's likely a continuation
- If pitch drops significantly, it's likely a true sentence end
- Merge incorrectly split sentences based on prosodic evidence
"""

import re
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class ProsodySentenceMerger:
    """
    Merges incorrectly split sentences using pitch/prosody analysis.
    """
    
    def __init__(self, pitch_drop_threshold: float = 0.15):
        """
        Initialize the prosody-based sentence merger.
        
        Args:
            pitch_drop_threshold: Minimum pitch drop (as fraction) to consider true sentence end
                                 0.15 = 15% pitch drop required
        """
        self.pitch_drop_threshold = pitch_drop_threshold
        self.librosa = None
        self._initialize_audio_lib()
    
    def _initialize_audio_lib(self):
        """Load audio analysis library."""
        try:
            import librosa
            self.librosa = librosa
            print("✅ Prosody analysis available (librosa)")
        except ImportError:
            print("⚠️  librosa not installed - prosody analysis unavailable")
            print("   Install with: pip install librosa")
    
    def analyze_sentence_boundaries(self, audio_path: str, segments: List[dict]) -> List[bool]:
        """
        Analyze each segment boundary to determine if it's a true sentence end.
        
        Args:
            audio_path: Path to audio file
            segments: List of segment dicts with 'start', 'end', 'text' keys
        
        Returns:
            List of booleans - True if segment should end sentence, False if should merge
        """
        if not self.librosa or not segments:
            return [True] * len(segments)  # Default: keep all breaks
        
        try:
            # Load audio
            y, sr = self.librosa.load(audio_path, sr=22050)
            
            # Extract pitch contour
            pitches, magnitudes = self.librosa.piptrack(y=y, sr=sr, fmin=75, fmax=400)
            
            # Analyze each boundary
            is_sentence_end = []
            
            for i, segment in enumerate(segments):
                if i == len(segments) - 1:
                    # Last segment is always sentence end
                    is_sentence_end.append(True)
                    continue
                
                # Get pitch around boundary (end of current segment)
                boundary_time = segment['end']
                
                # Sample pitch before and after boundary
                pitch_before = self._get_pitch_at_time(pitches, magnitudes, sr, boundary_time - 0.2)
                pitch_after = self._get_pitch_at_time(pitches, magnitudes, sr, boundary_time + 0.2)
                
                if pitch_before is None or pitch_after is None:
                    # Can't determine - use linguistic heuristics
                    is_sentence_end.append(self._linguistic_heuristic(segment['text']))
                    continue
                
                # Calculate pitch drop
                pitch_drop = (pitch_before - pitch_after) / pitch_before if pitch_before > 0 else 0
                
                # True sentence end: significant pitch drop
                # Continuation: pitch sustained or rising
                is_true_end = pitch_drop > self.pitch_drop_threshold
                is_sentence_end.append(is_true_end)
            
            return is_sentence_end
            
        except Exception as e:
            print(f"⚠️  Prosody analysis failed: {e}")
            return [True] * len(segments)
    
    def _get_pitch_at_time(self, pitches, magnitudes, sr, time_sec):
        """Extract dominant pitch at specific time."""
        try:
            frame = int(time_sec * sr / 512)  # librosa default hop_length
            if frame >= pitches.shape[1]:
                frame = pitches.shape[1] - 1
            
            # Get pitch with highest magnitude at this frame
            pitch_slice = pitches[:, frame]
            mag_slice = magnitudes[:, frame]
            
            if mag_slice.max() == 0:
                return None
            
            pitch = pitch_slice[mag_slice.argmax()]
            return pitch if pitch > 0 else None
            
        except Exception:
            return None
    
    def _linguistic_heuristic(self, text: str) -> bool:
        """
        Fallback heuristic when prosody unavailable.
        AGGRESSIVE merging - assume continuation unless strong evidence of sentence end.
        """
        text = text.strip()
        
        # Count words
        word_count = len(text.split())
        
        # Anything under 8 words: MERGE (unless has proper ending punctuation and reasonable length)
        if word_count < 8:
            # Only allow break if ends with proper punctuation AND has at least 5 words
            if word_count >= 5 and re.search(r'[.!?]\s*$', text):
                # Check it's not a fragment like "Is red." or "For example."
                if word_count <= 3 or re.match(r'^(For example|Is |Are |Was |Were |And |But |Or |So )', text, re.IGNORECASE):
                    return False  # Still merge
                return True  # OK to end
            return False  # Merge
        
        # Strong indicators of continuation (should NOT end sentence)
        continuation_patterns = [
            r'\b(and|but|or|so|because|if|when|while|as|for|with|to|in|on|at|by|from|about|into|through|of|that|which|who)\s*$',
            r'\b(the|a|an|this|that|these|those|our|my|your|his|her|its|their|some|any|each|every|all)\s*$',
            r'\b(is|are|was|were|be|been|being|have|has|had|will|would|should|could|may|might|must|can|do|does|did)\s*$',
            r',\s*$',  # Ends with comma
            r':\s*$',  # Ends with colon
            r'-\s*$',  # Ends with dash
            r'\.\s*$',  # Even with period, check other factors
        ]
        
        for pattern in continuation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False  # Continuation, not sentence end
        
        # Check for proper sentence structure - needs both subject and predicate
        # If missing verb, definitely continue
        has_verb = re.search(r'\b(is|are|was|were|be|been|being|have|has|had|will|would|should|could|may|might|must|can|do|does|did|go|goes|went|come|came|make|made|take|took|get|got|see|saw|know|knew|think|thought|say|said|tell|told)\b', text, re.IGNORECASE)
        
        if not has_verb and word_count < 15:
            return False  # No verb = fragment = merge
        
        # Must be reasonably long (20+ words) to be a complete thought
        if word_count < 20:
            # Check if it reads as complete (has subject + verb + object/complement)
            # Simple heuristic: if it has conjunctions mid-sentence, likely complete
            if re.search(r'\b(and|but|or)\b', text, re.IGNORECASE) and word_count >= 12:
                return True
            return False  # Default: merge shorter sentences
        
        # Long sentence (20+) with verb: likely complete
        return True
    
    def merge_segments(self, segments: List[dict], is_sentence_end: List[bool]) -> str:
        """
        Merge segments based on prosody analysis.
        
        Args:
            segments: List of segment dicts with 'text' key
            is_sentence_end: Boolean list indicating true sentence boundaries
        
        Returns:
            Merged text with proper sentence breaks
        """
        if not segments:
            return ""
        
        merged_text = []
        current_sentence = []
        
        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            current_sentence.append(text)
            
            if is_sentence_end[i]:
                # Join current sentence and add to output
                sentence = ' '.join(current_sentence)
                # Ensure proper punctuation
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                merged_text.append(sentence)
                current_sentence = []
        
        # Handle any remaining text
        if current_sentence:
            sentence = ' '.join(current_sentence)
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            merged_text.append(sentence)
        
        return ' '.join(merged_text)


def create_prosody_merger(pitch_drop_threshold: float = 0.15) -> ProsodySentenceMerger:
    """
    Factory function to create a prosody sentence merger.
    
    Args:
        pitch_drop_threshold: Minimum pitch drop to consider true sentence end (0.15 = 15%)
    
    Returns:
        Configured ProsodySentenceMerger instance
    """
    return ProsodySentenceMerger(pitch_drop_threshold=pitch_drop_threshold)


if __name__ == "__main__":
    # Test
    print("Prosody-based Sentence Merger")
    print("=" * 80)
    
    merger = create_prosody_merger()
    
    # Example segments (would come from Whisper with timestamps)
    test_segments = [
        {'start': 0.0, 'end': 2.5, 'text': 'And we start off just using the fingertips as lightly as possible'},
        {'start': 2.5, 'end': 5.0, 'text': 'Just lightly touching the forehead'},
        {'start': 5.0, 'end': 7.5, 'text': 'Stroking over the head as lightly as possible'},
        {'start': 7.5, 'end': 9.0, 'text': 'Then going down'},
        {'start': 9.0, 'end': 11.0, 'text': 'Behind the ears bringing the fingers down'},
    ]
    
    print("\nWithout prosody (would need actual audio):")
    print("Using linguistic heuristics as fallback...")
    
    # Simulate analysis
    is_end = [merger._linguistic_heuristic(seg['text']) for seg in test_segments]
    print(f"Sentence boundaries: {is_end}")
    
    merged = merger.merge_segments(test_segments, is_end)
    print(f"\nMerged text:\n{merged}")
