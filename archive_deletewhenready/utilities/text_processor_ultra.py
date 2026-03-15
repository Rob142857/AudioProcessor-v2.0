"""
Ultra-Enhanced Text Processing Module for AudioProcessor Alpha Version

This module provides multi-threaded, multi-pass text processing with advanced NLP capabilities:

1. Parallel processing across multiple CPU cores
2. Multiple specialized text processing passes
3. Advanced sentence and paragraph segmentation
4. Better punctuation and capitalization
5. Context-aware formatting
6. Quality assessment and iterative improvement

Author: AudioProcessor Team
Date: September 2025
"""

import re
import os
import sys
import time
import threading
import multiprocessing
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Lightweight dependency only (local module)
from custom_dictionary import CustomDictionary


class UltraTextProcessor:
    """
    Ultra-enhanced text processor with parallel processing and multiple specialized passes.
    """

    def __init__(self, 
                 use_spacy: bool = True, 
                 use_nltk: bool = True,
                 use_parallel: bool = True,
                 max_workers: Optional[int] = None):
        """
        Initialize the ultra text processor.

        Args:
            use_spacy: Whether to use spaCy for advanced processing
            use_nltk: Whether to use NLTK for additional processing
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of worker threads (None = auto-detect)
        """
        # Heavy NLP disabled in ULTRA light mode; flags ignored intentionally
        self.use_spacy = False
        self.use_nltk = False
        self.use_parallel = use_parallel
        
        # Determine optimal worker count
        cpu_count = multiprocessing.cpu_count()
        if max_workers is None:
            # Use 75% of available cores for text processing to leave headroom
            self.max_workers = max(2, int(cpu_count * 0.75))
        else:
            self.max_workers = max(1, min(max_workers, cpu_count))

        # Initialize models
        self.nlp = None
        self.custom_dict = None
        self.quality_metrics = {}

        print(f"🧵 Ultra Text Processor: Using {self.max_workers} worker threads (of {cpu_count} available)")
        self._initialize_models()

    def _initialize_models(self):
        """Initialize available models with light defaults (no heavy NLP)."""
        print("🟡 ULTRA Light Mode: punctuation/spaCy/NLTK disabled by default")

        # Initialize custom dictionary (lightweight)
        try:
            self.custom_dict = CustomDictionary()
            print("✅ Custom dictionary loaded")
        except Exception as e:
            print(f"⚠️  Failed to load custom dictionary: {e}")
            self.custom_dict = None

    def process_text_ultra(self, text: str, passes: int = 6) -> str:
        """
        Ultra-enhanced text processing with multiple specialized passes.

        Args:
            text: Raw text to process
            passes: Number of processing passes (default: 6)

        Returns:
            Ultra-processed text with optimal formatting
        """
        if not text or not text.strip():
            return text

        start_time = time.time()
        original_text = text
        print(f"🚀 Starting ultra text processing with {passes} passes...")

        # Split text into chunks for parallel processing if it's large
        chunks = self._split_for_parallel_processing(text)
        processed_chunks = []

        if len(chunks) > 1 and self.use_parallel:
            print(f"📊 Processing {len(chunks)} text chunks in parallel...")
            
            # Process chunks in parallel WITH ORDER PRESERVATION
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all chunks with their index to maintain order
                future_to_index = {
                    executor.submit(self._process_chunk_multi_pass, chunk, passes): i
                    for i, chunk in enumerate(chunks)
                }
                
                # Collect results in a dictionary to maintain order
                results_dict = {}
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        processed_chunk = future.result()
                        results_dict[index] = processed_chunk
                    except Exception as e:
                        print(f"⚠️  Chunk {index} processing failed: {e}")
                        # Use original chunk as fallback
                        results_dict[index] = chunks[index]
                
                # Reconstruct chunks in proper order
                processed_chunks = [results_dict[i] for i in range(len(chunks))]
        else:
            # Process single chunk or sequential processing
            for chunk in chunks:
                processed_chunk = self._process_chunk_multi_pass(chunk, passes)
                processed_chunks.append(processed_chunk)

        # Reassemble processed text
        processed_text = self._reassemble_chunks(processed_chunks)

        # Final pass: Global coherence and quality check
        processed_text = self._final_coherence_pass(processed_text)

        # Quality assessment
        elapsed = time.time() - start_time
        self._assess_quality(original_text, processed_text, elapsed)

        # Safety check
        if len(processed_text) < len(original_text) * 0.4:
            print("⚠️  Processed text significantly shorter, using original")
            return original_text

        return processed_text

    def _split_for_parallel_processing(self, text: str) -> List[str]:
        """Split text into chunks for parallel processing."""
        # For shorter texts, don't split
        if len(text) < 2000:
            return [text]

        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        if len(paragraphs) <= 2:
            # Split by sentences if few paragraphs
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunk_size = max(1, len(sentences) // self.max_workers)
            chunks = []
            for i in range(0, len(sentences), chunk_size):
                chunk = ' '.join(sentences[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk)
            return chunks if chunks else [text]
        
        # Group paragraphs into chunks
        chunk_size = max(1, len(paragraphs) // self.max_workers)
        chunks = []
        for i in range(0, len(paragraphs), chunk_size):
            chunk = '\n\n'.join(paragraphs[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else [text]

    def _process_chunk_multi_pass(self, text: str, passes: int) -> str:
        """Process a text chunk with multiple specialized passes."""
        current_text = text

        for pass_num in range(1, passes + 1):
            try:
                if pass_num == 1:
                    # Pass 1: Basic punctuation restoration
                    current_text = self._pass_1_basic_punctuation(current_text)
                elif pass_num == 2:
                    # Pass 2: Custom dictionary word substitutions
                    current_text = self._pass_2_custom_dictionary(current_text)
                elif pass_num == 3:
                    # Pass 3: Advanced sentence segmentation
                    current_text = self._pass_3_sentence_segmentation(current_text)
                elif pass_num == 4:
                    # Pass 4: Capitalization and proper nouns
                    current_text = self._pass_4_capitalization(current_text)
                elif pass_num == 5:
                    # Pass 5: Grammar and style improvements
                    current_text = self._pass_5_grammar_style(current_text)
                elif pass_num == 6:
                    # Pass 6: Final cleanup and formatting
                    current_text = self._pass_6_final_cleanup(current_text)
                
            except Exception as e:
                print(f"⚠️  Pass {pass_num} failed: {e}")
                # Continue with current text

        return current_text

    def _pass_1_basic_punctuation(self, text: str) -> str:
        """Pass 1: Basic punctuation restoration using specialized models."""
        # Fix contraction spacing FIRST (before other punctuation fixes)
        # Handles: "let 's" -> "let's", "I 'd" -> "I'd", "ca n't" -> "can't", etc.
        contraction_fixes = [
            (r"(\w)\s+'\s*s\b", r"\1's"),      # let 's -> let's
            (r"(\w)\s+'\s*d\b", r"\1'd"),      # I 'd -> I'd  
            (r"(\w)\s+'\s*ll\b", r"\1'll"),    # I 'll -> I'll
            (r"(\w)\s+'\s*ve\b", r"\1've"),    # I 've -> I've
            (r"(\w)\s+'\s*re\b", r"\1're"),    # you 're -> you're
            (r"(\w)\s+'\s*m\b", r"\1'm"),      # I 'm -> I'm
            (r"\bca\s+n\s*'\s*t\b", "can't"),  # ca n't -> can't
            (r"\bwo\s+n\s*'\s*t\b", "won't"),  # wo n't -> won't
            (r"\bdo\s+n\s*'\s*t\b", "don't"),  # do n't -> don't
            (r"(\w)\s+-\s+(\w)", r"\1-\2"),    # co - ed -> co-ed
        ]
        for pattern, replacement in contraction_fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Basic punctuation fixes (lightweight)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)  # Ensure space after punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        return text.strip()

    def _pass_2_custom_dictionary(self, text: str) -> str:
        """Pass 2: Apply custom dictionary word substitutions."""
        if self.custom_dict:
            return self.custom_dict.apply_substitutions(text)
        return text
    
    def _pass_3_sentence_segmentation(self, text: str) -> str:
        """Pass 2: Advanced sentence segmentation and boundary detection."""
        # Regex-based sentence segmentation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not sentence[0].isupper():
                sentence = sentence[0].upper() + sentence[1:]
            processed_sentences.append(sentence)
        return ' '.join(processed_sentences)

    def _pass_4_capitalization(self, text: str) -> str:
        """Pass 3: Advanced capitalization and proper noun handling."""
        
        # Advanced proper noun patterns
        proper_nouns = {
            # Days and months
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b': lambda m: m.group(1).capitalize(),
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b': lambda m: m.group(1).capitalize(),
            
            # Countries and languages
            r'\b(america|american|england|english|france|french|germany|german|spain|spanish|italy|italian|china|chinese|japan|japanese|korea|korean|russia|russian)\b': lambda m: m.group(1).capitalize(),
            
            # Common titles and honorifics
            r'\b(doctor|dr|professor|prof|president|senator|governor|mayor|mister|mr|mrs|ms|miss)\b': lambda m: m.group(1).capitalize(),
            
            # Technology companies and brands
            r'\b(microsoft|google|apple|amazon|facebook|meta|twitter|x|youtube|instagram|tiktok|netflix|spotify|adobe|intel|nvidia|amd)\b': lambda m: m.group(1).capitalize(),
            
            # First person pronoun
            r'\bi\b': lambda m: 'I',
            r'\bi\'': lambda m: 'I\'',
        }

        for pattern, replacement in proper_nouns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Handle contractions properly
        contractions = {
            r"\bi'm\b": "I'm",
            r"\bi'll\b": "I'll", 
            r"\bi've\b": "I've",
            r"\bi'd\b": "I'd",
            r"\byou're\b": "you're",
            r"\byou'll\b": "you'll",
            r"\byou've\b": "you've",
            r"\byou'd\b": "you'd",
            r"\bhe's\b": "he's",
            r"\bshe's\b": "she's",
            r"\bit's\b": "it's",
            r"\bthat's\b": "that's",
            r"\bthere's\b": "there's",
            r"\bwhere's\b": "where's",
            r"\bwho's\b": "who's",
            r"\bwhat's\b": "what's",
            r"\bcan't\b": "can't",
            r"\bwon't\b": "won't",
            r"\bdon't\b": "don't",
            r"\bdoesn't\b": "doesn't",
            r"\bdidn't\b": "didn't",
            r"\bwouldn't\b": "wouldn't",
            r"\bshouldn't\b": "shouldn't",
            r"\bcouldn't\b": "couldn't",
        }

        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _pass_5_grammar_style(self, text: str) -> str:
        """Pass 4: Grammar improvements and style enhancements."""
        
        # Common grammar fixes
        grammar_fixes = {
            # Double negatives
            r"\bdon't\s+not\b": "don't",
            r"\bcan't\s+not\b": "can't",
            
            # Redundant words
            r"\bvery\s+very\b": "very",
            r"\breally\s+really\b": "really",
            
            # Common speech-to-text errors
            r"\bther\b": "there",
            r"\byour\s+welcome\b": "you're welcome",
            r"\bits\s+it\b": "it's",
            r"\bto\s+to\b": "to",
            r"\bthe\s+the\b": "the",
            r"\band\s+and\b": "and",
            r"\bor\s+or\b": "or",
            r"\bbut\s+but\b": "but",
            
            # Fix excessive word repetition (Whisper hallucination: "yes, yes, yes, yes...")
            # Reduce 4+ repetitions of same word to max 2
            r'\b(yes)(?:[,\s]+\1){3,}\b': r'\1, \1',
            r'\b(no)(?:[,\s]+\1){3,}\b': r'\1, \1', 
            r'\b(okay|ok)(?:[,\s]+\1){3,}\b': r'\1, \1',
            r'\b(right)(?:[,\s]+\1){3,}\b': r'\1, \1',
            r'\b(well)(?:[,\s]+\1){3,}\b': r'\1, \1',
            r'\b(so)(?:[,\s]+\1){3,}\b': r'\1, \1',
            r'\b(now)(?:[,\s]+\1){3,}\b': r'\1, \1',
            r'\b(thank you)(?:[,\s]+\1){2,}': r'\1',
            
            # Fix repeated punctuation - EXPANDED
            r'[.]{3,}': '...',
            r'[!]{2,}': '!',
            r'[?]{2,}': '?',
            r'[,]{2,}': ',',
            r'\.\s*\.': '.',  # Fix ". ." double period pattern
            r'\.\s+\.\s*': '. ',  # Fix ". . " patterns
            r'([.!?])\s*\1+': r'\1',  # Remove consecutive same punctuation,
        }

        for pattern, replacement in grammar_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Improve sentence flow
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        text = re.sub(r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)

        return text

    def _pass_6_final_cleanup(self, text: str) -> str:
        """Pass 5: Final cleanup and formatting polish."""
        
        # Final punctuation and spacing cleanup
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)  # Consistent space after punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize all whitespace
        
        # Fix quote spacing
        text = re.sub(r'"\s+', '"', text)  # No space after opening quotes
        text = re.sub(r'\s+"', ' "', text)  # Space before closing quotes
        
        # Ensure proper sentence endings
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Capitalize first letter
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        return text.strip()

    def _reassemble_chunks(self, chunks: List[str]) -> str:
        """Reassemble processed chunks into coherent text."""
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Join chunks with appropriate spacing
        assembled = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                assembled = chunk
            else:
                # Determine appropriate joining
                if assembled.endswith(('.', '!', '?')) and chunk and chunk[0].isupper():
                    assembled += " " + chunk
                elif assembled.endswith(':'):
                    assembled += " " + chunk
                else:
                    assembled += ". " + chunk
        
        return assembled

    def _final_coherence_pass(self, text: str) -> str:
        """Final pass for global coherence and flow."""
        
        # Ensure consistent paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Normalize paragraph breaks
        
        # Fix any remaining capitalization issues
        sentences = re.split(r'(?<=[.!?])\s+', text)
        processed_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Ensure each sentence starts with capital letter
                if not sentence[0].isupper():
                    sentence = sentence[0].upper() + sentence[1:]
                processed_sentences.append(sentence)
        
        text = ' '.join(processed_sentences)
        
        return text

    def _assess_quality(self, original: str, processed: str, elapsed: float):
        """Assess the quality of text processing."""
        try:
            metrics = {
                'original_length': len(original),
                'processed_length': len(processed),
                'processing_time': elapsed,
                'length_ratio': len(processed) / len(original) if original else 0,
                'sentences_original': original.count('.') + original.count('!') + original.count('?'),
                'sentences_processed': processed.count('.') + processed.count('!') + processed.count('?'),
                'avg_processing_speed': len(original) / elapsed if elapsed > 0 else 0,
            }
            
            self.quality_metrics = metrics
            
            print(f"📊 Text Processing Quality Metrics:")
            print(f"   ⏱️  Processing time: {elapsed:.2f}s")
            print(f"   📏 Length change: {metrics['original_length']} → {metrics['processed_length']} chars ({metrics['length_ratio']:.2f}x)")
            print(f"   📝 Sentences: {metrics['sentences_original']} → {metrics['sentences_processed']}")
            print(f"   🚀 Processing speed: {metrics['avg_processing_speed']:.0f} chars/sec")
            
            # Readability metrics removed to avoid heavy deps
                    
        except Exception as e:
            print(f"⚠️  Quality assessment failed: {e}")


def create_advanced_paragraph_formatter(max_workers: Optional[int] = None) -> 'AdvancedParagraphFormatter':
    """Create an advanced paragraph formatter with parallel processing."""
    return AdvancedParagraphFormatter(max_workers=max_workers)


class AdvancedParagraphFormatter:
    """Advanced paragraph formatting with intelligent segmentation."""
    
    def __init__(self, max_workers: Optional[int] = None):
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = max_workers or max(2, int(cpu_count * 0.5))
        
    def format_paragraphs_advanced(self, text: str, target_length: int = 600) -> str:
        """
        Advanced paragraph formatting with intelligent content-aware segmentation.
        
        Args:
            text: Input text to format
            target_length: Target paragraph length in characters
            
        Returns:
            Formatted text with optimized paragraph breaks
        """
        if not text or not text.strip():
            return text
            
        print(f"📝 Advanced paragraph formatting (target: {target_length} chars)...")
        
        # Split into sentences for analysis
        sentences = self._smart_sentence_split(text)
        
        if len(sentences) <= 2:
            return text  # Too short to format
            
        # Analyze sentences for semantic grouping
        sentence_groups = self._group_sentences_semantically(sentences, target_length)
        
        # Format into paragraphs
        paragraphs = []
        for group in sentence_groups:
            paragraph = ' '.join(group).strip()
            if paragraph:
                paragraphs.append(paragraph)
                
        formatted_text = '\n\n'.join(paragraphs)
        
        print(f"✅ Formatted into {len(paragraphs)} paragraphs")
        return formatted_text
    
    def _smart_sentence_split(self, text: str) -> List[str]:
        """Smart sentence splitting that handles edge cases."""
        
        # Common abbreviations that shouldn't trigger sentence breaks
        abbreviations = {
            'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'St.',
            'vs.', 'etc.', 'e.g.', 'i.e.', 'cf.', 'Co.', 'Corp.', 'Inc.', 'Ltd.',
            'U.S.', 'U.K.', 'U.N.', 'No.', 'Mt.', 'Rd.', 'Ave.', 'Blvd.',
            'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.'
        }
        
        # Initial split on sentence boundaries
        potential_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        sentences = []
        for sentence in potential_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if this ends with an abbreviation (should not be a sentence break)
            words = sentence.split()
            if words and words[-1] in abbreviations:
                # This might be a false break, try to merge with next if available
                if sentences:
                    sentences[-1] += ' ' + sentence
                else:
                    sentences.append(sentence)
            else:
                sentences.append(sentence)
                
        return sentences
    
    def _group_sentences_semantically(self, sentences: List[str], target_length: int) -> List[List[str]]:
        """Group sentences into semantically coherent paragraphs."""
        
        if len(sentences) <= 3:
            return [sentences]
            
        groups = []
        current_group = []
        current_length = 0
        
        # Keywords that suggest topic changes (new paragraph triggers)
        topic_change_indicators = {
            'however', 'but', 'on the other hand', 'in contrast', 'meanwhile',
            'furthermore', 'moreover', 'additionally', 'in addition',
            'first', 'second', 'third', 'finally', 'in conclusion',
            'now', 'then', 'next', 'after that', 'subsequently',
            'for example', 'for instance', 'specifically', 'in particular',
        }
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Check if this sentence suggests a topic change
            sentence_lower = sentence.lower()
            is_topic_change = any(indicator in sentence_lower for indicator in topic_change_indicators)
            
            # Decide whether to start a new paragraph
            should_break = (
                # Length-based break
                (current_length + sentence_length > target_length and len(current_group) >= 2) or
                # Semantic break
                (is_topic_change and len(current_group) >= 1) or
                # Force break for very long paragraphs
                (current_length > target_length * 1.5)
            )
            
            if should_break and current_group:
                groups.append(current_group)
                current_group = []
                current_length = 0
                
            current_group.append(sentence)
            current_length += sentence_length + 1  # +1 for space
            
        # Add final group
        if current_group:
            groups.append(current_group)
            
        # Merge very short groups with adjacent ones
        final_groups = []
        for group in groups:
            group_text = ' '.join(group)
            if len(group_text) < target_length * 0.3 and final_groups:
                # Merge with previous group
                final_groups[-1].extend(group)
            else:
                final_groups.append(group)
                
        return final_groups


def create_ultra_processor(max_workers: Optional[int] = None, passes: int = 5) -> UltraTextProcessor:
    """
    Factory function to create an ultra text processor.

    Args:
        max_workers: Maximum number of worker threads
        passes: Number of processing passes

    Returns:
        Configured UltraTextProcessor instance
    """
    return UltraTextProcessor(max_workers=max_workers)


if __name__ == "__main__":
    # Test the ultra processor
    test_text = """hello world this is a test of the ultra enhanced text processing system 
    it should work very well for most cases and handle multiple types of text formatting issues 
    including punctuation capitalization and paragraph formatting the system uses multiple passes 
    and parallel processing to achieve optimal results"""

    print("Original text:")
    print(repr(test_text))
    print("\n" + "="*70 + "\n")

    processor = create_ultra_processor(max_workers=4, passes=5)
    enhanced_text = processor.process_text_ultra(test_text)

    print("\nUltra-enhanced text:")
    print(repr(enhanced_text))
    
    # Test paragraph formatting
    print("\n" + "="*70 + "\n")
    formatter = create_advanced_paragraph_formatter()
    formatted_text = formatter.format_paragraphs_advanced(enhanced_text)
    
    print("Final formatted text:")
    print(formatted_text)