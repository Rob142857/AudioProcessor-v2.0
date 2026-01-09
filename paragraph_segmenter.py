"""
Semantic Paragraph Segmentation Module

Uses sentence-transformers to detect topic shifts and intelligently
insert paragraph breaks based on semantic similarity between sentences.

This runs AFTER punctuation restoration and BEFORE final formatting.
"""

import re
import warnings
from typing import List, Tuple, Optional
import os

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class SemanticParagraphSegmenter:
    """
    Detects topic shifts and inserts paragraph breaks based on semantic understanding.
    """

    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.50):
        """
        Initialize the semantic paragraph segmenter.

        Args:
            model_name: SentenceTransformer model to use (default: all-MiniLM-L6-v2, ~80MB)
            similarity_threshold: Cosine similarity threshold for paragraph breaks (0-1)
                                Lower = more breaks, Higher = fewer breaks
                                Recommended: 0.45-0.55 for lectures (tuned for topic shifts)
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Load the sentence transformer model with offline fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"🔧 Loading semantic paragraph model: {self.model_name}")
            
            # Try offline mode first if model is cached
            try:
                import os
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                self.model = SentenceTransformer(self.model_name)
                print("✅ Semantic paragraph segmenter ready (from cache)")
            except Exception:
                # Fallback to online mode
                if "HF_HUB_OFFLINE" in os.environ:
                    del os.environ["HF_HUB_OFFLINE"]
                if "TRANSFORMERS_OFFLINE" in os.environ:
                    del os.environ["TRANSFORMERS_OFFLINE"]
                self.model = SentenceTransformer(self.model_name)
                print("✅ Semantic paragraph segmenter ready (downloaded)")
            
        except ImportError:
            print("⚠️  sentence-transformers not installed. Installing...")
            try:
                import subprocess
                import sys
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "sentence-transformers", "--quiet"
                ])
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                print("✅ Semantic paragraph segmenter installed and ready")
            except Exception as e:
                print(f"❌ Failed to install/load semantic model: {e}")
                self.model = None
                
        except Exception as e:
            print(f"❌ Failed to load semantic model: {e}")
            self.model = None

    def release(self):
        """Explicitly release model from memory (important for batch processing)."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("🧹 Paragraph model released from GPU memory")
            except ImportError:
                pass
        except Exception as e:
            print(f"⚠️  Error releasing paragraph model: {e}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, preserving existing structure.
        """
        # Split on sentence endings followed by space and capital letter
        # This preserves the punctuation restoration work
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

    def _calculate_similarity(self, sent1: str, sent2: str) -> float:
        """
        Calculate cosine similarity between two sentences.
        
        Returns:
            Float between 0 (completely different) and 1 (identical)
        """
        if not self.model:
            return 1.0  # If model unavailable, assume high similarity (no break)
        
        try:
            from sentence_transformers import util
            
            embeddings = self.model.encode([sent1, sent2], convert_to_tensor=False)
            similarity = util.cos_sim(embeddings[0], embeddings[1])
            
            return float(similarity[0][0])
        except Exception as e:
            print(f"⚠️  Similarity calculation failed: {e}")
            return 1.0

    def segment_paragraphs(self, text: str, verbose: bool = False) -> str:
        """
        Add paragraph breaks based on semantic topic shifts.

        Args:
            text: Input text with proper punctuation
            verbose: Print debugging info about breaks

        Returns:
            Text with paragraph breaks inserted at topic shifts
        """
        if not self.model:
            print("⚠️  Semantic model not available, returning text unchanged")
            return text

        if not text or not text.strip():
            return text

        print(f"🔍 Analyzing semantic structure (threshold={self.similarity_threshold})...")

        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) < 2:
            return text

        # Calculate similarities between consecutive sentences
        segments = []
        current_paragraph = [sentences[0]]
        breaks_inserted = 0

        for i in range(1, len(sentences)):
            prev_sent = sentences[i-1]
            curr_sent = sentences[i]
            
            # Calculate semantic similarity
            similarity = self._calculate_similarity(prev_sent, curr_sent)
            
            # Detect paragraph break
            should_break = similarity < self.similarity_threshold
            
            # Additional heuristics for better detection
            # 1. Strong transition words signal topic shift
            starts_with_transition = curr_sent.startswith(('Now,', 'Now ', 'So ', 
                                                           'Well,', 'Anyway,'))
            
            # 2. Avoid breaking on simple examples/lists
            is_simple_example = (prev_sent.lower().find('for example') >= 0 or
                                curr_sent.lower().startswith(('for example', 'for instance')))
            
            # 3. Very short sentences (< 20 chars) are usually continuations
            too_short = len(curr_sent) < 20 or len(prev_sent) < 20
            
            # Combine signals - be conservative
            if should_break and not is_simple_example and not too_short:
                # Insert paragraph break only for clear topic shifts
                segments.append(' '.join(current_paragraph))
                current_paragraph = [curr_sent]
                breaks_inserted += 1
                
                if verbose:
                    print(f"  Break {breaks_inserted}: similarity={similarity:.3f}")
                    print(f"    Before: ...{prev_sent[-50:]}")
                    print(f"    After:  {curr_sent[:50]}...")
            elif starts_with_transition and similarity < 0.60 and not too_short:
                # Insert paragraph break
                segments.append(' '.join(current_paragraph))
                current_paragraph = [curr_sent]
                breaks_inserted += 1
                
                if verbose:
                    print(f"  Break {breaks_inserted}: similarity={similarity:.3f}")
                    print(f"    Before: ...{prev_sent[-50:]}")
                    print(f"    After:  {curr_sent[:50]}...")
            else:
                # Continue current paragraph
                current_paragraph.append(curr_sent)

        # Add final paragraph
        if current_paragraph:
            segments.append(' '.join(current_paragraph))

        # Join with double newlines (paragraph breaks)
        result = '\n\n'.join(segments)
        
        print(f"✅ Inserted {breaks_inserted} semantic paragraph breaks")
        
        return result

    def refine_paragraphs(self, text: str, 
                         min_paragraph_length: int = 100,
                         max_paragraph_length: int = 800,
                         min_sentences: int = 2) -> str:
        """
        Post-process paragraphs to ensure reasonable lengths.
        Uses sentence count as primary metric to avoid losing semantic breaks.

        Args:
            text: Text with paragraph breaks
            min_paragraph_length: Merge paragraphs shorter than this (chars) - secondary check
            max_paragraph_length: Split paragraphs longer than this (chars)
            min_sentences: Minimum sentences per paragraph (default: 2)

        Returns:
            Text with refined paragraph lengths
        """
        paragraphs = text.split('\n\n')
        refined = []
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            sentences = self._split_into_sentences(para)
            sentence_count = len(sentences)
            char_count = len(para)
            
            # TRANSITIONAL SENTENCES: If a paragraph is a single sentence that
            # starts with a transition word, merge it with the PREVIOUS paragraph
            # (it signals a topic shift but belongs with what came before)
            transition_starters = ('Now,', 'Now ', 'So,', 'So ', 'Well,', 'Anyway,',
                                   'Now let', 'Let us', "Let's", 'Moving on', 
                                   'Turning to', 'Next,', 'Finally,')
            is_transitional = (sentence_count == 1 and 
                              any(para.startswith(t) for t in transition_starters))
            
            if is_transitional and refined:
                # Merge transitional sentence with PREVIOUS paragraph
                refined[-1] = refined[-1] + ' ' + para
            # If very short (1 sentence AND < min chars), merge with previous
            elif sentence_count < min_sentences and char_count < min_paragraph_length and refined:
                refined[-1] = refined[-1] + ' ' + para
            # If too long, split at sentence boundaries
            elif char_count > max_paragraph_length:
                temp_para = []
                current_length = 0
                
                for sent in sentences:
                    if current_length + len(sent) > max_paragraph_length and temp_para:
                        refined.append(' '.join(temp_para))
                        temp_para = [sent]
                        current_length = len(sent)
                    else:
                        temp_para.append(sent)
                        current_length += len(sent)
                
                if temp_para:
                    refined.append(' '.join(temp_para))
            else:
                refined.append(para)
        
        return '\n\n'.join(refined)


def create_paragraph_segmenter(similarity_threshold: float = 0.45) -> SemanticParagraphSegmenter:
    """
    Factory function to create a semantic paragraph segmenter.

    Args:
        similarity_threshold: Lower = more breaks (0.40-0.50 recommended for lectures)
                             0.45 is tuned for natural topic shifts in spoken content

    Returns:
        Configured SemanticParagraphSegmenter instance
    """
    return SemanticParagraphSegmenter(similarity_threshold=similarity_threshold)


if __name__ == "__main__":
    # Test the semantic segmenter
    test_text = """Last week we made an initial look at the role of colour in life. We pointed out that various colourful molecules fill our world. They touch us and affect us. Colour is a radiation of various wavelengths into the environment. Now once we were looking at colours last week they were confined mainly to plants. What we forget is in the natural physical world a lot of substances are brightly coloured. A lot of the chemical elements are coloured. For example chlorine has a green colour. Copper is a sort of brownish red colour. Gold is yellow. Most of the metals are silvery. But we notice that manganese not only is silvery it has a reddish tinge."""

    print("Original text:")
    print(test_text)
    print("\n" + "="*70 + "\n")

    segmenter = create_paragraph_segmenter(similarity_threshold=0.65)
    segmented_text = segmenter.segment_paragraphs(test_text, verbose=True)

    print("\nSegmented text:")
    print(segmented_text)
