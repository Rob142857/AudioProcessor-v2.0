"""
Advanced Punctuation Restoration Module

Uses specialized BERT-based model (oliverguhr/fullstop-punctuation-multilang-large)
to restore proper punctuation and paragraph breaks in spoken transcripts.

This is applied AFTER Whisper transcription to refine sentence boundaries
and paragraph breaks based on semantic understanding.
"""

import re
import warnings
from typing import Optional, List
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Set environment to reduce verbose logs
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class PunctuationRestorer:
    """
    Restores punctuation in transcribed text using specialized BERT model.
    
    Uses token classification - only adds punctuation markers, NEVER changes words.
    This is safe to use as post-processing for Whisper output.
    """

    def __init__(self, model_name: str = "oliverguhr/fullstop-punctuation-multilang-large"):
        """
        Initialize the punctuation restoration model.

        Args:
            model_name: HuggingFace model name 
                       (default: oliverguhr/fullstop-punctuation-multilang-large - best accuracy)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self._initialize_model()

    def release(self):
        """Explicitly release model from memory (important for batch processing)."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("🧹 Punctuation model released from GPU memory")
            except ImportError:
                pass
        except Exception as e:
            print(f"⚠️  Error releasing punctuation model: {e}")

    def _get_local_cache_path(self, model_name: str) -> Optional[str]:
        """Find model in HuggingFace cache if available."""
        try:
            # Convert model name to cache directory format
            cache_name = model_name.replace("/", "--")
            cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{cache_name}/snapshots")
            
            if os.path.exists(cache_dir):
                # Get the first (and usually only) snapshot
                snapshots = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
                if snapshots:
                    path = os.path.join(cache_dir, snapshots[0])
                    # Verify it has the needed files
                    if os.path.exists(os.path.join(path, "config.json")):
                        return path
        except Exception:
            pass
        return None

    def _initialize_model(self):
        """Load the punctuation restoration model."""
        # First try the deepmultilingualpunctuation library (preferred)
        try:
            if "oliverguhr" in self.model_name or "fullstop" in self.model_name:
                from deepmultilingualpunctuation import PunctuationModel
                
                print(f"🔧 Loading punctuation model: {self.model_name}")
                self.model = PunctuationModel(model=self.model_name)
                self.model_type = "oliverguhr"
                print("✅ Punctuation restoration model loaded (oliverguhr - 0.6B params)")
                return
                
        except Exception as e:
            # Network/SSL error - try local cache fallback
            if "SSL" in str(e) or "offline" in str(e).lower() or "reach" in str(e).lower():
                print(f"⚠️  Network unavailable, trying local cache...")
                local_path = self._get_local_cache_path(self.model_name)
                if local_path:
                    try:
                        self._load_from_local_path(local_path)
                        return
                    except Exception as local_err:
                        print(f"❌ Local cache load failed: {local_err}")
            else:
                print(f"❌ Failed to load punctuation model: {e}")
        
        # Try kredor model as fallback
        if "kredor" in self.model_name:
            try:
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                
                print(f"🔧 Loading kredor punctuation model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                self.model_type = "kredor"
                print("✅ Punctuation restoration model loaded (kredor)")
                return
            except Exception as e:
                print(f"❌ Failed to load kredor model: {e}")
        
        # Final fallback - try local cache for oliverguhr model
        if self.model is None:
            local_path = self._get_local_cache_path("oliverguhr/fullstop-punctuation-multilang-large")
            if local_path:
                try:
                    self._load_from_local_path(local_path)
                    return
                except Exception as e:
                    print(f"❌ Local cache fallback failed: {e}")
        
        self.model = None

    def _load_from_local_path(self, local_path: str):
        """Load model directly from local cache path using transformers."""
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        print(f"🔧 Loading punctuation model from cache: {local_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.model = AutoModelForTokenClassification.from_pretrained(local_path)
        self.model_type = "transformers_local"
        print("✅ Punctuation restoration model loaded from local cache")

    def restore_punctuation(self, text: str, preserve_whisper_hints: bool = True) -> str:
        """
        Restore punctuation in text using the BERT model.

        Args:
            text: Input text (can have basic punctuation from Whisper)
            preserve_whisper_hints: If True, use Whisper's punctuation as hints

        Returns:
            Text with restored punctuation
        """
        if not self.model:
            print("⚠️  Punctuation model not available, returning original text")
            return text

        if not text or not text.strip():
            return text

        try:
            # kredor model uses different approach
            if hasattr(self, 'model_type') and self.model_type == "kredor":
                return self._restore_kredor(text, preserve_whisper_hints)
            else:
                # oliverguhr model
                if preserve_whisper_hints:
                    return self._restore_with_hints(text)
                else:
                    return self._restore_full(text)
                
        except Exception as e:
            print(f"⚠️  Punctuation restoration failed: {e}")
            return text
    
    def _restore_kredor(self, text: str, preserve_hints: bool = True) -> str:
        """Restore punctuation using kredor/punctuate-all model."""
        import torch
        
        # Remove existing punctuation if not preserving hints
        if not preserve_hints:
            text = re.sub(r'[.!?,;:—–\-]+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = text.split()
        if not tokens:
            return text
        
        # Process in chunks
        chunk_size = 256  # Token limit for kredor model
        result_text = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = ' '.join(chunk_tokens)
            
            # Tokenize for model
            inputs = self.tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Decode predictions
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_labels = [self.model.config.id2label[pred.item()] for pred in predictions[0]]
            
            # Reconstruct text with punctuation
            reconstructed = self._reconstruct_with_punctuation(predicted_tokens, predicted_labels)
            result_text.append(reconstructed)
        
        return ' '.join(result_text)
    
    def _reconstruct_with_punctuation(self, tokens: List[str], labels: List[str]) -> str:
        """Reconstruct text from tokens and punctuation labels."""
        result = []
        for token, label in zip(tokens, labels):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # Remove ## prefix from subwords
            if token.startswith('##'):
                if result:
                    result[-1] += token[2:]
            else:
                result.append(token)
            
            # Add punctuation based on label
            if label == 'PERIOD':
                result.append('.')
            elif label == 'COMMA':
                result.append(',')
            elif label == 'QUESTION':
                result.append('?')
            elif label == 'EXCLAMATION':
                result.append('!')
        
        # Clean up spacing
        text = ' '.join(result)
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation
        
        return text

    def _restore_with_hints(self, text: str) -> str:
        """
        Restore punctuation while preserving Whisper's hints.
        This is the recommended approach for refining Whisper output.
        """
        # Split into sentences/segments based on existing punctuation
        segments = re.split(r'([.!?]\s+)', text)
        
        # Recombine with separators
        original_segments = []
        for i in range(0, len(segments) - 1, 2):
            if i + 1 < len(segments):
                original_segments.append(segments[i] + segments[i + 1])
            else:
                original_segments.append(segments[i])
        if len(segments) % 2 == 1:
            original_segments.append(segments[-1])

        # Process each segment
        restored_segments = []
        for segment in original_segments:
            if segment.strip():
                # Apply model to refine punctuation within segment
                restored = self.model.restore_punctuation(segment)
                restored_segments.append(restored)

        # Rejoin
        result = ' '.join(restored_segments)
        
        # Post-process for paragraph breaks (double newlines from Whisper)
        result = self._restore_paragraph_breaks(text, result)
        
        return result

    def _restore_full(self, text: str) -> str:
        """
        Full punctuation restoration - removes existing and re-predicts.
        Use when Whisper punctuation is unreliable.
        """
        # Remove existing punctuation except apostrophes
        cleaned = re.sub(r'[.!?,;:—–\-]+', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Split into chunks if text is very long (model has token limits)
        max_tokens = 300  # Increased for better context
        words = cleaned.split()
        
        if len(words) <= max_tokens:
            return self.model.restore_punctuation(cleaned)
        
        # Process in overlapping chunks for continuity
        chunks = []
        overlap = 30  # Increased overlap for better continuity
        
        for i in range(0, len(words), max_tokens - overlap):
            chunk_words = words[i:i + max_tokens]
            chunk_text = ' '.join(chunk_words)
            
            restored_chunk = self.model.restore_punctuation(chunk_text)
            
            # Remove overlap from previous chunk's end (except first chunk)
            if i > 0 and chunks:
                # Trim overlap words from previous chunk
                prev_words = chunks[-1].split()
                if len(prev_words) > overlap:
                    chunks[-1] = ' '.join(prev_words[:-overlap])
            
            chunks.append(restored_chunk)
        
        return ' '.join(chunks)

    def _restore_paragraph_breaks(self, original: str, restored: str) -> str:
        """
        Preserve or improve paragraph breaks from original text.
        Uses double newlines as paragraph markers.
        """
        # If original had paragraph breaks, try to preserve them
        original_paragraphs = original.split('\n\n')
        
        if len(original_paragraphs) > 1:
            # Original had paragraphs - maintain rough structure
            # This is a simple heuristic; could be enhanced with semantic similarity
            
            # Calculate approximate paragraph positions
            total_chars = len(original)
            para_positions = []
            current_pos = 0
            
            for para in original_paragraphs[:-1]:  # All but last
                current_pos += len(para) + 2  # +2 for \n\n
                ratio = current_pos / total_chars
                para_positions.append(ratio)
            
            # Apply paragraph breaks to restored text at similar positions
            restored_len = len(restored)
            sentences = re.split(r'([.!?]\s+)', restored)
            
            # Recombine sentences with markers
            sentences_combined = []
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    sentences_combined.append(sentences[i] + sentences[i + 1])
                else:
                    sentences_combined.append(sentences[i])
            if len(sentences) % 2 == 1:
                sentences_combined.append(sentences[-1])
            
            # Build result with paragraph breaks at appropriate positions
            result_parts = []
            current_length = 0
            next_para_idx = 0
            
            for sentence in sentences_combined:
                current_length += len(sentence)
                result_parts.append(sentence)
                
                # Check if we should insert paragraph break
                if next_para_idx < len(para_positions):
                    current_ratio = current_length / restored_len
                    if current_ratio >= para_positions[next_para_idx]:
                        result_parts.append('\n\n')
                        next_para_idx += 1
                        continue
                
                # Otherwise just add space
                if sentence != sentences_combined[-1]:
                    result_parts.append(' ')
            
            return ''.join(result_parts)
        
        return restored

    def _fix_capitalization(self, text: str) -> str:
        """
        Fix BERT's tendency to over-capitalize mid-sentence.
        Keep capitals only after sentence-ending punctuation.
        """
        # Split into sentences
        sentences = re.split(r'([.!?]\s+)', text)
        fixed = []
        
        for i, part in enumerate(sentences):
            if i % 2 == 0:  # Actual sentence content (not punctuation)
                # First letter should be capital
                if part:
                    # Lowercase everything except first char and proper nouns
                    words = part.split()
                    if words:
                        # First word: capitalize first letter only
                        words[0] = words[0][0].upper() + words[0][1:].lower() if len(words[0]) > 1 else words[0].upper()
                        # Rest: lowercase unless it looks like a proper noun (all caps, or common proper noun patterns)
                        for j in range(1, len(words)):
                            # Keep words that are all caps (acronyms like "SP")
                            if words[j].isupper() and len(words[j]) <= 3:
                                continue
                            # Otherwise lowercase
                            words[j] = words[j].lower()
                        part = ' '.join(words)
                fixed.append(part)
            else:  # Punctuation separator
                fixed.append(part)
        
        return ''.join(fixed)

    def _fix_discourse_markers(self, text: str) -> str:
        """
        Fix common discourse markers that should start new sentences.
        
        In spoken language, words like "now", "so", "well" often introduce new thoughts
        but BERT may attach them to the previous sentence. This fixes patterns like:
        - "generated now. This is" -> "generated. Now, this is"
        - "happened so. The next" -> "happened. So, the next"
        """
        # Discourse markers that typically start new sentences when followed by certain patterns
        # Pattern: word + " now/so/well" + "." + "This/That/The/We/I/It/He/She/They"
        # Should become: word + "." + " Now/So/Well," + " this/that/the/we/i/it/he/she/they"
        
        # Fix "X now. This/That/The/It/We" -> "X. Now, this/that/the/it/we"
        text = re.sub(
            r'(\w+)\s+now\.\s+(This|That|The|It|We|I|He|She|They|What|There)\s+',
            lambda m: f"{m.group(1)}. Now, {m.group(2).lower()} ",
            text,
            flags=re.IGNORECASE
        )
        
        # Fix "X so. The/This/That/It/We" -> "X. So, the/this/that/it/we"
        text = re.sub(
            r'(\w+)\s+so\.\s+(The|This|That|It|We|I|He|She|They|What|There)\s+',
            lambda m: f"{m.group(1)}. So, {m.group(2).lower()} ",
            text,
            flags=re.IGNORECASE
        )
        
        # Fix "X well. The/This/That/It/We" -> "X. Well, the/this/that/it/we"  
        text = re.sub(
            r'(\w+)\s+well\.\s+(The|This|That|It|We|I|He|She|They|What|There)\s+',
            lambda m: f"{m.group(1)}. Well, {m.group(2).lower()} ",
            text,
            flags=re.IGNORECASE
        )
        
        # Also fix cases where there's no period yet but should be:
        # "X now this is" -> "X. Now, this is" (when "now" is clearly a discourse marker)
        text = re.sub(
            r'(\w+)\s+now\s+(this is|that is|the question|the point|the thing|what we|we have|we need|we see|I want|I think)',
            lambda m: f"{m.group(1)}. Now, {m.group(2)}",
            text,
            flags=re.IGNORECASE
        )
        
        # "X so the" at sentence boundaries often means "So, the..."
        text = re.sub(
            r'(\w+)\s+so\s+(the next|the first|the second|the question|this is|that is|what we|we have|we need)',
            lambda m: f"{m.group(1)}. So, {m.group(2)}",
            text,
            flags=re.IGNORECASE
        )
        
        return text
    
    def refine_transcription(self, transcription: str, aggressive: bool = False, chunk_size: int = 300) -> str:
        """
        Main entry point for refining Whisper transcription output.

        Args:
            transcription: Raw Whisper output text
            aggressive: If True, removes Whisper punctuation and fully re-predicts
            chunk_size: Number of words per chunk (larger = better context, slower)

        Returns:
            Refined transcription with better punctuation and paragraph breaks
        """
        print(f"🔤 Refining punctuation (aggressive={aggressive}, chunk_size={chunk_size})...")
        
        if aggressive:
            # For aggressive mode, process the whole text as one if possible
            words = transcription.split()
            if len(words) <= chunk_size:
                # Remove punctuation and let BERT re-predict everything
                cleaned = re.sub(r'[.!?,;:—–\-]+', ' ', transcription)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                result = self.model.restore_punctuation(cleaned) if self.model else transcription
            else:
                result = self._restore_full(transcription)
            
            # Fix BERT's over-capitalization
            result = self._fix_capitalization(result)
            
            # Fix discourse markers (now, so, well) that should start new sentences
            result = self._fix_discourse_markers(result)
        else:
            result = self.restore_punctuation(transcription, preserve_whisper_hints=True)
        
        print("✅ Punctuation refinement complete")
        return result


def create_punctuation_restorer(model_name: Optional[str] = None) -> PunctuationRestorer:
    """
    Factory function to create a punctuation restorer.

    Args:
        model_name: Optional custom model name

    Returns:
        Configured PunctuationRestorer instance
    """
    if model_name:
        return PunctuationRestorer(model_name=model_name)
    return PunctuationRestorer()


if __name__ == "__main__":
    # Test the punctuation restorer
    test_text = """what can be said about the difference between a system of development and a way of development
now people like yourselves who are interested in doing something about your spiritual life sooner or later
will come to contact with a group with maybe a school maybe a teacher maybe somebody who has some knowledge
of these things and its very important in the first few months to try and decide whether that person or
group or school is teaching a system or is indicating a way of development"""

    print("Original text:")
    print(test_text)
    print("\n" + "="*70 + "\n")

    restorer = create_punctuation_restorer()
    refined_text = restorer.refine_transcription(test_text, aggressive=True)

    print("\nRefined text:")
    print(refined_text)
