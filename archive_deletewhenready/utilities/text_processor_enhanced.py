"""
Enhanced Text Processing Module for AudioProcessor Alpha Version

This module provides improved punctuation and capitalization for transcription output.
It combines multiple approaches for optimal text quality:

1. Deep Multilingual Punctuation for basic punctuation restoration
2. SpaCy for advanced sentence segmentation and part-of-speech tagging
3. Custom capitalization rules for proper nouns and edge cases
4. Multi-pass processing for iterative improvements

Author: AudioProcessor Team
Date: September 2025
"""

import re
import os
import sys
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set offline mode for Hugging Face to prevent network checks
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Optional imports with fallbacks
try:
    from deepmultilingualpunctuation import PunctuationModel
    _punctuation_available = True
except ImportError:
    PunctuationModel = None
    _punctuation_available = False

try:
    import spacy
    from spacy.lang.en import English
    _spacy_available = True
except ImportError:
    spacy = None
    English = None
    _spacy_available = False

try:
    import transformers
    from transformers import pipeline
    _transformers_available = True
except ImportError:
    transformers = None
    pipeline = None
    _transformers_available = False


class EnhancedTextProcessor:
    """
    Enhanced text processor with multiple passes for optimal punctuation and capitalization.
    """

    def __init__(self, use_spacy: bool = True, use_transformers: bool = False):
        """
        Initialize the enhanced text processor.

        Args:
            use_spacy: Whether to use spaCy for advanced processing
            use_transformers: Whether to use transformer models (slower but potentially better)
        """
        self.use_spacy = use_spacy and _spacy_available
        self.use_transformers = use_transformers and _transformers_available

        # Initialize models
        self.punctuation_model = None
        self.nlp = None
        self.capitalization_model = None

        self._initialize_models()

    def _initialize_models(self):
        """Initialize available models based on what's installed."""
        # Initialize punctuation model
        if _punctuation_available:
            try:
                self.punctuation_model = PunctuationModel()
                print("✅ Deep Multilingual Punctuation model loaded")
            except Exception as e:
                print(f"⚠️  Failed to load punctuation model: {e}")

        # Initialize spaCy
        if self.use_spacy and _spacy_available:
            try:
                # Try to load English model, download if needed
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("📥 Downloading spaCy English model...")
                    os.system("python -m spacy download en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                print("✅ SpaCy English model loaded")
            except Exception as e:
                print(f"⚠️  Failed to load spaCy model: {e}")
                self.use_spacy = False

        # Initialize transformer model for capitalization (optional)
        if self.use_transformers and _transformers_available:
            try:
                self.capitalization_model = pipeline(
                    "token-classification",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                print("✅ Transformer capitalization model loaded")
            except Exception as e:
                print(f"⚠️  Failed to load transformer model: {e}")
                self.use_transformers = False

    def restore_punctuation(self, text: str) -> str:
        """
        Enhanced punctuation restoration with multiple passes.

        Args:
            text: Raw text without punctuation

        Returns:
            Text with improved punctuation and capitalization
        """
        if not text or not text.strip():
            return text

        original_text = text
        print("🔄 Starting enhanced punctuation restoration...")

        # Pass 1: Basic punctuation restoration
        if self.punctuation_model:
            try:
                text = self.punctuation_model.restore_punctuation(text)
                print("✅ Pass 1: Basic punctuation restored")
            except Exception as e:
                print(f"⚠️  Pass 1 failed: {e}")

        # Pass 2: SpaCy-based sentence segmentation and capitalization
        if self.use_spacy and self.nlp:
            try:
                text = self._spacy_enhance_text(text)
                print("✅ Pass 2: SpaCy enhancement completed")
            except Exception as e:
                print(f"⚠️  Pass 2 failed: {e}")

        # Pass 3: Custom capitalization rules
        try:
            text = self._apply_custom_capitalization_rules(text)
            print("✅ Pass 3: Custom capitalization rules applied")
        except Exception as e:
            print(f"⚠️  Pass 3 failed: {e}")

        # Pass 4: Final cleanup and edge case handling
        try:
            text = self._final_cleanup(text)
            print("✅ Pass 4: Final cleanup completed")
        except Exception as e:
            print(f"⚠️  Pass 4 failed: {e}")

        # Quality check
        if len(text) < len(original_text) * 0.5:
            print("⚠️  Text length significantly reduced, using original")
            return original_text

        return text

    def _spacy_enhance_text(self, text: str) -> str:
        """Use spaCy for advanced sentence segmentation and POS-based capitalization."""
        if not self.nlp:
            return text

        # Process text with spaCy
        doc = self.nlp(text)

        # Build enhanced text
        enhanced_sentences = []

        for sent in doc.sents:
            sentence_text = sent.text.strip()

            # Apply POS-based capitalization rules
            words = []
            for token in sent:
                word = token.text

                # Capitalize based on POS and context
                if self._should_capitalize_token(token, sent):
                    word = word.capitalize()
                elif token.pos_ in ['PROPN', 'NOUN'] and len(word) > 3:
                    # Keep proper nouns capitalized
                    pass
                else:
                    word = word.lower()

                words.append(word)

            enhanced_sentence = ' '.join(words)

            # Ensure sentence starts with capital letter
            if enhanced_sentence and not enhanced_sentence[0].isupper():
                enhanced_sentence = enhanced_sentence[0].upper() + enhanced_sentence[1:]

            enhanced_sentences.append(enhanced_sentence)

        return ' '.join(enhanced_sentences)

    def _should_capitalize_token(self, token, sentence) -> bool:
        """Determine if a token should be capitalized based on context."""
        # Always capitalize first word of sentence
        if token.i == sentence.start:
            return True

        # Capitalize after sentence-ending punctuation
        if token.i > sentence.start:
            prev_token = sentence.doc[token.i - 1]
            if prev_token.text in ['.', '!', '?', ':']:
                return True

        # Capitalize I (first person singular)
        if token.text.lower() == 'i':
            return True

        # Capitalize proper nouns (spaCy should handle this, but reinforce)
        if token.pos_ == 'PROPN':
            return True

        # Capitalize after quotes in dialogue
        if token.i > sentence.start:
            prev_token = sentence.doc[token.i - 1]
            if prev_token.text in ['"', '"', ''', ''']:
                return True

        return False

    def _apply_custom_capitalization_rules(self, text: str) -> str:
        """Apply custom capitalization rules for edge cases."""

        # Common proper nouns and titles that should be capitalized
        proper_nouns = [
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'america', 'american', 'english', 'french', 'german', 'spanish', 'italian',
            'chinese', 'japanese', 'korean', 'russian', 'arabic', 'hindi',
            'doctor', 'professor', 'president', 'senator', 'governor', 'mayor',
            'microsoft', 'google', 'apple', 'amazon', 'facebook', 'twitter',
            'youtube', 'instagram', 'tiktok', 'netflix', 'spotify'
        ]

        # Replace proper nouns
        for noun in proper_nouns:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(noun) + r'\b'
            replacement = noun.capitalize()
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Handle contractions and possessives
        contractions = {
            r"\bi\b": "I",  # Standalone I
            r"\bi'": "I'",  # I contractions
            r"\bi'm\b": "I'm",
            r"\bi'll\b": "I'll",
            r"\bi've\b": "I've",
            r"\bi'd\b": "I'd",
        }

        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Capitalize after colons in titles or formal contexts
        text = re.sub(r'(:)\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)

        # Fix double spaces that might have been introduced
        text = re.sub(r'\s+', ' ', text)

        return text

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup and edge case handling."""

        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)  # Ensure space after punctuation

        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Ensure proper spacing around quotes
        text = re.sub(r'"\s+', '"', text)  # Remove space after opening quote
        text = re.sub(r'\s+"', '" ', text)  # Ensure space before closing quote

        # Capitalize first letter after quotes
        text = re.sub(r'[""]\s*([a-z])', lambda m: '"' + m.group(1).upper(), text)

        # Fix common OCR/transcription errors
        corrections = {
            r'\bim\b': 'I\'m',
            r'\bid\b': 'I\'d',
            r'\bive\b': 'I\'ve',
            r'\bill\b': 'I\'ll',
            r'\bits\b': 'it\'s',
            r'\bthats\b': 'that\'s',
            r'\btheres\b': 'there\'s',
            r'\bwheres\b': 'where\'s',
            r'\bwhos\b': 'who\'s',
            r'\bwhats\b': 'what\'s',
        }

        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()


def create_enhanced_processor(use_spacy: bool = True, use_transformers: bool = False) -> EnhancedTextProcessor:
    """
    Factory function to create an enhanced text processor.

    Args:
        use_spacy: Whether to use spaCy for advanced processing
        use_transformers: Whether to use transformer models

    Returns:
        Configured EnhancedTextProcessor instance
    """
    return EnhancedTextProcessor(use_spacy=use_spacy, use_transformers=use_transformers)


# Backwards compatibility function
def restore_punctuation_enhanced(text: str) -> str:
    """
    Enhanced punctuation restoration with backwards compatibility.

    Args:
        text: Raw text to process

    Returns:
        Processed text with improved punctuation and capitalization
    """
    processor = create_enhanced_processor()
    return processor.restore_punctuation(text)


if __name__ == "__main__":
    # Test the enhanced processor
    test_text = "hello world this is a test of the enhanced punctuation system it should work well for most cases"

    print("Original text:")
    print(test_text)
    print("\n" + "="*50 + "\n")

    processor = create_enhanced_processor()
    enhanced_text = processor.restore_punctuation(test_text)

    print("Enhanced text:")
    print(enhanced_text)