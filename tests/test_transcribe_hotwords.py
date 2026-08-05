import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import transcribe_optimised as engine


def fake_encode(text):
    return re.findall(r"[\w'-]+|[^\w\s]", text, flags=re.UNICODE)


class HotwordSelectionTests(unittest.TestCase):
    def test_complete_glossary_is_loaded_without_first_100_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glossary = root / "full-glossary.txt"
            glossary.write_text(
                "\n".join(f"term-{index:04d}" for index in range(180)),
                encoding="utf-8",
            )
            source = root / "lecture.wav"
            with mock.patch.dict(
                os.environ,
                {
                    "TRANSCRIBE_AWKWARD_FILE": str(glossary),
                    "TRANSCRIBE_AWKWARD_TERMS": "",
                },
            ):
                terms = engine.load_awkward_terms(str(source))

        self.assertIn("term-0179", terms)
        self.assertGreaterEqual(len(terms), 180)

    def test_initial_prompt_never_contains_a_partial_term_or_ellipsis(self):
        prompt = engine.build_initial_prompt(
            ["alpha", "this entry is too long", "omega"],
            max_chars=12,
        )
        self.assertEqual(prompt, "alpha; omega")
        self.assertNotIn("...", prompt)

    def test_core_terms_are_reserved_and_relevant_terms_rank_first(self):
        result = engine.select_faster_whisper_hotwords(
            [
                "irrelevant phrase",
                "Perdurable Vision",
                "another phrase",
                "esotericism",
            ],
            core_terms=["esotericism", "Fourth Way"],
            context=r"1994 Prepared\The Perdurable Vision.wav",
            encode=fake_encode,
            max_tokens=7,
        )

        self.assertEqual(
            result["selected_terms"],
            ["esotericism", "Fourth Way", "Perdurable Vision"],
        )
        self.assertIn("irrelevant phrase", result["dropped_terms"])
        self.assertLessEqual(result["token_count"], result["token_budget"])
        self.assertNotIn("...", result["hotwords"])

    def test_impossible_core_budget_fails_instead_of_slicing_a_term(self):
        with self.assertRaisesRegex(ValueError, "curated special_words"):
            engine.select_faster_whisper_hotwords(
                ["one two three four"],
                core_terms=["one two three four"],
                context="",
                encode=fake_encode,
                max_tokens=3,
            )

    def test_decoder_control_tokens_and_nontext_controls_are_rejected(self):
        cleaned = engine._clean_awkward_terms(
            ["esotericism", "<|startoftranscript|>", "unsafe\x01term"]
        )

        self.assertEqual(cleaned, ["esotericism"])

    def test_hotwords_take_precedence_over_a_separate_initial_prompt(self):
        self.assertFalse(
            engine.should_apply_initial_prompt(
                prompt="MW; Lecture title",
                allow_prompt=True,
                using_faster_whisper=True,
                hotwords="esotericism; Fourth Way",
            )
        )
        self.assertTrue(
            engine.should_apply_initial_prompt(
                prompt="MW; Lecture title",
                allow_prompt=True,
                using_faster_whisper=False,
                hotwords=None,
            )
        )

    def test_loaded_model_tokenizer_sets_the_exact_installed_budget(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                self.add_special_tokens = add_special_tokens
                return SimpleNamespace(ids=fake_encode(text))

        tokenizer = FakeTokenizer()
        model = SimpleNamespace(hf_tokenizer=tokenizer, max_length=16)
        result = engine.build_faster_whisper_hotwords(
            model,
            ["Perdurable Vision"],
            core_terms=["esotericism"],
            context="Perdurable Vision",
        )

        self.assertEqual(result["token_budget"], 7)
        self.assertFalse(tokenizer.add_special_tokens)
        self.assertIn("esotericism", result["selected_terms"])


class FasterWhisperResultTests(unittest.TestCase):
    def tearDown(self):
        engine.clear_stop()

    def test_result_preserves_segment_and_word_confidence(self):
        word = SimpleNamespace(
            start=1.25,
            end=1.75,
            word=" figure",
            probability=0.42,
        )
        segment = SimpleNamespace(
            id=7,
            seek=160,
            start=1.0,
            end=2.0,
            text=" figure",
            tokens=[101, 202],
            avg_logprob=-0.73,
            compression_ratio=1.18,
            no_speech_prob=0.08,
            temperature=0.0,
            words=[word],
        )
        info = SimpleNamespace(
            language="en",
            language_probability=0.99,
            duration=20.0,
            duration_after_vad=18.0,
            all_language_probs=[("en", 0.99)],
        )

        result = engine._as_result_dict((iter([segment]), info))
        output = result["segments"][0]

        self.assertEqual(output["id"], 7)
        self.assertEqual(output["seek"], 160)
        self.assertEqual(output["tokens"], [101, 202])
        self.assertEqual(output["avg_logprob"], -0.73)
        self.assertEqual(output["compression_ratio"], 1.18)
        self.assertEqual(output["no_speech_prob"], 0.08)
        self.assertEqual(output["temperature"], 0.0)
        self.assertEqual(
            output["words"],
            [
                {
                    "start": 1.25,
                    "end": 1.75,
                    "word": " figure",
                    "probability": 0.42,
                }
            ],
        )
        self.assertEqual(result["language_probability"], 0.99)

    def test_lazy_segment_failure_never_returns_a_partial_transcript(self):
        first = SimpleNamespace(
            start=0.0,
            end=1.0,
            text="This partial sentence must not be published.",
            words=[],
        )

        def failing_segments():
            yield first
            raise OSError("decoder stream failed")

        with self.assertRaisesRegex(
            RuntimeError, "segment iteration failed before completion"
        ):
            engine._as_result_dict(
                (failing_segments(), SimpleNamespace(language="en"))
            )

    def test_preprocessing_padding_is_removed_from_segment_and_word_times(self):
        original = {
            "text": "first last",
            "segments": [
                {
                    "text": "first last",
                    "start": 1.2,
                    "end": 13.5,
                    "words": [
                        {"word": " first", "start": 1.2, "end": 1.8},
                        {"word": " last", "start": 12.9, "end": 13.5},
                    ],
                }
            ],
        }

        shifted = engine._shift_transcript_timestamps(
            original,
            lead_seconds=1.5,
            source_duration_seconds=12.0,
        )

        self.assertEqual(shifted["segments"][0]["start"], 0.0)
        self.assertEqual(shifted["segments"][0]["end"], 12.0)
        self.assertEqual(shifted["segments"][0]["words"][0]["start"], 0.0)
        self.assertAlmostEqual(
            shifted["segments"][0]["words"][0]["end"], 0.3
        )
        self.assertAlmostEqual(
            shifted["segments"][0]["words"][1]["start"], 11.4
        )
        self.assertEqual(shifted["segments"][0]["words"][1]["end"], 12.0)
        self.assertEqual(original["segments"][0]["start"], 1.2)

    def test_hotwords_remove_prefix_at_the_compatibility_boundary(self):
        class FakeModel:
            def __init__(self):
                self.received = None

            def transcribe(
                self,
                audio,
                *,
                hotwords=None,
                prefix=None,
                initial_prompt=None,
            ):
                self.received = {
                    "audio": audio,
                    "hotwords": hotwords,
                    "prefix": prefix,
                    "initial_prompt": initial_prompt,
                }
                return "ok"

        model = FakeModel()
        result = engine._compatible_transcribe_call(
            model,
            "lecture.wav",
            {
                "hotwords": "esotericism; Fourth Way",
                "prefix": "must not be sent",
                "initial_prompt": "MW; Lecture title",
            },
        )

        self.assertEqual(result, "ok")
        self.assertEqual(model.received["hotwords"], "esotericism; Fourth Way")
        self.assertIsNone(model.received["prefix"])
        self.assertEqual(model.received["initial_prompt"], "MW; Lecture title")


if __name__ == "__main__":
    unittest.main()
