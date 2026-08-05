import unittest

from benchmark_transcripts import edit_counts, score, words


class BenchmarkTranscriptTests(unittest.TestCase):
    def test_word_normalization_is_case_and_punctuation_insensitive(self):
        self.assertEqual(words("Gurdjieff—said: ‘Hello.’"), ["gurdjieff", "said", "hello"])

    def test_edit_counts_identify_each_error_type(self):
        result = edit_counts(
            ["one", "two", "three", "four"],
            ["one", "too", "four", "extra"],
        )
        self.assertEqual(result["errors"], 3)
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["deletions"], 1)
        self.assertEqual(result["insertions"], 1)

    def test_score_reports_domain_term_recall(self):
        result = score(
            "Gurdjieff discussed the Enneagram.",
            "Gurdjieff discussed a diagram.",
            ["Gurdjieff", "Enneagram", "Swedenborg"],
        )
        self.assertEqual(result["term_recall"]["relevant"], 2)
        self.assertEqual(result["term_recall"]["found"], 1)
        self.assertEqual(result["term_recall"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
