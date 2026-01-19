"""
Tests for new strategies added in v0.3.0:
- MarkovChainStrategy
- NGramFrequencyStrategy
- WordLookupStrategy
"""

import pytest
from pygarble import GarbleDetector, Strategy


class TestMarkovChainStrategy:
    """Tests for Markov chain based detection."""

    def test_valid_english_text(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        assert detector.predict("hello world") is False
        assert detector.predict("the quick brown fox") is False
        assert detector.predict("natural language processing") is False

    def test_keyboard_mashing(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        assert detector.predict("asdfghjkl") is True
        assert detector.predict("qwertyuiop") is True
        assert detector.predict("zxcvbnm") is True

    def test_random_gibberish(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        assert detector.predict("xzqkjhf") is True
        assert detector.predict("bvnmxzq") is True

    def test_repeated_characters(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        assert detector.predict("aaaaaaaaa") is True

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        for text in ["hello", "asdfgh", "the cat sat"]:
            proba = detector.predict_proba(text)
            assert 0.0 <= proba <= 1.0

    def test_valid_text_low_probability(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        proba = detector.predict_proba("hello world")
        assert proba < 0.5

    def test_gibberish_high_probability(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        proba = detector.predict_proba("xzqkjhf")
        assert proba > 0.5

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_custom_threshold(self):
        # More permissive threshold
        detector = GarbleDetector(
            Strategy.MARKOV_CHAIN,
            threshold_per_char=-6.0
        )
        # Should be more lenient
        proba = detector.predict_proba("qwerty")
        assert proba < 0.9  # Should be lower with more permissive threshold


class TestNGramFrequencyStrategy:
    """Tests for n-gram frequency based detection."""

    def test_valid_english_text(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        assert detector.predict("hello world") is False
        assert detector.predict("the quick brown fox") is False
        assert detector.predict("programming language") is False

    def test_random_gibberish(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        assert detector.predict("xzqkjhf") is True
        assert detector.predict("bvnmxzq") is True

    def test_keyboard_mashing(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        assert detector.predict("asdfghjkl") is True

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        for text in ["hello", "xyzzy", "the cat"]:
            proba = detector.predict_proba(text)
            assert 0.0 <= proba <= 1.0

    def test_valid_text_low_probability(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        proba = detector.predict_proba("the quick brown fox")
        assert proba < 0.5

    def test_gibberish_high_probability(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        proba = detector.predict_proba("xzqkjhf")
        assert proba > 0.5

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_short_text(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        # Very short text should not be flagged
        assert detector.predict("hi") is False

    def test_custom_threshold(self):
        # Stricter threshold
        detector = GarbleDetector(
            Strategy.NGRAM_FREQUENCY,
            common_ratio_threshold=0.5
        )
        # Should be stricter about what's considered valid
        proba = detector.predict_proba("hello world")
        assert proba <= 0.5


class TestWordLookupStrategy:
    """Tests for dictionary-based word lookup detection."""

    def test_valid_english_words(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        assert detector.predict("hello world") is False
        assert detector.predict("the quick brown fox") is False
        assert detector.predict("computer science") is False

    def test_unknown_words(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        assert detector.predict("xyzzy plugh") is True
        assert detector.predict("asdfgh qwerty") is True

    def test_mixed_valid_invalid(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        # Half valid, half invalid - should be borderline
        proba = detector.predict_proba("hello xyzzy")
        assert 0.4 <= proba <= 0.6

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        for text in ["hello", "xyzzy", "the cat"]:
            proba = detector.predict_proba(text)
            assert 0.0 <= proba <= 1.0

    def test_all_valid_zero_probability(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        proba = detector.predict_proba("hello world")
        assert proba == 0.0

    def test_all_invalid_full_probability(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        proba = detector.predict_proba("xyzzy plugh")
        assert proba == 1.0

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_numbers_only(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        # No words to check
        assert detector.predict("12345") is False

    def test_case_insensitive(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        # Should work regardless of case
        assert detector.predict("HELLO WORLD") is False
        assert detector.predict("HeLLo WoRLd") is False


class TestBatchProcessing:
    """Test batch processing with new strategies."""

    def test_markov_chain_batch(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        texts = ["hello world", "asdfghjkl", "the cat"]
        results = detector.predict(texts)
        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0] is False  # hello world
        assert results[1] is True   # asdfghjkl
        assert results[2] is False  # the cat

    def test_ngram_frequency_batch(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        texts = ["hello world", "xzqkjhf", "the cat"]
        results = detector.predict(texts)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_word_lookup_batch(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        texts = ["hello", "xyzzy", "world"]
        results = detector.predict(texts)
        assert results == [False, True, False]


class TestDataModule:
    """Test that embedded data loads correctly."""

    def test_bigram_data_loaded(self):
        from pygarble.data import BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB
        assert isinstance(BIGRAM_LOG_PROBS, dict)
        assert len(BIGRAM_LOG_PROBS) == 729  # 27 * 27
        assert DEFAULT_LOG_PROB == -10.0

    def test_trigram_data_loaded(self):
        from pygarble.data import COMMON_TRIGRAMS
        assert isinstance(COMMON_TRIGRAMS, frozenset)
        assert len(COMMON_TRIGRAMS) == 2000
        assert "the" in COMMON_TRIGRAMS
        assert "ing" in COMMON_TRIGRAMS

    def test_words_data_loaded(self):
        from pygarble.data import ENGLISH_WORDS
        assert isinstance(ENGLISH_WORDS, frozenset)
        assert len(ENGLISH_WORDS) == 50000
        assert "hello" in ENGLISH_WORDS
        assert "world" in ENGLISH_WORDS
        assert "the" in ENGLISH_WORDS


class TestNoDependencies:
    """Test that core strategies work without optional dependencies."""

    def test_markov_chain_no_deps(self):
        # Should work without any external dependencies
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        assert detector.predict("hello") is False

    def test_ngram_frequency_no_deps(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        assert detector.predict("hello") is False

    def test_word_lookup_no_deps(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        assert detector.predict("hello") is False

    def test_keyboard_pattern_no_deps(self):
        detector = GarbleDetector(Strategy.KEYBOARD_PATTERN)
        # KEYBOARD_PATTERN detects keyboard row sequences
        # It correctly identifies keyboard mashing
        assert detector.predict("asdfghjkl") is True

    def test_entropy_based_no_deps(self):
        detector = GarbleDetector(Strategy.ENTROPY_BASED)
        assert detector.predict("hello") is False

    def test_vowel_ratio_no_deps(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        assert detector.predict("hello") is False
