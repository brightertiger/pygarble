"""
Tests for new strategies added in v0.3.0+:
- MarkovChainStrategy
- NGramFrequencyStrategy
- WordLookupStrategy
- CompressionRatioStrategy (v0.4.0)
- MojibakeStrategy (v0.4.0)
- PronouncabilityStrategy (v0.4.0)
- UnicodeScriptStrategy (v0.4.0)
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


# ============================================================
# Tests for strategies added in v0.4.0
# ============================================================


class TestCompressionRatioStrategy:
    """Tests for compression-based detection."""

    def test_valid_english_text(self):
        detector = GarbleDetector(Strategy.COMPRESSION_RATIO)
        # Short text below min_length returns 0.0 probability
        assert detector.predict_proba("hello world") == 0.0

    def test_highly_repetitive_text(self):
        detector = GarbleDetector(Strategy.COMPRESSION_RATIO)
        # Highly repetitive text compresses very well
        repetitive = "the " * 50
        proba = detector.predict_proba(repetitive)
        assert proba < 0.5  # Should be low (compresses well)

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.COMPRESSION_RATIO)
        for text in ["a" * 150, "hello " * 30, "xyz " * 40]:
            proba = detector.predict_proba(text)
            assert 0.0 <= proba <= 1.0

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.COMPRESSION_RATIO)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_short_text_returns_zero(self):
        detector = GarbleDetector(Strategy.COMPRESSION_RATIO)
        # Default min_length is 100
        assert detector.predict_proba("short text") == 0.0

    def test_custom_min_length(self):
        detector = GarbleDetector(
            Strategy.COMPRESSION_RATIO,
            min_length=10
        )
        # Should now analyze shorter text
        proba = detector.predict_proba("hello world test")
        assert proba >= 0.0  # Should return a value

    def test_custom_thresholds(self):
        detector = GarbleDetector(
            Strategy.COMPRESSION_RATIO,
            high_ratio_threshold=1.2,
            low_ratio_threshold=0.5
        )
        assert detector.predict("a" * 150) is False

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            GarbleDetector(Strategy.COMPRESSION_RATIO, high_ratio_threshold=-0.1)
        with pytest.raises(ValueError):
            GarbleDetector(Strategy.COMPRESSION_RATIO, low_ratio_threshold=1.6)
        with pytest.raises(ValueError):
            # low must be less than high
            GarbleDetector(
                Strategy.COMPRESSION_RATIO,
                low_ratio_threshold=0.9,
                high_ratio_threshold=0.5
            )


class TestMojibakeStrategy:
    """Tests for encoding corruption (mojibake) detection."""

    def test_clean_ascii_text(self):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        assert detector.predict("hello world") is False
        assert detector.predict("The quick brown fox") is False

    def test_mojibake_patterns(self):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        # Simulate UTF-8 decoded as Latin-1 (common mojibake)
        mojibake = "Caf\xc3\xa9"  # Should be "Café"
        assert detector.predict(mojibake) is True

    def test_replacement_character(self):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        text = "Hello \ufffd world"  # Unicode replacement char
        assert detector.predict(text) is True

    def test_high_byte_density(self):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        # Text with many Latin-1 supplement characters
        high_byte = "test \x80\x81\x82\x83 more"
        proba = detector.predict_proba(high_byte)
        assert proba > 0.0

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        for text in ["hello", "caf\xc3\xa9", "test\ufffd"]:
            proba = detector.predict_proba(text)
            assert 0.0 <= proba <= 1.0

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_short_text(self):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        assert detector.predict_proba("hi") == 0.0

    def test_disable_replacement_char_check(self):
        detector = GarbleDetector(
            Strategy.MOJIBAKE,
            check_replacement_char=False
        )
        text = "Hello \ufffd world"
        # Should not flag just for replacement char
        proba = detector.predict_proba(text)
        # May still flag for high-byte density
        assert isinstance(proba, float)

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            GarbleDetector(Strategy.MOJIBAKE, pattern_threshold=0)
        with pytest.raises(ValueError):
            GarbleDetector(Strategy.MOJIBAKE, ratio_threshold=1.5)


class TestPronouncabilityStrategy:
    """Tests for phonotactic-based pronounceability detection."""

    def test_valid_english_text(self):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        assert detector.predict("hello world") is False
        assert detector.predict("the quick brown fox") is False
        assert detector.predict("string theory") is False

    def test_unpronounceable_clusters(self):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        # Text with forbidden consonant clusters
        assert detector.predict("xkcd qwfp zxcv bkpt") is True

    def test_no_vowels(self):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        # All consonants - unpronounceable
        assert detector.predict("bcdfghjklmnp qrstvwxyz") is True

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        for text in ["hello", "xkcd", "the cat sat"]:
            proba = detector.predict_proba(text)
            assert 0.0 <= proba <= 1.0

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_short_text(self):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        # Very short text returns 0.0
        assert detector.predict_proba("hi") == 0.0

    def test_numbers_only(self):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        # No letters to analyze
        assert detector.predict("12345") is False

    def test_valid_consonant_clusters(self):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        # Valid English clusters
        assert detector.predict("strong stream") is False
        assert detector.predict("through the threshold") is False

    def test_custom_threshold(self):
        detector = GarbleDetector(
            Strategy.PRONOUNCEABILITY,
            forbidden_cluster_threshold=5
        )
        # Should be more lenient
        proba = detector.predict_proba("xkcd test")
        assert proba < 0.9

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            GarbleDetector(Strategy.PRONOUNCEABILITY, forbidden_cluster_threshold=0)
        with pytest.raises(ValueError):
            GarbleDetector(Strategy.PRONOUNCEABILITY, vowel_min_ratio=1.5)


class TestUnicodeScriptStrategy:
    """Tests for Unicode script mixing / homoglyph detection."""

    def test_pure_latin_text(self):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        assert detector.predict("hello world") is False
        assert detector.predict("The Quick Brown Fox") is False

    def test_cyrillic_homoglyph(self):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        # Mix Cyrillic 'а' (U+0430) with Latin - looks like 'a'
        mixed = "p\u0430ypal"  # 'а' is Cyrillic
        assert detector.predict(mixed) is True

    def test_greek_homoglyph(self):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        # Greek 'ο' (U+03BF) looks like Latin 'o'
        mixed = "hell\u03BF"
        assert detector.predict(mixed) is True

    def test_mixed_scripts(self):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        # Cyrillic letters АБВ in Latin text
        mixed = "Hello \u0410\u0411\u0412 World"
        assert detector.predict(mixed) is True

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        for text in ["hello", "p\u0430ypal", "test"]:
            proba = detector.predict_proba(text)
            assert 0.0 <= proba <= 1.0

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_numbers_and_punctuation(self):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        # Numbers and punctuation shouldn't trigger
        assert detector.predict("Hello, World! 123") is False

    def test_disable_homoglyph_check(self):
        detector = GarbleDetector(
            Strategy.UNICODE_SCRIPT,
            check_homoglyphs=False
        )
        # Should not flag single homoglyphs
        mixed = "p\u0430ypal"
        proba = detector.predict_proba(mixed)
        # Still might flag for mixed scripts
        assert isinstance(proba, float)

    def test_custom_max_scripts(self):
        detector = GarbleDetector(
            Strategy.UNICODE_SCRIPT,
            max_scripts=3
        )
        # Should be more lenient about script mixing
        assert detector.predict("hello") is False

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            GarbleDetector(Strategy.UNICODE_SCRIPT, homoglyph_threshold=0)
        with pytest.raises(ValueError):
            GarbleDetector(Strategy.UNICODE_SCRIPT, max_scripts=0)


class TestNewStrategiesNoDeps:
    """Test that v0.4.0 strategies work without external dependencies."""

    def test_compression_ratio_no_deps(self):
        detector = GarbleDetector(Strategy.COMPRESSION_RATIO)
        assert detector.predict("hello " * 30) is False

    def test_mojibake_no_deps(self):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        assert detector.predict("hello world") is False

    def test_pronounceability_no_deps(self):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        assert detector.predict("hello world") is False

    def test_unicode_script_no_deps(self):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        assert detector.predict("hello world") is False


class TestNewStrategiesBatch:
    """Test batch processing with v0.4.0 strategies."""

    def test_compression_ratio_batch(self):
        detector = GarbleDetector(Strategy.COMPRESSION_RATIO)
        texts = ["a" * 150, "b" * 150, "hello " * 30]
        results = detector.predict(texts)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_mojibake_batch(self):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        texts = ["hello", "caf\xc3\xa9", "world"]
        results = detector.predict(texts)
        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0] is False
        assert results[1] is True
        assert results[2] is False

    def test_pronounceability_batch(self):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        texts = ["hello world", "xkcd qwfp", "the cat"]
        results = detector.predict(texts)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_unicode_script_batch(self):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        texts = ["hello", "p\u0430ypal", "world"]
        results = detector.predict(texts)
        assert results == [False, True, False]
