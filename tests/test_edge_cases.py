import pytest
from pygarble import GarbleDetector, EnsembleDetector, Strategy


class TestEdgeCases:
    def test_predict_proba_empty_string(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        assert detector.predict_proba("") == 0.0

    def test_predict_proba_whitespace_only(self):
        detector = GarbleDetector(Strategy.WORD_LENGTH)
        assert detector.predict_proba("   ") == 0.0

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        assert detector.predict("") is False

    def test_whitespace_only(self):
        detector = GarbleDetector(Strategy.WORD_LENGTH)
        assert detector.predict("   ") is False

    def test_extremely_long_string_no_spaces(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        long_string = "a" * 1001
        assert detector.predict(long_string) is True
        assert detector.predict_proba(long_string) == 1.0

    def test_extremely_long_string_with_spaces(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        long_string = (
            "This is a normal sentence with spaces. " * 30
        )  # Long but has spaces
        assert detector.predict(long_string) is False
        assert detector.predict_proba(long_string) < 1.0

    def test_extremely_long_string_with_tabs(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        long_string = "word\t" * 250  # Long string with tabs
        assert detector.predict(long_string) is False
        assert detector.predict_proba(long_string) < 1.0

    def test_extremely_long_string_with_newlines(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        long_string = "word\n" * 250  # Long string with newlines
        assert detector.predict(long_string) is False
        assert detector.predict_proba(long_string) < 1.0

    def test_extremely_long_string_with_mixed_whitespace(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        long_string = "word \t\n " * 200  # Long string with mixed whitespace
        assert detector.predict(long_string) is False
        assert detector.predict_proba(long_string) < 1.0

    def test_extremely_long_string_custom_threshold(self):
        detector = GarbleDetector(
            Strategy.CHARACTER_FREQUENCY, max_string_length=500
        )
        long_string = "a" * 501
        assert detector.predict(long_string) is True
        assert detector.predict_proba(long_string) == 1.0

    def test_extremely_long_string_below_threshold(self):
        detector = GarbleDetector(
            Strategy.STATISTICAL_ANALYSIS, max_string_length=2000
        )
        long_string = "a" * 1001
        assert detector.predict(long_string) is False
        assert detector.predict_proba(long_string) < 1.0

    def test_base64_like_string(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        base64_like = (
            "SGVsbG9Xb3JsZEhlbGxvV29ybGRIZWxsb1dvcmxk" * 50
        )  # Long base64-like string
        assert detector.predict(base64_like) is True
        assert detector.predict_proba(base64_like) == 1.0

    def test_english_word_validation_edge_cases(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION)
        
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0
        
        assert detector.predict("   ") is False
        assert detector.predict_proba("   ") == 0.0
        
        assert detector.predict("123") is False
        assert detector.predict_proba("123") == 0.0
        
        assert detector.predict("!@#") is False
        assert detector.predict_proba("!@#") == 0.0

    def test_english_word_validation_long_valid_text(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION)
        long_valid_text = "This is a very long sentence with many valid English words that should be recognized by the spell checker and classified as not garbled text because it contains proper English vocabulary throughout the entire string"
        assert detector.predict(long_valid_text) is False
        assert detector.predict_proba(long_valid_text) < 0.2

    def test_english_word_validation_long_garbled_text(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION, threshold=0.2)
        long_garbled_text = "asdfghjkl mnbvcxz lkjhgfds asdfghjkl mnbvcxz lkjhgfds asdfghjkl mnbvcxz lkjhgfds asdfghjkl mnbvcxz lkjhgfds"
        assert detector.predict(long_garbled_text) is True
        assert detector.predict_proba(long_garbled_text) > 0.2


class TestParameterValidation:
    """Test parameter validation across all classes."""

    def test_garble_detector_invalid_threshold_high(self):
        with pytest.raises(ValueError, match="threshold must be between"):
            GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=1.5)

    def test_garble_detector_invalid_threshold_negative(self):
        with pytest.raises(ValueError, match="threshold must be between"):
            GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=-0.1)

    def test_garble_detector_invalid_threads(self):
        with pytest.raises(ValueError, match="threads must be a positive"):
            GarbleDetector(Strategy.CHARACTER_FREQUENCY, threads=0)

    def test_ensemble_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold must be between"):
            EnsembleDetector(threshold=2.0)

    def test_ensemble_invalid_voting_mode(self):
        with pytest.raises(ValueError, match="voting must be"):
            EnsembleDetector(voting="invalid_mode")

    def test_ensemble_weighted_without_weights(self):
        with pytest.raises(ValueError, match="weights required"):
            EnsembleDetector(voting="weighted")

    def test_ensemble_weights_wrong_length(self):
        with pytest.raises(ValueError, match="weights must have same length"):
            EnsembleDetector(
                strategies=[Strategy.MARKOV_CHAIN, Strategy.NGRAM_FREQUENCY],
                voting="weighted",
                weights=[1.0],  # Only 1 weight for 2 strategies
            )

    def test_ensemble_negative_weights(self):
        with pytest.raises(ValueError, match="weights must be non-negative"):
            EnsembleDetector(
                strategies=[Strategy.MARKOV_CHAIN],
                voting="weighted",
                weights=[-1.0],
            )

    def test_ensemble_zero_weights(self):
        with pytest.raises(ValueError, match="weights must not all be zero"):
            EnsembleDetector(
                strategies=[Strategy.MARKOV_CHAIN, Strategy.NGRAM_FREQUENCY],
                voting="weighted",
                weights=[0.0, 0.0],
            )

    def test_ensemble_empty_strategies(self):
        with pytest.raises(ValueError, match="at least one strategy"):
            EnsembleDetector(strategies=[])


class TestNewStrategyEdgeCases:
    """Edge case tests for new strategies (v0.3.0)."""

    def test_markov_chain_empty_string(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_markov_chain_only_numbers(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        assert detector.predict("12345") is False
        assert detector.predict_proba("12345") == 0.0

    def test_markov_chain_short_text(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN, min_length=10)
        assert detector.predict("hi") is False

    def test_ngram_frequency_empty_string(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_ngram_frequency_no_trigrams(self):
        detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
        # Very short words - no trigrams possible
        assert detector.predict("a b c d") is False

    def test_word_lookup_empty_string(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_word_lookup_only_short_words(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP, min_word_length=5)
        # All words shorter than min_word_length
        assert detector.predict("a b c d e") is False

    def test_symbol_ratio_empty_string(self):
        detector = GarbleDetector(Strategy.SYMBOL_RATIO)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_symbol_ratio_all_spaces(self):
        detector = GarbleDetector(Strategy.SYMBOL_RATIO)
        assert detector.predict("     ") is False

    def test_symbol_ratio_short_text(self):
        detector = GarbleDetector(Strategy.SYMBOL_RATIO, min_length=10)
        assert detector.predict("!@#") is False

    def test_repetition_empty_string(self):
        detector = GarbleDetector(Strategy.REPETITION)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_repetition_very_short(self):
        detector = GarbleDetector(Strategy.REPETITION)
        assert detector.predict("ab") is False

    def test_hex_string_empty_string(self):
        detector = GarbleDetector(Strategy.HEX_STRING)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_hex_string_short_text(self):
        detector = GarbleDetector(Strategy.HEX_STRING)
        # Shorter than 8 chars
        assert detector.predict("abc") is False


class TestNewStrategyParameterValidation:
    """Parameter validation for new strategies."""

    def test_markov_chain_positive_threshold(self):
        with pytest.raises(ValueError, match="threshold_per_char must be non-positive"):
            GarbleDetector(Strategy.MARKOV_CHAIN, threshold_per_char=1.0)

    def test_markov_chain_zero_min_length(self):
        with pytest.raises(ValueError, match="min_length must be at least 1"):
            GarbleDetector(Strategy.MARKOV_CHAIN, min_length=0)

    def test_ngram_invalid_threshold(self):
        with pytest.raises(ValueError, match="common_ratio_threshold must be between"):
            GarbleDetector(Strategy.NGRAM_FREQUENCY, common_ratio_threshold=1.5)

    def test_ngram_zero_min_length(self):
        with pytest.raises(ValueError, match="min_length must be at least 1"):
            GarbleDetector(Strategy.NGRAM_FREQUENCY, min_length=0)

    def test_word_lookup_invalid_threshold(self):
        with pytest.raises(ValueError, match="unknown_threshold must be between"):
            GarbleDetector(Strategy.WORD_LOOKUP, unknown_threshold=-0.1)

    def test_word_lookup_zero_word_length(self):
        with pytest.raises(ValueError, match="min_word_length must be at least 1"):
            GarbleDetector(Strategy.WORD_LOOKUP, min_word_length=0)

    def test_symbol_ratio_invalid_threshold(self):
        with pytest.raises(ValueError, match="symbol_threshold must be between"):
            GarbleDetector(Strategy.SYMBOL_RATIO, symbol_threshold=2.0)

    def test_symbol_ratio_negative_min_length(self):
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            GarbleDetector(Strategy.SYMBOL_RATIO, min_length=-1)

    def test_repetition_zero_char_repeat(self):
        with pytest.raises(ValueError, match="max_char_repeat must be at least 1"):
            GarbleDetector(Strategy.REPETITION, max_char_repeat=0)

    def test_repetition_zero_pattern_repeat(self):
        with pytest.raises(ValueError, match="max_pattern_repeat must be at least 1"):
            GarbleDetector(Strategy.REPETITION, max_pattern_repeat=0)

    def test_repetition_invalid_diversity(self):
        with pytest.raises(ValueError, match="diversity_threshold must be between"):
            GarbleDetector(Strategy.REPETITION, diversity_threshold=1.5)

    def test_hex_string_negative_min_length(self):
        with pytest.raises(ValueError, match="min_hex_length must be non-negative"):
            GarbleDetector(Strategy.HEX_STRING, min_hex_length=-1)

    def test_hex_string_invalid_ratio(self):
        with pytest.raises(ValueError, match="hex_ratio_threshold must be between"):
            GarbleDetector(Strategy.HEX_STRING, hex_ratio_threshold=1.5)


class TestTypeErrors:
    """Test type error handling."""

    def test_predict_invalid_type_int(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        with pytest.raises(TypeError, match="Input must be a string"):
            detector.predict(123)

    def test_predict_invalid_type_none(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        with pytest.raises(TypeError, match="Input must be a string"):
            detector.predict(None)

    def test_predict_proba_invalid_type(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        with pytest.raises(TypeError, match="Input must be a string"):
            detector.predict_proba({"text": "hello"})

    def test_ensemble_predict_invalid_type(self):
        ensemble = EnsembleDetector()
        with pytest.raises(TypeError, match="Input must be a string"):
            ensemble.predict(42)

    def test_ensemble_predict_proba_invalid_type(self):
        ensemble = EnsembleDetector()
        with pytest.raises(TypeError, match="Input must be a string"):
            ensemble.predict_proba(42)


class TestUnicodeEdgeCases:
    """Test handling of Unicode and non-ASCII text."""

    def test_unicode_emoji(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        # Emoji-only text should not crash
        result = detector.predict("😀😎🎉")
        assert isinstance(result, bool)

    def test_unicode_chinese(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        # Chinese text should be handled gracefully
        result = detector.predict("你好世界")
        assert isinstance(result, bool)

    def test_unicode_mixed(self):
        detector = GarbleDetector(Strategy.MARKOV_CHAIN)
        # Mixed Unicode and ASCII
        result = detector.predict("Hello 世界 café résumé")
        assert isinstance(result, bool)

    def test_unicode_combining_chars(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        # Text with combining characters (e.g., accents)
        result = detector.predict("café résumé naïve")
        assert isinstance(result, bool)


class TestBatchProcessing:
    """Test batch processing edge cases."""

    def test_empty_list(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        result = detector.predict([])
        assert result == []

    def test_single_item_list(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        result = detector.predict(["hello"])
        assert len(result) == 1
        assert isinstance(result[0], bool)

    def test_list_with_empty_strings(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        result = detector.predict(["", "hello", "", "world"])
        assert len(result) == 4
        assert result[0] is False  # Empty string
        assert result[2] is False  # Empty string

    def test_ensemble_batch(self):
        ensemble = EnsembleDetector()
        result = ensemble.predict(["hello world", "asdfghjkl"])
        assert len(result) == 2
        assert all(isinstance(r, bool) for r in result)


class TestEnsembleVotingModes:
    """Test all ensemble voting modes."""

    def test_majority_voting(self):
        ensemble = EnsembleDetector(voting="majority")
        # Normal text should not be garbled with majority vote
        result = ensemble.predict("This is a normal sentence.")
        assert isinstance(result, bool)

    def test_any_voting(self):
        ensemble = EnsembleDetector(voting="any")
        # "any" is stricter - might flag more text
        result = ensemble.predict("This is a normal sentence.")
        assert isinstance(result, bool)

    def test_all_voting(self):
        ensemble = EnsembleDetector(voting="all")
        # "all" is more lenient - only flags if all agree
        result = ensemble.predict("This is a normal sentence.")
        assert isinstance(result, bool)

    def test_average_voting(self):
        ensemble = EnsembleDetector(voting="average")
        result = ensemble.predict("hello world")
        assert isinstance(result, bool)

    def test_weighted_voting(self):
        ensemble = EnsembleDetector(
            strategies=[Strategy.MARKOV_CHAIN, Strategy.NGRAM_FREQUENCY],
            voting="weighted",
            weights=[0.7, 0.3],
        )
        result = ensemble.predict("hello world")
        assert isinstance(result, bool)

    def test_weighted_proba(self):
        ensemble = EnsembleDetector(
            strategies=[Strategy.MARKOV_CHAIN, Strategy.NGRAM_FREQUENCY],
            voting="weighted",
            weights=[0.7, 0.3],
        )
        proba = ensemble.predict_proba("hello world")
        assert 0.0 <= proba <= 1.0


class TestBoundaryConditions:
    """Test boundary conditions and numeric edge cases."""

    def test_threshold_exactly_zero(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=0.0)
        # Everything should be classified as garbled with threshold=0
        # (any non-zero probability exceeds threshold)
        result = detector.predict("hello")
        assert isinstance(result, bool)

    def test_threshold_exactly_one(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=1.0)
        # Only text with proba >= 1.0 should be classified as garbled
        result = detector.predict("hello")
        assert result is False  # Normal text has proba < 1.0

    def test_symbol_threshold_zero(self):
        # Edge case: threshold=0 means any symbol is suspicious
        detector = GarbleDetector(Strategy.SYMBOL_RATIO, symbol_threshold=0.0)
        result = detector.predict("hello123")
        assert isinstance(result, bool)

    def test_hex_ratio_threshold_one(self):
        # Edge case: threshold=1.0 means only 100% hex is flagged
        detector = GarbleDetector(Strategy.HEX_STRING, hex_ratio_threshold=1.0)
        result = detector.predict("abcdef123456")
        assert isinstance(result, bool)

    def test_vowel_ratio_extreme_min(self):
        # Very low min ratio
        detector = GarbleDetector(Strategy.VOWEL_RATIO, min_vowel_ratio=0.01)
        result = detector.predict("bcdfg")  # No vowels
        assert isinstance(result, bool)

    def test_vowel_ratio_extreme_max(self):
        # Very high max ratio
        detector = GarbleDetector(Strategy.VOWEL_RATIO, max_vowel_ratio=0.99)
        result = detector.predict("aeiouaeiou")  # All vowels
        assert isinstance(result, bool)
