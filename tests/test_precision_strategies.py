"""
Tests for high-precision gibberish detection strategies.

These tests verify that:
1. Strategies correctly identify garbled text
2. Strategies DON'T flag valid text as garbled (high precision)
3. Typos and minor errors are tolerated
"""

import pytest
from pygarble import GarbleDetector, Strategy


class TestBigramProbabilityStrategy:
    """Tests for BIGRAM_PROBABILITY strategy."""

    def test_detects_impossible_bigrams(self):
        """Should detect text with impossible letter combinations."""
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)

        # Text with impossible bigrams (qx, jj, xz, etc.)
        assert detector.predict("qxjjxz") is True
        assert detector.predict("bxcxdx") is True

    def test_allows_normal_text(self):
        """Should NOT flag normal English text."""
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)

        # Normal sentences
        assert detector.predict("The quick brown fox jumps") is False
        assert detector.predict("Hello world") is False
        assert detector.predict("Python programming") is False
        assert detector.predict("Machine learning") is False

    def test_allows_typos(self):
        """Should tolerate common typos."""
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)

        # Text with typos
        assert detector.predict("teh quick brown fox") is False
        assert detector.predict("recieve the package") is False
        assert detector.predict("definately going") is False

    def test_allows_unusual_but_valid_words(self):
        """Should allow unusual but valid English words."""
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)

        # Unusual but valid words
        assert detector.predict("rhythms") is False
        assert detector.predict("strengths") is False
        assert detector.predict("sphinx") is False

    def test_empty_and_short_text(self):
        """Should handle empty and very short text."""
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)

        assert detector.predict("") is False
        assert detector.predict("a") is False
        assert detector.predict("ab") is False


class TestLetterPositionStrategy:
    """Tests for LETTER_POSITION strategy."""

    def test_detects_invalid_endings(self):
        """Should detect words ending in impossible letters."""
        detector = GarbleDetector(Strategy.LETTER_POSITION)

        # Words ending in j, q are impossible
        assert detector.predict("wordj endq") is True

    def test_detects_invalid_starts(self):
        """Should detect words starting with impossible pairs."""
        detector = GarbleDetector(Strategy.LETTER_POSITION)

        # Impossible word starts
        assert detector.predict("xjword bwtext") is True

    def test_allows_normal_text(self):
        """Should NOT flag normal English text."""
        detector = GarbleDetector(Strategy.LETTER_POSITION)

        assert detector.predict("Strange strings") is False
        assert detector.predict("Through the threshold") is False
        assert detector.predict("Screen and strength") is False

    def test_allows_uncommon_but_valid_words(self):
        """Should allow uncommon but valid word patterns."""
        detector = GarbleDetector(Strategy.LETTER_POSITION)

        # Unusual starts that ARE valid
        assert detector.predict("Gnocchi for dinner") is False
        assert detector.predict("Pneumonia treatment") is False
        assert detector.predict("Pterodactyl fossil") is False


class TestConsonantSequenceStrategy:
    """Tests for CONSONANT_SEQUENCE strategy."""

    def test_detects_impossible_consonant_runs(self):
        """Should detect impossibly long consonant sequences."""
        detector = GarbleDetector(Strategy.CONSONANT_SEQUENCE)

        # 7+ consonants in a row is impossible
        assert detector.predict("bcdfghjklmn") is True
        assert detector.predict("xyzpqrstv") is True

    def test_allows_valid_consonant_clusters(self):
        """Should allow valid English consonant clusters."""
        detector = GarbleDetector(Strategy.CONSONANT_SEQUENCE)

        # Valid clusters: str, scr, ngths, etc.
        assert detector.predict("strengths") is False
        assert detector.predict("scratch") is False
        assert detector.predict("twelfths") is False

    def test_allows_normal_text(self):
        """Should NOT flag normal English text."""
        detector = GarbleDetector(Strategy.CONSONANT_SEQUENCE)

        assert detector.predict("The strength of the string") is False
        assert detector.predict("Christmas presents") is False


class TestVowelPatternStrategy:
    """Tests for VOWEL_PATTERN strategy."""

    def test_detects_impossible_vowel_runs(self):
        """Should detect impossibly long vowel sequences."""
        detector = GarbleDetector(Strategy.VOWEL_PATTERN)

        # Repeated same vowel (impossible in English)
        assert detector.predict("aaaaaaa") is True
        assert detector.predict("iiiiiii") is True
        assert detector.predict("eeeeeee") is True

    def test_allows_valid_vowel_sequences(self):
        """Should allow valid English vowel sequences."""
        detector = GarbleDetector(Strategy.VOWEL_PATTERN)

        # Valid sequences: eau, iou, etc.
        assert detector.predict("beautiful") is False
        assert detector.predict("queue") is False
        assert detector.predict("onomatopoeia") is False
        assert detector.predict("continuous") is False

    def test_allows_normal_text(self):
        """Should NOT flag normal English text."""
        detector = GarbleDetector(Strategy.VOWEL_PATTERN)

        assert detector.predict("The beautiful view") is False
        assert detector.predict("Audio and video") is False


class TestLetterFrequencyStrategy:
    """Tests for LETTER_FREQUENCY strategy."""

    def test_detects_abnormal_frequency(self):
        """Should detect text with abnormal letter frequencies."""
        detector = GarbleDetector(Strategy.LETTER_FREQUENCY)

        # Text dominated by rare letters
        assert detector.predict("jjjj qqqq xxxx zzzz jqxz") is True
        assert detector.predict("xqzjxqzjxqzjxqzjxqzj") is True

    def test_allows_normal_text(self):
        """Should NOT flag normal English text."""
        detector = GarbleDetector(Strategy.LETTER_FREQUENCY)

        assert detector.predict(
            "The quick brown fox jumps over the lazy dog"
        ) is False
        assert detector.predict(
            "Python is a great programming language"
        ) is False
        assert detector.predict(
            "This is a normal sentence with typical letter distribution"
        ) is False

    def test_handles_short_text(self):
        """Short text should not be flagged."""
        detector = GarbleDetector(Strategy.LETTER_FREQUENCY)

        # Short text has naturally noisy frequency
        assert detector.predict("xyz") is False
        assert detector.predict("quick") is False


class TestRareTrigramStrategy:
    """Tests for RARE_TRIGRAM strategy."""

    def test_detects_impossible_trigrams(self):
        """Should detect text with impossible trigrams."""
        detector = GarbleDetector(Strategy.RARE_TRIGRAM)

        # Impossible trigrams: jjj, qqq, xqz, etc.
        assert detector.predict("jjjqqq") is True
        assert detector.predict("xqzjxq") is True

    def test_allows_normal_text(self):
        """Should NOT flag normal English text."""
        detector = GarbleDetector(Strategy.RARE_TRIGRAM)

        assert detector.predict("The quick brown fox") is False
        assert detector.predict("Strength and wisdom") is False
        assert detector.predict("Through the threshold") is False

    def test_allows_unusual_valid_words(self):
        """Should allow unusual but valid English words."""
        detector = GarbleDetector(Strategy.RARE_TRIGRAM)

        # These have unusual trigrams but are valid
        assert detector.predict("rhythms and myths") is False
        assert detector.predict("Czech and quirky") is False


class TestHighPrecisionOnRealText:
    """
    Integration tests verifying high precision on realistic text samples.
    All new strategies should NOT flag these as garbled.
    """

    VALID_TEXTS = [
        # Normal sentences
        "The quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",
        "Machine learning algorithms process data.",
        # Technical text
        "HTTP/HTTPS protocols use TCP/IP stack.",
        "The API returns JSON responses.",
        "Run npm install to get dependencies.",
        # Text with typos (should NOT be flagged)
        "Recieve the pacakge tommorrow.",
        "The definately occured today.",
        "Seperate the peices carefully.",
        # Unusual but valid words
        "The rhythm of the sphinx.",
        "Strengths and twelfths measured.",
        "Onomatopoeia is beautiful.",
        # Proper nouns
        "Microsoft and Google compete.",
        "Dr. Johnson visited NYC.",
        # All caps (valid)
        "THIS IS IMPORTANT",
        "URGENT MESSAGE HERE",
    ]

    @pytest.mark.parametrize("text", VALID_TEXTS)
    def test_bigram_precision(self, text):
        """BIGRAM_PROBABILITY should not flag valid text."""
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)
        assert detector.predict(text) is False, f"False positive on: {text}"

    @pytest.mark.parametrize("text", VALID_TEXTS)
    def test_letter_position_precision(self, text):
        """LETTER_POSITION should not flag valid text."""
        detector = GarbleDetector(Strategy.LETTER_POSITION)
        assert detector.predict(text) is False, f"False positive on: {text}"

    @pytest.mark.parametrize("text", VALID_TEXTS)
    def test_consonant_sequence_precision(self, text):
        """CONSONANT_SEQUENCE should not flag valid text."""
        detector = GarbleDetector(Strategy.CONSONANT_SEQUENCE)
        assert detector.predict(text) is False, f"False positive on: {text}"

    @pytest.mark.parametrize("text", VALID_TEXTS)
    def test_vowel_pattern_precision(self, text):
        """VOWEL_PATTERN should not flag valid text."""
        detector = GarbleDetector(Strategy.VOWEL_PATTERN)
        assert detector.predict(text) is False, f"False positive on: {text}"

    @pytest.mark.parametrize("text", VALID_TEXTS)
    def test_letter_frequency_precision(self, text):
        """LETTER_FREQUENCY should not flag valid text."""
        detector = GarbleDetector(Strategy.LETTER_FREQUENCY)
        assert detector.predict(text) is False, f"False positive on: {text}"

    @pytest.mark.parametrize("text", VALID_TEXTS)
    def test_rare_trigram_precision(self, text):
        """RARE_TRIGRAM should not flag valid text."""
        detector = GarbleDetector(Strategy.RARE_TRIGRAM)
        assert detector.predict(text) is False, f"False positive on: {text}"


class TestRecallOnGarbledText:
    """
    Tests verifying that strategies DO detect actual garbled text.
    """

    GARBLED_TEXTS = [
        "xjqzxjqzxjqz",  # Impossible bigrams and trigrams
        "bxcxdxfxgx",    # Impossible bigrams
        "jjjkkkqqq",     # Invalid doubles
        "aaaaaeeeeeiiiii",  # Invalid vowel runs
        "qqqxxx",        # Invalid doubles and rare letters
    ]

    def test_at_least_some_detection(self):
        """At least some strategies should flag each garbled text."""
        strategies = [
            Strategy.BIGRAM_PROBABILITY,
            Strategy.CONSONANT_SEQUENCE,
            Strategy.VOWEL_PATTERN,
            Strategy.LETTER_FREQUENCY,
            Strategy.RARE_TRIGRAM,
        ]

        for text in self.GARBLED_TEXTS:
            detections = 0
            for strategy in strategies:
                detector = GarbleDetector(strategy)
                if detector.predict(text):
                    detections += 1

            # At least 1 strategy should flag each garbled text
            # (being conservative to match high-precision design)
            assert detections >= 1, (
                f"No strategies flagged: {text}"
            )


class TestBatchProcessing:
    """Test batch processing for new strategies."""

    def test_batch_predict(self):
        """Should handle batch predictions."""
        strategies = [
            Strategy.BIGRAM_PROBABILITY,
            Strategy.LETTER_POSITION,
            Strategy.CONSONANT_SEQUENCE,
            Strategy.VOWEL_PATTERN,
            Strategy.LETTER_FREQUENCY,
            Strategy.RARE_TRIGRAM,
        ]

        texts = [
            "Normal text here",
            "Another normal sentence",
            "xjqzxjqz",  # Garbled
            "Hello world",
        ]

        for strategy in strategies:
            detector = GarbleDetector(strategy)
            results = detector.predict(texts)

            assert isinstance(results, list)
            assert len(results) == len(texts)
            # First, second, and fourth should be normal (False)
            assert results[0] is False
            assert results[1] is False
            assert results[3] is False

    def test_batch_predict_proba(self):
        """Should handle batch probability predictions."""
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)

        texts = ["Normal text", "xjqzxjqz"]
        probas = detector.predict_proba(texts)

        assert isinstance(probas, list)
        assert len(probas) == 2
        assert 0.0 <= probas[0] <= 1.0
        assert 0.0 <= probas[1] <= 1.0
