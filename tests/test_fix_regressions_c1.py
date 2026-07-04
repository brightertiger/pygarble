"""Regression tests for verified bug fixes in five strategies.

Each test pins a repro that previously scored on the wrong side of 0.5,
plus a true-gibberish sanity check to make sure the fix did not blunt
detection.
"""

import pytest

from pygarble import GarbleDetector, Strategy


class TestKeyboardPatternFix:
    """Short texts must not be flagged just for missing common trigrams."""

    @pytest.mark.parametrize("text", ["hello world", "family dinner", "hello"])
    def test_normal_short_text_not_flagged(self, text):
        detector = GarbleDetector(Strategy.KEYBOARD_PATTERN)
        assert detector.predict_proba(text) < 0.5
        assert detector.predict(text) is False

    @pytest.mark.parametrize("text", ["asdfghjkl", "qwertyuiop zxcvbnm"])
    def test_keyboard_mash_still_flagged(self, text):
        detector = GarbleDetector(Strategy.KEYBOARD_PATTERN)
        assert detector.predict_proba(text) >= 0.5
        assert detector.predict(text) is True


class TestEntropyBasedFix:
    """Short common words must not be flagged; the tiny common-bigram set
    was replaced with the real bigram model and entropy is normalized
    against expected English entropy."""

    @pytest.mark.parametrize("text", ["dog", "sun", "run", "day", "big fluffy dog"])
    def test_common_words_not_flagged(self, text):
        detector = GarbleDetector(Strategy.ENTROPY_BASED)
        assert detector.predict_proba(text) < 0.5
        assert detector.predict(text) is False

    @pytest.mark.parametrize("text", ["xkjq zvwp qmfg", "aaaaaaaaaa"])
    def test_gibberish_still_flagged(self, text):
        detector = GarbleDetector(Strategy.ENTROPY_BASED)
        assert detector.predict_proba(text) >= 0.5
        assert detector.predict(text) is True

    def test_gibberish_scores_above_normal_text(self):
        detector = GarbleDetector(Strategy.ENTROPY_BASED)
        assert detector.predict_proba("xkjq zvwp qmfg") > detector.predict_proba(
            "big fluffy dog"
        )


class TestBigramProbabilityFix:
    """Real English bigrams (sq, nj, nx, fj, dj, hh, ww, gz, kg) must not
    be treated as impossible, and bigrams must not span word boundaries."""

    @pytest.mark.parametrize(
        "text", ["lynx", "sqft", "50 sq ft", "enjoy", "banjo"]
    )
    def test_real_words_not_flagged(self, text):
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)
        assert detector.predict_proba(text) < 0.5
        assert detector.predict(text) is False

    @pytest.mark.parametrize(
        "text",
        ["fjord", "zigzag", "withhold", "adjust", "anxiety", "savvy", "www"],
    )
    def test_more_real_words_not_flagged(self, text):
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)
        assert detector.predict(text) is False

    def test_no_bigrams_across_word_boundaries(self):
        # "of fjords" would previously form "ff" across the boundary;
        # cross-word pairs must never count.
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)
        assert detector.predict("radio jazz") is False  # "oj" never forms

    def test_gibberish_still_flagged(self):
        detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)
        assert detector.predict_proba("qxjjxz zzqp xjq") >= 0.5
        assert detector.predict("qxjjxz zzqp xjq") is True


class TestVowelRatioFix:
    """'y' must act as a vowel in otherwise-vowelless words; long
    uppercase gibberish must not be exempted as an acronym; predict must
    agree with predict_proba."""

    @pytest.mark.parametrize(
        "text", ["my gym crypt", "sky fly try dry", "rhythm", "strengths", "sixths"]
    )
    def test_y_words_not_flagged(self, text):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        assert detector.predict_proba(text) < 0.5
        assert detector.predict(text) is False

    def test_uppercase_gibberish_flagged(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        assert detector.predict_proba("QWRTPZXCV KLMNBVCX") >= 0.5
        assert detector.predict("QWRTPZXCV KLMNBVCX") is True

    def test_short_acronyms_still_exempt(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        assert detector.predict("the NHS and HTML docs") is False

    def test_lowercase_gibberish_still_flagged(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        assert detector.predict_proba("zxcvbnm qwrtp") >= 0.5
        assert detector.predict("zxcvbnm qwrtp") is True

    def test_predict_agrees_with_proba(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        for text in ["sixths", "rhythm", "strengths", "zxcvbnm qwrtp"]:
            assert detector.predict(text) == (
                detector.predict_proba(text) >= 0.5
            )


class TestPatternMatchingFix:
    """A single strong pattern match must be decisive (proba >= 0.6)."""

    def test_single_pattern_match_fires(self):
        detector = GarbleDetector(Strategy.PATTERN_MATCHING)
        assert detector.predict_proba("asdfghjkl") >= 0.5
        assert detector.predict("asdfghjkl") is True

    def test_normal_text_matches_nothing(self):
        detector = GarbleDetector(Strategy.PATTERN_MATCHING)
        assert detector.predict_proba("hello world") == 0.0
        assert detector.predict("hello world") is False

    def test_more_matches_score_higher(self):
        detector = GarbleDetector(Strategy.PATTERN_MATCHING)
        one_ish = detector.predict_proba("AAAAAAA")
        many = detector.predict_proba("AAAAAAA qwerty !!!### 123456789")
        assert many > one_ish >= 0.6
