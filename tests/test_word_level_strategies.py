"""
Tests for word-level strategies added in v0.6.0:
- FunctionWordDensityStrategy
- AffixDetectionStrategy
- ZipfConformityStrategy
- WordCollocationStrategy
"""

import pytest
from pygarble import GarbleDetector, Strategy


class TestFunctionWordDensityStrategy:
    """Tests for function word density detection."""

    def test_valid_english_text(self):
        detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
        assert detector.predict("The cat sat on the mat") is False
        assert detector.predict("I have been to the store") is False
        assert detector.predict("She was reading a book in the library") is False

    def test_garbled_text(self):
        detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
        assert detector.predict(
            "xkrf plmq bvzt nwsd jghc trbn mkpl wqzd lpnr fvxt"
        ) is True

    def test_short_text_exempt(self):
        detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
        # Fewer than 5 words -> exempt
        assert detector.predict("hello world") is False
        assert detector.predict("xkrf plmq bvzt") is False

    def test_all_caps(self):
        detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
        assert detector.predict("THE CAT SAT ON THE MAT AND THE DOG") is False

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
        for text in ["The cat sat on the mat", "xkrf plmq bvzt nwsd jghc"]:
            proba = detector.predict_proba(text)
            assert 0.0 <= proba <= 1.0

    def test_valid_text_low_probability(self):
        detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
        proba = detector.predict_proba(
            "The quick brown fox jumps over the lazy dog"
        )
        assert proba < 0.5

    def test_garbled_high_probability(self):
        detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
        proba = detector.predict_proba(
            "xkrf plmq bvzt nwsd jghc trbn mkpl wqzd lpnr fvxt"
        )
        assert proba > 0.5

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_technical_short_text(self):
        """Short technical text should not be flagged."""
        detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
        assert detector.predict("HTTP API REST") is False

    def test_text_with_some_function_words(self):
        """Text with even a few function words should pass."""
        detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
        proba = detector.predict_proba(
            "Python is a great programming language for data science"
        )
        assert proba < 0.5


class TestAffixDetectionStrategy:
    """Tests for affix detection."""

    def test_valid_english_text(self):
        detector = GarbleDetector(Strategy.AFFIX_DETECTION)
        assert detector.predict(
            "The programming language is incredibly powerful and usable"
        ) is False

    def test_short_text_exempt(self):
        detector = GarbleDetector(Strategy.AFFIX_DETECTION)
        assert detector.predict("cat dog run") is False
        assert detector.predict("xkrf plmq") is False

    def test_short_words_exempt(self):
        """Words shorter than min_word_length should be excluded."""
        detector = GarbleDetector(Strategy.AFFIX_DETECTION)
        # All words < 4 chars -> no analyzable words -> exempt
        assert detector.predict("cat dog run sit eat the and") is False

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.AFFIX_DETECTION)
        for text in ["understanding programming", "xkrf plmq bvzt nwsd"]:
            proba = detector.predict_proba(text)
            assert 0.0 <= proba <= 1.0

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.AFFIX_DETECTION)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_words_with_affixes_pass(self):
        """Text with common affixes should not be flagged."""
        detector = GarbleDetector(Strategy.AFFIX_DETECTION)
        proba = detector.predict_proba(
            "The unbelievable transformation was incredibly"
            " powerful and meaningful for everyone involved"
        )
        assert proba < 0.5

    def test_garbled_long_words(self):
        """Long garbled words without affixes should be flagged."""
        detector = GarbleDetector(Strategy.AFFIX_DETECTION)
        proba = detector.predict_proba(
            "xkrfm plmqn bvztk nwsdr jghcm"
            " trbnp mkplw wqzdl lpnrx fvxtb"
        )
        assert proba > 0.5


class TestZipfConformityStrategy:
    """Tests for Zipf's law conformity detection."""

    def test_valid_long_text(self):
        detector = GarbleDetector(Strategy.ZIPF_CONFORMITY)
        # Natural text with repeated function words
        text = (
            "The cat sat on the mat and the dog lay on the rug "
            "by the fire in the warm room near the big chair"
        )
        assert detector.predict(text) is False

    def test_short_text_exempt(self):
        detector = GarbleDetector(Strategy.ZIPF_CONFORMITY)
        assert detector.predict("hello world") is False
        assert detector.predict("short text here") is False

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.ZIPF_CONFORMITY)
        text = (
            "The cat sat on the mat and the dog lay on the rug "
            "by the fire in the warm room near the big chair"
        )
        proba = detector.predict_proba(text)
        assert 0.0 <= proba <= 1.0

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.ZIPF_CONFORMITY)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_all_unique_words_flagged(self):
        """30+ unique random words should be flagged."""
        detector = GarbleDetector(Strategy.ZIPF_CONFORMITY)
        # 35 unique all-alpha garbled words
        words = [
            "xkrf", "plmq", "bvzt", "nwsd", "jghc",
            "trbn", "mkpl", "wqzd", "lpnr", "fvxt",
            "qzml", "hkrp", "bntw", "xvfd", "cjmg",
            "rlwp", "gthx", "znkm", "vbqf", "djsr",
            "xtlw", "npfz", "mkcb", "ghvr", "wjqt",
            "bfrk", "nlgz", "xpcm", "hvtq", "dwrj",
            "ktsg", "fmqb", "zxwn", "pljr", "cvdh",
        ]
        proba = detector.predict_proba(" ".join(words))
        assert proba > 0.5

    def test_repeated_text_not_flagged(self):
        """Text with natural word repetition should pass."""
        detector = GarbleDetector(Strategy.ZIPF_CONFORMITY)
        text = (
            "the the the the the a a a a is is is "
            "and and or or but the a the is and the"
        )
        proba = detector.predict_proba(text)
        assert proba < 0.5


class TestWordCollocationStrategy:
    """Tests for word collocation detection."""

    def test_valid_english_text(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        assert detector.predict(
            "It is going to be a long day for the team"
        ) is False

    def test_short_text_exempt(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        assert detector.predict("hello world") is False
        assert detector.predict("xkrf plmq") is False

    def test_garbled_text(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        assert detector.predict(
            "xkrf plmq bvzt nwsd jghc trbn mkpl wqzd lpnr fvxt qzml hkrp"
        ) is True

    def test_probability_range(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        for text in [
            "It is going to be a long day",
            "xkrf plmq bvzt nwsd jghc",
        ]:
            proba = detector.predict_proba(text)
            assert 0.0 <= proba <= 1.0

    def test_valid_text_low_probability(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        proba = detector.predict_proba(
            "The cat sat on the mat in the room"
        )
        assert proba < 0.5

    def test_garbled_high_probability(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        proba = detector.predict_proba(
            "xkrf plmq bvzt nwsd jghc trbn mkpl wqzd lpnr fvxt qzml hkrp"
        )
        assert proba > 0.5

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_short_garbled_below_threshold(self):
        """Short garbled text should score below 0.5."""
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        proba = detector.predict_proba("xkrf plmq bvzt nwsd jghc trbn mkpl")
        assert proba < 0.5


class TestHighPrecisionWordLevel:
    """Ensure no false positives on valid English text."""

    VALID_TEXTS = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a great programming language",
        "I have been to the store and back again",
        "She was reading a book in the library",
        "We are going to be late for the meeting",
        "The United States of America is a country",
        "It is important to understand the problem",
        "He said that he would be there on time",
        "Natural language processing is fascinating",
        "The best way to learn is by doing it yourself",
    ]

    @pytest.mark.parametrize("strategy", [
        Strategy.FUNCTION_WORD_DENSITY,
        Strategy.WORD_COLLOCATION,
    ])
    def test_no_false_positives(self, strategy):
        detector = GarbleDetector(strategy)
        for text in self.VALID_TEXTS:
            assert detector.predict(text) is False, (
                f"{strategy.value} flagged valid text: {text!r}"
            )

    @pytest.mark.parametrize("strategy", [
        Strategy.FUNCTION_WORD_DENSITY,
        Strategy.WORD_COLLOCATION,
    ])
    def test_low_probability_on_valid(self, strategy):
        detector = GarbleDetector(strategy)
        for text in self.VALID_TEXTS:
            proba = detector.predict_proba(text)
            assert proba < 0.5, (
                f"{strategy.value} gave {proba:.2f} for: {text!r}"
            )
