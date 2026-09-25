"""Function-word density and collocation: casing and short words."""

import pytest

from pygarble import GarbleDetector, Strategy
from pygarble.data import FUNCTION_WORDS
from pygarble.preprocessing import title_case_ratio

ALL_CAPS_MASH = (
    "XKRF PLMQ BVZT NWSD JGHC TRBN MKPL QWRT ZXCV POIU "
    "LKJH MNBV GHJK TYUI VBNM RTYU DFGH CVBN ERTY XCVB"
)  # 20 tokens: zero-collocation text needs 20+ words to cross 0.5
PLAIN_13 = (
    "researchers analyzed thousands of samples collected across multiple "
    "regions during several recent years"
)


def test_function_word_table_is_shared():
    assert "the" in FUNCTION_WORDS
    assert "a" in FUNCTION_WORDS and "i" in FUNCTION_WORDS


def test_title_case_ratio_excludes_all_caps():
    assert title_case_ratio("Alice Bob Carol") == 1.0
    assert title_case_ratio("XKRF PLMQ") == 0.0
    assert title_case_ratio("I am") == pytest.approx(0.5)


def test_single_letter_function_words_count():
    detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
    assert detector._strategy_instance._tokenize("I am a cat") == [
        "i",
        "am",
        "a",
        "cat",
    ]


def test_all_caps_mash_is_not_title_case_exempt():
    fwd = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
    collocation = GarbleDetector(Strategy.WORD_COLLOCATION)
    assert fwd.score(ALL_CAPS_MASH) >= 0.5
    assert collocation.score(ALL_CAPS_MASH) >= 0.5


def test_plain_sentence_without_listed_collocations_is_clean():
    assert GarbleDetector(Strategy.WORD_COLLOCATION).score(PLAIN_13) < 0.5


def test_curly_apostrophe_keeps_contractions_whole():
    strategy = GarbleDetector(Strategy.WORD_COLLOCATION)._strategy_instance
    assert strategy._tokenize("don’t stop") == ["don't", "stop"]
