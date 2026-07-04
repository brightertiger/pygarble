"""
Regression tests for verified bug fixes (batch C2):

- FunctionWordDensityStrategy: expanded function words, graded scoring
- RareTrigramStrategy: real-world tokens removed, within-word trigrams
- ZipfConformityStrategy: flat distribution requires corroboration
- WordLookupStrategy + data/words.py: junk entries removed, diacritic
  folding, proper-noun dampening
"""

import pytest

from pygarble.data import ENGLISH_WORDS
from pygarble.strategies.function_word_density import (
    FunctionWordDensityStrategy,
)
from pygarble.strategies.rare_trigram import RareTrigramStrategy
from pygarble.strategies.word_lookup import WordLookupStrategy
from pygarble.strategies.zipf_conformity import ZipfConformityStrategy


class TestFunctionWordDensityFix:
    """Real prose with few function words must not be flagged."""

    ML_SENTENCE = (
        "Machine learning models process natural language text using "
        "neural networks trained on massive datasets today"
    )

    def test_dense_technical_sentence_not_flagged(self):
        strategy = FunctionWordDensityStrategy()
        assert strategy.predict_proba(self.ML_SENTENCE) < 0.5
        assert strategy.predict(self.ML_SENTENCE) is False

    def test_gibberish_20_tokens_flagged(self):
        strategy = FunctionWordDensityStrategy()
        gibberish = (
            "rekxqz vbnmpo zxqwer plmokn qazwsx edcrfv tgbyhn ujmikl "
            "wsxedc rfvtgb yhnujm iklopz qwaszx erdfcv tyghbn uijklm "
            "opzxcw asdfgh zxcvbn qwerty"
        )
        assert strategy.predict_proba(gibberish) >= 0.5

    def test_scores_above_half_require_zero_function_words(self):
        strategy = FunctionWordDensityStrategy()
        # 15 words, exactly one function word -> must stay below 0.5
        text = "the " + " ".join(["gorilla"] * 14)
        assert strategy.predict_proba(text) < 0.5

    def test_abstains_below_min_words(self):
        strategy = FunctionWordDensityStrategy()
        assert strategy.applicable("HTTP API REST") is False
        assert strategy.applicable("one two three four five six") is True


class TestRareTrigramFix:
    """www/xxx are real tokens; trigrams stay within words."""

    def test_web_address_not_flagged(self):
        strategy = RareTrigramStrategy()
        assert strategy.predict_proba("visit www xxx com") < 0.5
        assert strategy.predict("visit www xxx com") is False

    def test_gibberish_still_detected(self):
        strategy = RareTrigramStrategy()
        assert strategy.predict_proba("qxzqxz jqxwv") >= 0.5

    def test_trigrams_do_not_cross_word_boundaries(self):
        strategy = RareTrigramStrategy()
        # "zux" + "jqu": crossing the boundary would form "xjq"/"jqu"
        # style phantom trigrams; within-word there are no impossible ones
        assert strategy.predict_proba("relax jaunt quiz oxen") == 0.0

    def test_impossible_doubles_still_flagged(self):
        strategy = RareTrigramStrategy()
        assert strategy.predict("jjjqqq") is True
        assert strategy.predict("xqzjxq") is True


class TestZipfConformityFix:
    """All-unique real words are not gibberish without corroboration."""

    FRUITS = (
        "apple banana cherry mango papaya grape kiwi melon peach plum "
        "lion tiger bear wolf deer eagle hawk owl fox rabbit apricot "
        "fig date guava lime lemon orange pear quince berry"
    )

    def test_distinct_real_words_not_flagged(self):
        strategy = ZipfConformityStrategy()
        assert strategy.predict_proba(self.FRUITS) < 0.5
        assert strategy.predict(self.FRUITS) is False

    def test_unique_gibberish_still_flagged(self):
        strategy = ZipfConformityStrategy()
        gibberish = (
            "xkrf plmq bvzt nwsd jghc trbn mkpl wqzd lpnr fvxt qzml "
            "hkrp bntw xvfd cjmg rlwp gthx znkm vbqf djsr xtlw npfz "
            "mkcb ghvr wjqt bfrk nlgz xpcm hvtq dwrj ktsg fmqb zxwn "
            "pljr cvdh"
        )
        assert strategy.predict_proba(gibberish) > 0.5

    def test_abstains_below_min_words(self):
        strategy = ZipfConformityStrategy()
        assert strategy.applicable("hello world") is False

    def test_docstring_matches_defaults(self):
        strategy = ZipfConformityStrategy()
        assert strategy.min_words == 30
        assert strategy.ttr_threshold == 0.95
        assert strategy.hapax_threshold == 0.95
        doc = ZipfConformityStrategy.__doc__
        assert "30" in doc and "0.95" in doc


class TestWordLookupAndDataFix:
    """Junk dictionary entries removed; proper nouns dampened."""

    def test_junk_entries_removed_from_dictionary(self):
        for junk in ("qq", "zz", "xx", "jj", "aaaa", "abcd", "caf",
                     "abcdefghijklmnopqrstuvwxyz"):
            assert junk not in ENGLISH_WORDS, junk

    def test_real_short_words_kept(self):
        # Note: "a" and "i" are whitelisted but were never present in
        # the original 50K list, so only two-letter survivors are checked.
        for word in ("am", "an", "as", "at", "be", "do", "go",
                     "if", "in", "is", "it", "of", "on", "or", "to", "we"):
            assert word in ENGLISH_WORDS, word

    def test_junk_tokens_now_flagged(self):
        strategy = WordLookupStrategy()
        assert strategy.predict_proba("qq zz xx jj aaaa abcd") >= 0.5

    def test_proper_nouns_dampened(self):
        strategy = WordLookupStrategy()
        assert strategy.predict_proba("Ujjwal Srao went home") < 0.5
        assert strategy.predict_proba("Pikachu evolved") < 0.5

    def test_common_english_near_zero(self):
        strategy = WordLookupStrategy()
        assert strategy.predict_proba("the quick brown fox") < 0.2

    def test_lowercase_gibberish_still_flagged(self):
        strategy = WordLookupStrategy()
        assert strategy.predict_proba("xkqzv wpfjg mzxcv") >= 0.5

    def test_diacritics_folded_before_lookup(self):
        strategy = WordLookupStrategy()
        # café folds to cafe, which is a real dictionary word
        assert "cafe" in ENGLISH_WORDS
        assert strategy.predict_proba("café latte") == 0.0
