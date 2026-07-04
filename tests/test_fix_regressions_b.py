"""Regression tests for verified bug fixes (batch B).

Covers repetition, consonant_sequence, vowel_pattern, affix_detection,
markov_chain, ngram_frequency, letter_frequency and symbol_ratio.
"""

from pygarble.strategies.affix_detection import AffixDetectionStrategy
from pygarble.strategies.consonant_sequence import ConsonantSequenceStrategy
from pygarble.strategies.letter_frequency import LetterFrequencyStrategy
from pygarble.strategies.markov_chain import MarkovChainStrategy
from pygarble.strategies.ngram_frequency import NGramFrequencyStrategy
from pygarble.strategies.repetition import RepetitionStrategy
from pygarble.strategies.symbol_ratio import SymbolRatioStrategy
from pygarble.strategies.vowel_pattern import VowelPatternStrategy

NORMAL_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly "
    "behind the distant mountains. Children play in the park as birds sing "
    "their evening songs. A gentle breeze carries the scent of blooming "
    "flowers across the quiet neighborhood streets tonight."
)


class TestRepetitionDiversityLengthAware:
    """Bug 1: unique chars cap at 36, so long normal paragraphs failed
    the diversity floor and were flagged."""

    def test_normal_paragraph_not_flagged(self):
        strategy = RepetitionStrategy()
        assert len(NORMAL_PARAGRAPH) > 150
        assert strategy.predict_proba(NORMAL_PARAGRAPH) < 0.3

    def test_long_normal_text_not_flagged(self):
        strategy = RepetitionStrategy()
        text = (NORMAL_PARAGRAPH + " ") * 2  # ~500 chars
        assert strategy.predict_proba(text) < 0.3

    def test_low_diversity_still_flagged(self):
        strategy = RepetitionStrategy()
        assert strategy.predict_proba("aaaaabbbbbaaaaabbbbb") >= 0.5


class TestRepetitionWhitespaceAndFormatting:
    """Bug 2: (.)\\1{3,} matched whitespace/punctuation runs."""

    def test_aligned_whitespace_not_flagged(self):
        strategy = RepetitionStrategy()
        assert strategy.predict_proba("name          value") < 0.3

    def test_section_divider_not_flagged(self):
        strategy = RepetitionStrategy()
        assert strategy.predict_proba("Section ----------") < 0.3

    def test_repeated_letters_still_flagged(self):
        strategy = RepetitionStrategy()
        assert strategy.predict_proba("aaaaaaa") >= 0.5


class TestRepetitionWordLevel:
    """Bug 3: docstring promised word-level detection that didn't exist."""

    def test_single_repeated_word(self):
        strategy = RepetitionStrategy()
        assert strategy.predict_proba("test test test") >= 0.5

    def test_repeated_word_four_times(self):
        strategy = RepetitionStrategy()
        assert strategy.predict_proba(
            "buffalo buffalo buffalo buffalo"
        ) >= 0.5

    def test_two_word_cycle(self):
        strategy = RepetitionStrategy()
        assert strategy.predict_proba("foo bar foo bar foo bar") >= 0.5

    def test_occasional_repeats_not_flagged(self):
        strategy = RepetitionStrategy()
        assert strategy.predict_proba("it is what it is") < 0.5


class TestConsonantSequenceAllCaps:
    """Bug 4: every all-caps word was exempted as an acronym, making
    shouted gibberish invisible."""

    def test_long_uppercase_gibberish_flagged(self):
        strategy = ConsonantSequenceStrategy()
        assert strategy.predict_proba("XKCDQRTZPLM BLAH") >= 0.5

    def test_real_acronyms_not_flagged(self):
        strategy = ConsonantSequenceStrategy()
        assert strategy.predict_proba("NASA FBI HTTP") < 0.5

    def test_lowercase_equivalent_still_flagged(self):
        strategy = ConsonantSequenceStrategy()
        assert strategy.predict_proba("xkcdqrtzplm blah") >= 0.5


class TestVowelPatternWhitelistExactMatch:
    """Bug 5: substring containment whitelisted any run containing a
    valid pattern (e.g. 'aeiouaeiou' contains 'iou')."""

    def test_long_vowel_soup_flagged(self):
        strategy = VowelPatternStrategy()
        assert strategy.predict_proba("xy aeiouaeiou") >= 0.5

    def test_vowel_run_with_valid_tail_flagged(self):
        strategy = VowelPatternStrategy()
        assert strategy.predict_proba("xy uuuuuaie") >= 0.5

    def test_legitimate_words_not_flagged(self):
        strategy = VowelPatternStrategy()
        assert strategy.predict_proba("queueing sequoia") < 0.5


class TestVowelPatternPerWordRuns:
    """Bug 6: spaces were stripped before run detection, merging vowel
    runs across word boundaries."""

    def test_adjacent_word_vowels_not_merged(self):
        strategy = VowelPatternStrategy()
        assert strategy.predict_proba("Hawaii oasis") < 0.5


class TestDiacriticFolding:
    """Bug 7: accented letters were absent from ASCII n-gram models, so
    normal accented text scored as gibberish."""

    ACCENTED = "café résumé naïve déjà vu"

    def test_markov_accented_text_not_flagged(self):
        strategy = MarkovChainStrategy()
        assert strategy.predict_proba(self.ACCENTED) < 0.5

    def test_ngram_accented_text_not_flagged(self):
        strategy = NGramFrequencyStrategy()
        assert strategy.predict_proba(self.ACCENTED) < 0.5

    def test_markov_gibberish_still_flagged(self):
        strategy = MarkovChainStrategy()
        assert strategy.predict_proba("asdfghjkl") >= 0.5

    def test_markov_normal_text_still_passes(self):
        strategy = MarkovChainStrategy()
        assert strategy.predict_proba("hello world") < 0.5

    def test_ngram_normal_text_still_passes(self):
        strategy = NGramFrequencyStrategy()
        assert strategy.predict_proba("hello world") < 0.5


class TestAffixDetectionGibberish:
    """Bug 8: one suffix-lookalike among 20 gibberish words dropped the
    score to a hard 0.0."""

    GIBBERISH_20 = (
        "rekxqz vbnmpo zxqwer jklmwe pqzxcv mnbvcx qwertz plkjhg "
        "zaqwsx cderfv bgtyhn mjuikl polkmn qazwsx edcrfv tgbyhn "
        "ujmikl olpqaz wsxedc rfvtgb"
    )

    def test_twenty_word_gibberish_flagged(self):
        strategy = AffixDetectionStrategy()
        assert strategy.predict_proba(self.GIBBERISH_20) >= 0.5

    def test_suffix_lookalike_needs_plausible_stem(self):
        strategy = AffixDetectionStrategy()
        # 'zxqwer' ends in 'er' but its stem has no vowel
        assert strategy._has_affix("zxqwer") is False
        assert strategy._has_affix("water") is True


class TestAffixDetectionWeakSignalCap:
    """Bug 9: legitimate zero-affix text (name lists) scored 0.65."""

    NAMES = (
        "John Smith from Texas met Mary Jones near Boston "
        "with Frank Adams today"
    )

    def test_name_list_not_flagged(self):
        strategy = AffixDetectionStrategy()
        assert strategy.predict_proba(self.NAMES) < 0.5

    def test_normal_text_still_passes(self):
        strategy = AffixDetectionStrategy()
        assert strategy.predict_proba(
            "The programming language is incredibly powerful and usable"
        ) < 0.5

    def test_applicable_requires_min_words(self):
        strategy = AffixDetectionStrategy()
        assert strategy.applicable("cat dog run") is False
        assert strategy.applicable(self.NAMES) is True


class TestLetterFrequencyRareLetters:
    """Bug 11: letters with expected count < 1 were skipped, so a
    surplus of j/x/q/z in short text contributed nothing."""

    def test_rare_letter_surplus_registers(self):
        strategy = LetterFrequencyStrategy()
        low = strategy.predict_proba(
            "the quick brown fox jumps over the lazy dog"
        )
        high = strategy.predict_proba("jxqz jjxx qqzz jxjx zqzq jxqz")
        assert high > low
        assert high >= 0.5

    def test_normal_sentences_stay_low(self):
        strategy = LetterFrequencyStrategy()
        assert strategy.predict_proba(NORMAL_PARAGRAPH) < 0.3
        assert strategy.predict_proba(
            "The quick brown fox jumps over the lazy dog every day"
        ) < 0.3


class TestSymbolRatioDigits:
    """Bug 12: digits counted as symbols, flagging phone numbers,
    prices and simple math."""

    def test_phone_number_not_flagged(self):
        strategy = SymbolRatioStrategy()
        assert strategy.predict_proba("Call 555 867 5309") < 0.5

    def test_price_not_flagged(self):
        strategy = SymbolRatioStrategy()
        assert strategy.predict_proba("Price: $19.99") < 0.5

    def test_symbol_soup_still_flagged(self):
        strategy = SymbolRatioStrategy()
        assert strategy.predict_proba("@#$%^&*!@#$") >= 0.5

    def test_count_digits_opt_in(self):
        strategy = SymbolRatioStrategy(count_digits=True)
        assert strategy.predict_proba("1234567890") >= 0.5
