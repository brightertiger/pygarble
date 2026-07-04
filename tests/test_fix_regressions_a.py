"""
Regression tests for verified bug fixes in:

- MojibakeStrategy (accented text false positives, cp1252 patterns)
- UnicodeScriptStrategy (Japanese kanji/kana mixing, whole-word
  foreign text vs in-word homoglyphs)
- LetterPositionStrategy (proper nouns / acronyms, q-final words)
- HexStringStrategy (base64 false positives, digit-only runs,
  pooled hex ratio)
- PronouncabilityStrategy (y-as-vowel, valid onset clusters)
- WordCollocationStrategy (name lists / headlines, contractions,
  applicable() gating)
"""

import pytest

from pygarble import GarbleDetector, Strategy


class TestMojibakeRegressions:
    """Accented text is not mojibake; cp1252 mojibake is."""

    @pytest.mark.parametrize("text", [
        "café", "naïve", "déjà vu", "über schön", "résumé",
    ])
    def test_accented_text_not_flagged(self, text):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        assert detector.predict_proba(text) < 0.2
        assert detector.predict(text) is False

    @pytest.mark.parametrize("text", [
        "cafÃ©",
        "cafÃ© rÃ©sumÃ©",
        "donâ€™t worry itâ€™s fine",
    ])
    def test_real_mojibake_still_flagged(self, text):
        detector = GarbleDetector(Strategy.MOJIBAKE)
        assert detector.predict_proba(text) >= 0.5
        assert detector.predict(text) is True


class TestUnicodeScriptRegressions:
    """Japanese and cross-word script mixing are legitimate."""

    @pytest.mark.parametrize("text", [
        "これは日本語のテキストです",  # kanji + kana
        "食べる",
    ])
    def test_japanese_not_flagged(self, text):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        assert detector.predict_proba(text) < 0.2

    @pytest.mark.parametrize("text", [
        "Привет hello",
        "Спасибо thanks",
    ])
    def test_whole_word_foreign_text_not_flagged(self, text):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        assert detector.predict_proba(text) < 0.2

    def test_in_word_homoglyph_still_flagged(self):
        detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
        # Cyrillic 'а' (U+0430) inside a Latin word
        assert detector.predict_proba("pаypal") >= 0.5


class TestLetterPositionRegressions:
    """Proper nouns and acronyms must not be flagged."""

    @pytest.mark.parametrize("text", [
        "McDonald", "Nguyen", "Iraq", "FBI", "Dvorak",
        "Mrs Nguyen met Mr McDonald in Iraq",
    ])
    def test_proper_nouns_not_flagged(self, text):
        detector = GarbleDetector(Strategy.LETTER_POSITION)
        assert detector.predict_proba(text) < 0.5
        assert detector.predict(text) is False

    @pytest.mark.parametrize("text", [
        "vbnmpo zxqwer wqpzj",
        "zxqwj vbnmk",
        "xjword bwtext",
    ])
    def test_gibberish_still_flagged(self, text):
        detector = GarbleDetector(Strategy.LETTER_POSITION)
        assert detector.predict_proba(text) >= 0.5


class TestHexStringRegressions:
    """Long words and digit-only runs are not hex/base64."""

    @pytest.mark.parametrize("text", [
        "internationalization", "electroencephalography",
    ])
    def test_long_words_not_base64(self, text):
        detector = GarbleDetector(Strategy.HEX_STRING)
        assert detector.predict_proba(text) < 0.2

    def test_real_base64_still_flagged(self):
        detector = GarbleDetector(Strategy.HEX_STRING)
        assert detector.predict_proba(
            "aGVsbG8gd29ybGQgdGhpcyBpcw=="
        ) >= 0.5

    @pytest.mark.parametrize("text", [
        "4111111111111111",       # card-number-like
        "12345678901234567890",   # digit run
    ])
    def test_digit_runs_not_hex(self, text):
        detector = GarbleDetector(Strategy.HEX_STRING)
        assert detector.predict_proba(text) < 0.5

    def test_hex_run_still_flagged(self):
        detector = GarbleDetector(Strategy.HEX_STRING)
        assert detector.predict_proba(
            "4f8a9b2c1d3e5f6a7b8c9d0e"
        ) >= 0.5

    def test_hexy_english_words_not_pooled(self):
        detector = GarbleDetector(Strategy.HEX_STRING)
        assert detector.predict_proba(
            "deed dad added a bad decade face"
        ) < 0.2

    @pytest.mark.parametrize("text", [
        "5d41402abc4b2a76b9719d911017c592",  # MD5
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855",  # SHA256
    ])
    def test_real_hashes_still_flagged(self, text):
        detector = GarbleDetector(Strategy.HEX_STRING)
        assert detector.predict_proba(text) >= 0.7


class TestPronounceabilityRegressions:
    """'y' acts as a vowel in non-initial positions."""

    @pytest.mark.parametrize("text", [
        "rhythm",
        "sync systems byte myth",
        "MY GYM RHYTHM MYTHS",
        "pneumonia pneumatic patients treated",
    ])
    def test_y_words_and_valid_onsets_not_flagged(self, text):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        assert detector.predict_proba(text) < 0.5
        assert detector.predict(text) is False

    @pytest.mark.parametrize("text", [
        "xkcd qwfp zxcv",
        "xkcd qwfp zxcv bkpt",
        "bcdfghjklmnp qrstvwxyz",
    ])
    def test_gibberish_still_flagged(self, text):
        detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
        assert detector.predict_proba(text) >= 0.5


class TestWordCollocationRegressions:
    """Name lists / headlines are collocation-free but legitimate."""

    NAME_LIST = (
        "John Smith Mary Johnson Robert Williams Patricia Brown "
        "Michael Davis Linda Miller William Wilson Elizabeth Moore "
        "David Taylor Barbara Anderson"
    )
    HEADLINE = (
        "Fed Signals Rate Cuts While Markets Rally Sharply Amid "
        "Renewed Investor Optimism Over Global Trade Talks "
        "Progress Report Today"
    )

    def test_name_list_not_flagged(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        assert detector.predict_proba(self.NAME_LIST) < 0.5

    def test_headline_not_flagged(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        assert detector.predict_proba(self.HEADLINE) < 0.5

    def test_contractions_stay_whole(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        text = (
            "don't worry it's fine because they will not be there "
            "until later tonight anyway my friend"
        )
        assert detector.predict_proba(text) < 0.5

    def test_gibberish_still_flagged(self):
        detector = GarbleDetector(Strategy.WORD_COLLOCATION)
        text = (
            "xkrf plmq bvzt nwsd jghc trbn mkpl wqzd lpnr fvxt "
            "qzml hkrp"
        )
        assert detector.predict_proba(text) >= 0.5

    def test_applicable_gates_short_text(self):
        from pygarble.strategies.word_collocation import (
            WordCollocationStrategy,
        )
        strategy = WordCollocationStrategy()
        assert strategy.applicable("hello world") is False
        assert strategy.applicable(
            "one two three four five six seven eight"
        ) is True
