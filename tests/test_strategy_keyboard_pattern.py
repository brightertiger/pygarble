"""KeyboardPattern: repeated bigrams are judged per novel word."""

from pygarble import GarbleDetector, Strategy


def test_repeated_dictionary_words_are_clean():
    detector = GarbleDetector(Strategy.KEYBOARD_PATTERN)
    assert detector.score("Go go go") < 0.5
    assert detector.score("no no no") < 0.5


def test_repeated_bigram_inside_a_novel_word_fires():
    detector = GarbleDetector(Strategy.KEYBOARD_PATTERN)
    assert detector.score("xkxkxkxk") >= 0.5


def test_keyboard_rows_still_fire():
    assert GarbleDetector(Strategy.KEYBOARD_PATTERN).predict("asdfghjkl")
