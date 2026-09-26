"""Word list invariants: normalised, deduplicated, attributed, safe."""

import re

from pygarble.data import ENGLISH_WORDS
from pygarble.profanity.normalize import normalize_token
from pygarble.profanity.wordlist import (
    ATTRIBUTION,
    EMBEDDED,
    PHRASES,
    PROFANITY_MILD,
    PROFANITY_STRONG,
    export,
)

LETTERS = re.compile(r"^[a-z]+$")

# The frequency list includes adult vocabulary. These entries contain an
# EMBEDDED word but are profanity themselves, not clean collisions.
ADULT_ENTRIES = {"assfuck", "assfucked", "assfucking", "collegefuckfest"}


def test_lists_are_normalised_sorted_and_disjoint():
    for words in (PROFANITY_STRONG, PROFANITY_MILD):
        assert list(words) == sorted(set(words))
        for word in words:
            assert LETTERS.match(word), word
            assert normalize_token(word) == word, word
    assert not set(PROFANITY_STRONG) & set(PROFANITY_MILD)
    assert len(PROFANITY_STRONG) >= 80 and len(PROFANITY_MILD) >= 8


def test_phrases_are_tuples_of_normalised_words():
    for phrase in PHRASES:
        assert isinstance(phrase, tuple) and len(phrase) >= 2
        assert all(LETTERS.match(w) for w in phrase), phrase


def test_embedded_words_are_strong_and_not_inside_english_words():
    profane = set(PROFANITY_STRONG) | set(PROFANITY_MILD) | ADULT_ENTRIES
    clean = [english for english in ENGLISH_WORDS if english not in profane]
    for word in EMBEDDED:
        assert word in PROFANITY_STRONG and len(word) >= 4
        assert not any(word in english for english in clean), word


def test_no_ordinary_english_word_in_lists():
    # Words that appear in ordinary text must never be listed outright.
    for common in ["hell", "sex", "nude", "breast", "god", "screw", "bloody"]:
        assert common not in PROFANITY_STRONG, common
    for common in ["hell", "god", "sex", "nude"]:
        assert common not in PROFANITY_MILD, common


def test_attribution_and_export():
    assert "LDNOOBW" in ATTRIBUTION and "CC-BY-4.0" in ATTRIBUTION
    payload = export()
    assert set(payload) == {
        "strong",
        "mild",
        "phrases",
        "embedded",
        "leet_map",
        "attribution",
    }
    assert payload["strong"] == list(PROFANITY_STRONG)
