"""PatternMatching: real words and round numbers are not patterns."""

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "The strengths of the plan are clear",
        "It was worthwhile after all",
        "A nightclub and a birthplace",
        "We raised 10000 dollars for the school",
    ],
)
def test_english_with_consonant_runs_or_round_numbers_is_clean(text):
    assert GarbleDetector(Strategy.PATTERN_MATCHING).score(text) < 0.5
    assert EnsembleDetector(profile="english_extended").predict(text) is False


@pytest.mark.parametrize("text", ["asdfghjkl", "AAAAAAA", "xkrfplmqbvzt"])
def test_true_patterns_still_fire(text):
    assert GarbleDetector(Strategy.PATTERN_MATCHING).predict(text) is True


def test_custom_consonant_cluster_override_is_honoured():
    detector = GarbleDetector(
        Strategy.PATTERN_MATCHING,
        patterns={"consonant_cluster": r"[bcdfghjklmnpqrstvwxz]{3,}"},
    )
    # "xkr" is a novel word with a 3-consonant run under the custom rule
    assert detector.predict("xkr") is True
    # dictionary words are never fed to consonant_cluster
    assert detector.predict("strengths") is False


def test_alternating_pattern_is_letters_only():
    detector = GarbleDetector(Strategy.PATTERN_MATCHING)
    assert detector.score("abababab") >= 0.5
    assert detector.score("0000000123") < 0.5
