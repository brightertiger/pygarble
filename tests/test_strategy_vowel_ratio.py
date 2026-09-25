"""VowelRatio: parameters are validated; tiny inputs abstain."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_vowel_ratio": 5.0},
        {"max_vowel_ratio": -1},
        {"consonant_cluster_len": -3},
        {"min_vowel_ratio": 0.7, "max_vowel_ratio": 0.6},
        {"min_length": 0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GarbleDetector(Strategy.VOWEL_RATIO, **kwargs)


@pytest.mark.parametrize("text", ["a", "I", "Hmm", "Shh", "Mr. Ng"])
def test_tiny_inputs_abstain(text):
    detector = GarbleDetector(Strategy.VOWEL_RATIO)
    assert detector.predict(text) is False
    assert detector.analyze(text).status == "insufficient_evidence"


def test_real_signals_still_fire():
    detector = GarbleDetector(Strategy.VOWEL_RATIO)
    assert detector.predict("bcdfghjklmnpqrstvwxyz") is True
    assert detector.predict("aeiouaeiou") is True
    assert detector.predict("hello world") is False
