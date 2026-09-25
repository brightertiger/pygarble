"""Repetition: digits and short emphasis are not repetition evidence."""

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "One million is written 1000000",
        "very very very good",
        "Order 000000123 shipped",
    ],
)
def test_digit_runs_and_short_emphasis_are_clean(text):
    assert GarbleDetector(Strategy.REPETITION).score(text) < 0.5
    assert EnsembleDetector(profile="english_extended").predict(text) is False


@pytest.mark.parametrize(
    "text", ["aaaaaaaaaa", "abababababab", "test test test", "no no no no no"]
)
def test_documented_repetition_still_fires(text):
    assert GarbleDetector(Strategy.REPETITION).predict(text) is True
