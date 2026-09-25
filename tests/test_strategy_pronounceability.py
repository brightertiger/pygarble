"""Pronounceability: real onsets are valid; min_word_length is honoured."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize(
    "word",
    [
        "squonk",
        "squib",
        "sphinxes",
        "chlorinate",
        "schlep",
        "ghoulish",
        "sclerotic",
        "phlox",
    ],
)
def test_valid_english_onsets_are_not_violations(word):
    assert GarbleDetector(Strategy.PRONOUNCEABILITY).score(word) < 0.5


def test_min_word_length_is_honoured():
    strict = GarbleDetector(Strategy.PRONOUNCEABILITY, min_word_length=2)
    lenient = GarbleDetector(Strategy.PRONOUNCEABILITY)
    assert strict.score("xkq bkx") > 0.0
    assert lenient.score("xkq bkx") == 0.0


def test_gibberish_still_fires():
    assert GarbleDetector(Strategy.PRONOUNCEABILITY).predict("bkxq tpfk vzjk")


def test_dead_helper_removed():
    from pygarble.strategies.pronounceability import PronouncabilityStrategy

    assert not hasattr(PronouncabilityStrategy, "_extract_consonant_clusters")
