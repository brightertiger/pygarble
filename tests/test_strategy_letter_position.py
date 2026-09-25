"""LetterPosition: dictionary words and acronyms are exempt."""

import pytest

from pygarble import GarbleDetector, Strategy
from pygarble.strategies.letter_position import LetterPositionStrategy


@pytest.mark.parametrize(
    "text",
    [
        "Czech svelte vlog",
        "Save as PDF or JPG",
        "speed 60 mph at 3000 rpm",
        "Gnocchi for dinner",
    ],
)
def test_real_words_and_acronyms_are_clean(text):
    assert GarbleDetector(Strategy.LETTER_POSITION).predict(text) is False


def test_novel_violations_still_fire():
    detector = GarbleDetector(Strategy.LETTER_POSITION)
    assert detector.predict("wordj endq") is True
    assert detector.predict("xjword bwtext") is True


def test_zero_threshold_is_rejected():
    # GarbleDetector's own ``threshold`` is the decision cut-off, where
    # 0.0 is legal; the strategy threshold arrives via strategy_kwargs.
    with pytest.raises(ValueError, match="threshold"):
        GarbleDetector(
            Strategy.LETTER_POSITION, strategy_kwargs={"threshold": 0.0}
        )
    with pytest.raises(ValueError, match="threshold"):
        LetterPositionStrategy(threshold=0.0)
