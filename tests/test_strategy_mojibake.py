"""Mojibake: replacement characters count at any length."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize("text", ["�", "a�", "ab�"])
def test_replacement_char_is_detected_regardless_of_length(text):
    detector = GarbleDetector(Strategy.MOJIBAKE)
    assert detector.score(text) >= 0.8
    assert detector.predict(text) is True


def test_replacement_check_can_be_disabled():
    detector = GarbleDetector(Strategy.MOJIBAKE, check_replacement_char=False)
    assert detector.score("a�") == 0.0


@pytest.mark.parametrize("text", ["Café crème brûlée", "Sí, señor", "NÃO É"])
def test_accented_text_is_clean(text):
    assert GarbleDetector(Strategy.MOJIBAKE).predict(text) is False
