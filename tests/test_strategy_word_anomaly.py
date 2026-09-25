"""WordAnomaly: dictionary acronyms are not anomalies."""

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "Please enable DHCP on the router",
        "Our auditor is KPMG",
        "Watch it on HGTV tonight",
        "Use JDBC to connect",
    ],
)
def test_dictionary_acronyms_are_clean(text):
    assert GarbleDetector(Strategy.WORD_ANOMALY).predict(text) is False
    assert EnsembleDetector().predict(text) is False


def test_single_mashed_token_still_registers():
    detector = GarbleDetector(Strategy.WORD_ANOMALY)
    assert detector.predict("order confirmed asdkjfhq thanks") is True
    assert detector.predict("order confirmed successfully thanks") is False


def test_fraction_is_over_all_scoreable_words():
    detector = GarbleDetector(Strategy.WORD_ANOMALY)
    # 1 bad of 4 words * weight 2.0 = 0.5
    assert detector.score("order confirmed asdkjfhq thanks") == pytest.approx(
        0.5
    )


def test_structured_only_text_is_not_applicable():
    detector = GarbleDetector(Strategy.WORD_ANOMALY)
    result = detector.analyze("https://example.com/a1b2 12345 v2.0")
    assert result.status == "insufficient_evidence"
