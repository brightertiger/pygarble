"""Allowlist must suppress evidence in every strategy, not just four."""

import unicodedata

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy
from pygarble.preprocessing import TextFeatures

TEXT = "please ping asdfgh about the deploy"


def test_scrubbed_blanks_allowlisted_tokens_and_keeps_offsets():
    features = TextFeatures(TEXT, frozenset({"asdfgh"}))
    assert len(features.scrubbed) == len(TEXT)
    assert features.scrubbed == "please ping        about the deploy"
    assert TextFeatures(TEXT).scrubbed == TEXT


def test_scrubbed_handles_punctuation_attached_to_token():
    text = "ping asdfgh, then (asdfgh) deploy"
    features = TextFeatures(text, frozenset({"asdfgh"}))
    assert features.scrubbed == "ping       , then (      ) deploy"


@pytest.mark.parametrize(
    "strategy",
    [
        Strategy.KEYBOARD_ADJACENCY,
        Strategy.KEYBOARD_PATTERN,
        Strategy.PATTERN_MATCHING,
        Strategy.PRONOUNCEABILITY,
        Strategy.NGRAM_FREQUENCY,
    ],
)
def test_allowlist_suppresses_text_level_strategies(strategy):
    assert GarbleDetector(strategy).predict(TEXT) is True
    allowed = GarbleDetector(strategy, allowlist=["asdfgh"])
    assert allowed.predict(TEXT) is False
    assert allowed.analyze(TEXT).score < 0.5


def test_default_profile_honours_allowlist():
    assert EnsembleDetector().predict(TEXT) is True
    detector = EnsembleDetector(allowlist=["asdfgh"])
    assert detector.predict(TEXT) is False
    assert detector.analyze(TEXT).garbled is False


def test_extended_profile_honours_allowlist():
    text = "the xqzvkj report was fine"
    detector = EnsembleDetector(
        profile="english_extended", allowlist=["xqzvkj"]
    )
    assert detector.analyze(text).garbled is False


def test_fully_allowlisted_text_is_insufficient_evidence():
    detector = GarbleDetector(
        Strategy.KEYBOARD_ADJACENCY, allowlist=["asdfgh"]
    )
    result = detector.analyze("asdfgh asdfgh")
    assert result.garbled is False
    assert result.status == "insufficient_evidence"


def test_scrubbed_blanks_composed_and_decomposed_forms():
    for form in ("NFC", "NFD"):
        text = unicodedata.normalize(form, "café ok")
        scrubbed = TextFeatures(text, frozenset({"cafe"})).scrubbed
        assert len(scrubbed) == len(text)
        assert scrubbed.strip() == "ok"


@pytest.mark.parametrize(
    "text, word",
    [
        ("#!? asdfghjkl", "asdfghjkl"),
        ("(x) => {asdfghjkl}", "asdfghjkl"),
        ("ok -> Kubernetes!!", "Kubernetes"),
    ],
)
def test_allowlist_never_raises_symbol_ratio(text, word):
    plain = GarbleDetector(Strategy.SYMBOL_RATIO)
    allowed = GarbleDetector(Strategy.SYMBOL_RATIO, allowlist=[word])
    if plain.predict(text) is False:
        assert allowed.predict(text) is not True
    assert allowed.analyze(text).score <= plain.analyze(text).score
