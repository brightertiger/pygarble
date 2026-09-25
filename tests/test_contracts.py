"""Contracts every registered strategy must satisfy."""

import inspect
import re

import pytest

from pygarble import GarbleDetector, Strategy
from pygarble.preprocessing import ascii_alpha_words
from pygarble.strategies import rare_trigram


def test_ascii_alpha_words_matches_legacy_regex():
    text = "Hello, wörld! it's 42 x-ray"
    assert ascii_alpha_words(text) == [
        "hello",
        "w",
        "rld",
        "it",
        "s",
        "x",
        "ray",
    ]


def test_rare_trigram_source_lists_each_entry_once():
    # IMPOSSIBLE_TRIGRAMS is a set literal, so duplicates collapse at
    # runtime; guard the source text instead.
    source = inspect.getsource(rare_trigram)
    entries = re.findall(r'"([a-z]{3})"', source)
    dupes = {e for e in entries if entries.count(e) > 1}
    assert not dupes


ALL = list(Strategy)


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.value)
@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\t", "a", "👍👏🙏", "你好世界", "Привет мир", "x" * 20000],
    ids=["empty", "spaces", "ws", "one", "emoji", "cjk", "cyrillic", "long"],
)
def test_every_strategy_survives_edge_inputs(strategy, text):
    detector = GarbleDetector(strategy)
    score = detector.score(text)
    assert 0.0 <= score <= 1.0
    assert detector.predict(text) is (score >= 0.5)
    analysis = detector.analyze(text)
    assert analysis.score == score
    if not text.strip():
        assert analysis.status == "insufficient_evidence"


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.value)
def test_every_strategy_rejects_non_strings(strategy):
    detector = GarbleDetector(strategy)
    for bad in [None, 0, b"x", [1]]:
        with pytest.raises(TypeError):
            detector.predict(bad)


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.value)
def test_every_strategy_is_listed_in_options(strategy):
    from pygarble.options import PARAMETERS
    from pygarble.registry import STRATEGY_MAP

    assert STRATEGY_MAP[strategy].__name__ in PARAMETERS


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.value)
def test_instances_do_not_share_state(strategy):
    a = GarbleDetector(strategy, allowlist=["qxzjkwpv"])
    b = GarbleDetector(strategy)
    assert a.allowlist != b.allowlist
    assert a._strategy_instance is not b._strategy_instance
