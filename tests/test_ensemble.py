"""EnsembleDetector construction contracts."""

import warnings

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy


def test_unknown_setting_is_a_future_warning_at_the_call_site():
    with pytest.warns(FutureWarning, match="min_lenght") as record:
        GarbleDetector(Strategy.MARKOV_CHAIN, min_lenght=8)
    assert len(record) == 1
    assert record[0].filename == __file__


def test_shared_kwargs_reach_only_strategies_that_accept_them():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detector = EnsembleDetector(profile="english", min_word_length=3)
    by_name = {d.strategy: d for d in detector._detectors}
    assert by_name[Strategy.WORD_ANOMALY].kwargs == {"min_word_length": 3}
    assert by_name[Strategy.MARKOV_CHAIN].kwargs == {}
    assert detector.kwargs == {"min_word_length": 3}


def test_kwarg_no_member_accepts_warns_once():
    with pytest.warns(FutureWarning, match="min_lenght") as record:
        EnsembleDetector(profile="english", min_lenght=3)
    assert len(record) == 1
    assert record[0].filename == __file__


def test_strategy_kwargs_unknown_key_warns_once_at_call_site():
    with pytest.warns(FutureWarning, match="typo") as record:
        EnsembleDetector(
            profile="english",
            strategy_kwargs={Strategy.MARKOV_CHAIN: {"typo": 1}},
        )
    assert len(record) == 1
    assert record[0].filename == __file__


def test_strategy_kwargs_are_retained_for_introspection():
    detector = EnsembleDetector(
        strategies=[Strategy.MARKOV_CHAIN],
        strategy_kwargs={Strategy.MARKOV_CHAIN: {"min_length": 6}},
    )
    assert detector.strategy_kwargs == {
        Strategy.MARKOV_CHAIN: {"min_length": 6}
    }
