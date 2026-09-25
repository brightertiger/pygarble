"""EnsembleDetector construction contracts."""

import warnings

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy
from pygarble.ensemble import PROFILES


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


@pytest.mark.parametrize("weights", [[1, 0], [2, 1]])
@pytest.mark.parametrize("voting", ["majority", "any", "all", "average"])
def test_weights_without_weighted_voting_is_an_error(voting, weights):
    with pytest.raises(ValueError, match="weights"):
        EnsembleDetector(
            strategies=[Strategy.MARKOV_CHAIN, Strategy.WORD_ANOMALY],
            voting=voting,
            weights=weights,
        )


def test_strategy_kwargs_accepts_string_keys():
    detector = EnsembleDetector(
        strategies=[Strategy.MARKOV_CHAIN],
        strategy_kwargs={"markov_chain": {"min_length": 6}},
    )
    assert detector._detectors[0]._strategy_instance.min_length == 6


def test_detector_accepts_strategy_name_string():
    assert GarbleDetector("markov_chain").strategy is Strategy.MARKOV_CHAIN
    with pytest.raises(ValueError):
        GarbleDetector("no_such_strategy")


@pytest.mark.parametrize(
    "kwargs", [{"threshold": 10**400}, {"threads": 10**400}]
)
def test_huge_ints_raise_value_error_not_overflow(kwargs):
    with pytest.raises(ValueError):
        GarbleDetector(Strategy.MARKOV_CHAIN, **kwargs)


def test_weighted_mean_matches_plain_formula():
    detector = EnsembleDetector(
        strategies=[Strategy.MARKOV_CHAIN, Strategy.WORD_ANOMALY],
        voting="weighted",
        weights=[3, 1],
    )
    analysis = detector.analyze("hello qxzjkwpv")
    scores = {s.strategy: s.score for s in analysis.signals if s.applicable}
    expected = (3 * scores["markov_chain"] + 1 * scores["word_anomaly"]) / 4
    assert analysis.score == pytest.approx(expected)


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_profile_constructs_and_votes_any(profile):
    detector = EnsembleDetector(profile=profile)
    assert detector.profile == profile
    assert detector.voting == "any"
    assert [d.strategy for d in detector._detectors] == list(PROFILES[profile])


def test_custom_strategy_list_votes_majority():
    detector = EnsembleDetector(strategies=list(PROFILES["english"]))
    assert detector.profile == "custom"
    assert detector.voting == "majority"


def test_unknown_profile_and_profile_plus_strategies_are_errors():
    with pytest.raises(ValueError, match="unknown profile"):
        EnsembleDetector(profile="nope")
    with pytest.raises(ValueError, match="either profile or strategies"):
        EnsembleDetector(profile="english", strategies=[Strategy.MARKOV_CHAIN])


@pytest.mark.parametrize(
    "profile,text,expected",
    [
        ("english", "The quick brown fox jumps over the lazy dog", False),
        ("english", "qxzjkwpv bnmqwer zxcvbnm", True),
        ("english_extended", "The strengths of the plan are clear", False),
        ("legacy", "hello world", False),
        ("corruption", "CafÃ© crÃ¨me", True),
        ("corruption", "Café crème", False),
        ("spoofing", "pаypal login", True),
        ("spoofing", "paypal login", False),
    ],
)
def test_profile_decisions(profile, text, expected):
    assert EnsembleDetector(profile=profile).predict(text) is expected
