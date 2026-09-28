"""Fisher voting: tail p-values, their combination and english_fusion."""

import math
import subprocess
import sys
import warnings

import pytest

from pygarble import EnsembleDetector, Strategy
from pygarble.data import SCORE_NULL_TAILS, TAIL_GRID
from pygarble.ensemble import PROFILES
from pygarble.gibberish.analysis import Evidence
from pygarble.gibberish.fisher import (
    combined_p_value,
    fisher_score,
    tail_p_value,
)

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FUSION = [
    Strategy.WORD_LOOKUP,
    Strategy.LOG_LIKELIHOOD_RATIO,
    Strategy.CROSS_PARSING,
]
ENGLISH = [
    "The committee will meet again next week.",
    "Please send the signed contract to our office before Friday.",
    (
        "The weather has been unusually warm this autumn, and many "
        "gardeners are still harvesting tomatoes and beans from their "
        "plots while the evenings slowly grow shorter."
    ),
    (
        "Sure. To rotate the logs, set the handler to RotatingFileHandler "
        "with a maximum size of ten megabytes and keep five backups. "
        "Restart the service afterwards and confirm that the new file is "
        "being written. If nothing appears, check the permissions first."
    ),
]
INVENTED = (
    "glorb biga nuba sinja dabba horp minbo samboz sillinth manj hoopi "
    "alganth morphobon nila napt"
)
MASH = "asdkfj qwpoeiru zxmcnv lkjasdf poiuqwer mnbvzx qwerlkj asdfpoiu"


def test_tail_p_value_walks_the_grid():
    assert tail_p_value(0.05, THRESHOLDS, TAIL_GRID) == 1.0
    assert tail_p_value(0.15, THRESHOLDS, TAIL_GRID) == 0.5
    assert tail_p_value(0.45, THRESHOLDS, TAIL_GRID) == 0.05
    assert tail_p_value(0.95, THRESHOLDS, TAIL_GRID) == 0.001


def test_tail_p_value_is_strict_at_a_threshold():
    assert tail_p_value(0.1, THRESHOLDS, TAIL_GRID) == 1.0
    assert tail_p_value(0.4, THRESHOLDS, TAIL_GRID) == 0.1
    # A mass point at zero never counts as extreme.
    assert tail_p_value(0.0, (0.0,) * 9, TAIL_GRID) == 1.0


def test_tail_p_value_stops_at_first_failure():
    assert tail_p_value(0.5, (0.1, 0.9, 0.2), TAIL_GRID) == 0.5


def test_empty_tail_table_gives_one():
    assert tail_p_value(1.0, (), TAIL_GRID) == 1.0


def test_single_member_at_alpha_scores_one_half():
    p = combined_p_value([0.001])
    assert p == 0.001
    assert fisher_score(p, 0.001) == 0.5


def test_two_members_match_the_hand_computed_value():
    x = -2 * 2 * math.log(0.01)
    assert x == pytest.approx(18.420680743952367)
    expected = 0.0001 * (1 + 9.210340371976184)
    assert combined_p_value([0.01, 0.01]) == pytest.approx(expected)
    assert combined_p_value([0.01, 0.01]) == pytest.approx(0.00102103404)


def test_three_members_use_the_full_series():
    half = -3 * math.log(0.1)
    expected = 0.001 * (1 + half + half**2 / 2)
    assert combined_p_value([0.1, 0.1, 0.1]) == pytest.approx(expected)


def test_quiet_members_score_zero():
    assert combined_p_value([1.0, 1.0, 1.0]) == 1.0
    assert fisher_score(1.0, 0.001) == 0.0


def test_combined_p_value_is_capped_at_one():
    assert combined_p_value([1.0] * 32) <= 1.0


def test_fisher_score_decides_at_alpha():
    for alpha in (0.001, 0.01, 0.05):
        assert fisher_score(alpha, alpha) == pytest.approx(0.5)
        assert fisher_score(alpha / 2, alpha) > 0.5
        assert fisher_score(alpha * 2, alpha) < 0.5


def test_higher_member_scores_never_lower_the_combined_score():
    steps = [i / 20 for i in range(21)]
    for other in (0.05, 0.35, 0.95):
        scores = [
            fisher_score(
                combined_p_value(
                    [
                        tail_p_value(x, THRESHOLDS, TAIL_GRID),
                        tail_p_value(other, THRESHOLDS, TAIL_GRID),
                    ]
                ),
                0.001,
            )
            for x in steps
        ]
        assert scores == sorted(scores)


def test_score_null_tails_cover_every_strategy():
    assert list(SCORE_NULL_TAILS) == [s.value for s in Strategy]
    for name, row in SCORE_NULL_TAILS.items():
        assert len(row) in (0, len(TAIL_GRID)), name
        assert list(row) == sorted(row), name
        assert all(0.0 <= value <= 1.0 for value in row), name


def test_fusion_members_have_informative_tails():
    for strategy in FUSION:
        row = SCORE_NULL_TAILS[strategy.value]
        assert row[-1] > row[0]


def test_english_fusion_profile():
    assert PROFILES["english_fusion"] == tuple(FUSION)
    detector = EnsembleDetector(profile="english_fusion")
    assert detector.voting == "fisher"
    assert detector.fisher_alpha == 0.001
    assert detector.strategies == FUSION


def test_existing_profiles_keep_members_and_voting():
    legacy = (
        Strategy.MARKOV_CHAIN,
        Strategy.LOG_LIKELIHOOD_RATIO,
        Strategy.WORD_ANOMALY,
    )
    expected = {
        "english": legacy
        + (
            Strategy.MOJIBAKE,
            Strategy.KEYBOARD_ADJACENCY,
            Strategy.CONTROL_CHARACTERS,
        ),
        "english_extended": legacy
        + (
            Strategy.MOJIBAKE,
            Strategy.KEYBOARD_ADJACENCY,
            Strategy.CONTROL_CHARACTERS,
            Strategy.PATTERN_MATCHING,
            Strategy.LOCAL_ANOMALY,
            Strategy.REPETITION,
        ),
        "legacy": legacy,
        "corruption": (Strategy.MOJIBAKE, Strategy.CONTROL_CHARACTERS),
        "spoofing": (Strategy.UNICODE_SCRIPT,),
        "llm_output": (
            Strategy.REPETITION,
            Strategy.CONTROL_CHARACTERS,
            Strategy.MOJIBAKE,
            Strategy.LOCAL_ANOMALY,
        ),
    }
    assert {
        name: members
        for name, members in PROFILES.items()
        if name != "english_fusion"
    } == expected
    for name in expected:
        assert EnsembleDetector(profile=name).voting == "any"
    default = EnsembleDetector()
    assert default.profile == "english"
    assert default.voting == "any"
    assert tuple(default.strategies) == expected["english"]


@pytest.mark.parametrize("text", ENGLISH)
def test_english_fusion_keeps_english_clean(text):
    analysis = EnsembleDetector(profile="english_fusion").analyze(text)
    assert analysis.status == "clean"
    assert analysis.score < 0.5


@pytest.mark.parametrize("text", [INVENTED, MASH])
def test_english_fusion_flags_invented_text_and_mash(text):
    assert len(text) >= 60
    analysis = EnsembleDetector(profile="english_fusion").analyze(text)
    assert analysis.garbled
    assert analysis.status == "garbled"
    assert analysis.score >= 0.5


def test_english_fusion_short_text_uses_applicable_members_only():
    detector = EnsembleDetector(profile="english_fusion")
    analysis = detector.analyze("hi")
    # Cross parsing needs eight normalised characters; word lookup has no
    # minimum, so only empty input leaves the profile without evidence.
    assert [s.applicable for s in analysis.signals] == [True, True, False]
    assert analysis.status == "clean"
    for text in ("", "   "):
        analysis = detector.analyze(text)
        assert not analysis.garbled
        assert analysis.status == "insufficient_evidence"
        assert analysis.score == 0.0
        assert not any(signal.applicable for signal in analysis.signals)


def test_english_fusion_analysis_reports_every_member():
    analysis = EnsembleDetector(profile="english_fusion").analyze(INVENTED)
    assert analysis.profile == "english_fusion"
    assert [s.strategy for s in analysis.signals] == [s.value for s in FUSION]


def test_fisher_score_combines_member_tails():
    detector = EnsembleDetector(profile="english_fusion")
    analysis = detector.analyze(MASH)
    p_values = [
        tail_p_value(s.score, SCORE_NULL_TAILS[s.strategy], TAIL_GRID)
        for s in analysis.signals
        if s.applicable
    ]
    assert analysis.score == fisher_score(combined_p_value(p_values), 0.001)


def test_fisher_predict_score_and_analyze_agree():
    detector = EnsembleDetector(profile="english_fusion")
    texts = ENGLISH + [INVENTED, MASH, "hi", ""]
    analyses = detector.analyze(texts)
    assert detector.predict(texts) == [a.garbled for a in analyses]
    assert detector.score(texts) == [a.score for a in analyses]
    assert detector.predict_proba(texts) == [a.score for a in analyses]
    for text, analysis in zip(texts, analyses):
        assert detector.predict(text) is analysis.garbled


def test_fisher_threshold_applies_to_the_score():
    loose = EnsembleDetector(profile="english_fusion", threshold=0.0)
    assert loose.predict(ENGLISH[0])
    strict = EnsembleDetector(profile="english_fusion", threshold=1.0)
    assert not strict.predict(MASH)


def test_fisher_alpha_moves_the_decision():
    detector = EnsembleDetector(
        strategies=[Strategy.WORD_LOOKUP], voting="fisher"
    )
    detector._detectors[0]._strategy_instance.evaluate = lambda features: (
        Evidence(1.0)
    )
    # A score above every threshold takes the smallest tail probability
    # the table can support.
    row = SCORE_NULL_TAILS["word_lookup"]
    p = TAIL_GRID[sum(1.0 > t for t in row) - 1]
    assert detector.analyze("anything").score == fisher_score(p, 0.001)
    lenient = EnsembleDetector(
        strategies=[Strategy.WORD_LOOKUP], voting="fisher", fisher_alpha=p
    )
    lenient._detectors[0]._strategy_instance.evaluate = lambda features: (
        Evidence(1.0)
    )
    assert lenient.analyze("anything").score == 0.5
    assert lenient.predict("anything")


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5, float("nan")])
def test_fisher_alpha_must_be_inside_the_unit_interval(alpha):
    with pytest.raises(ValueError, match="fisher_alpha"):
        EnsembleDetector(profile="english_fusion", fisher_alpha=alpha)


@pytest.mark.parametrize(
    "voting", ["majority", "any", "all", "average", "weighted"]
)
def test_fisher_alpha_without_fisher_voting_warns(voting):
    options = {"weights": [1.0] * 6} if voting == "weighted" else {}
    with pytest.warns(FutureWarning, match="fisher_alpha") as record:
        detector = EnsembleDetector(
            voting=voting, fisher_alpha=0.01, **options
        )
    assert len(record) == 1
    assert record[0].filename == __file__
    assert detector.voting == voting


def test_weights_with_fisher_voting_warn_and_are_ignored():
    with pytest.warns(FutureWarning, match="weights"):
        weighted = EnsembleDetector(
            profile="english_fusion", weights=[5.0, 0.0, 1.0]
        )
    plain = EnsembleDetector(profile="english_fusion")
    for text in (MASH, ENGLISH[1]):
        assert weighted.analyze(text) == plain.analyze(text)


def test_explicit_voting_overrides_the_profile_default():
    detector = EnsembleDetector(profile="english_fusion", voting="any")
    assert detector.voting == "any"
    analysis = detector.analyze(MASH)
    assert analysis.score == max(s.score for s in analysis.signals)


def test_fisher_voting_on_an_existing_profile():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detector = EnsembleDetector(profile="english", voting="fisher")
    assert detector.profile == "english"
    assert not detector.predict(ENGLISH[1])
    assert detector.predict(MASH)


@pytest.mark.parametrize("strategy", list(Strategy))
def test_fisher_voting_accepts_any_strategy(strategy):
    detector = EnsembleDetector(strategies=[strategy], voting="fisher")
    assert detector.profile == "custom"
    for text in (ENGLISH[1], MASH, "hi"):
        analysis = detector.analyze(text)
        assert 0.0 <= analysis.score <= 1.0
        assert detector.predict(text) is analysis.garbled


def test_fisher_voting_without_applicable_members():
    detector = EnsembleDetector(
        strategies=[Strategy.CROSS_PARSING, Strategy.WORD_LOOKUP],
        voting="fisher",
    )
    analysis = detector.analyze("")
    assert analysis.status == "insufficient_evidence"
    assert analysis.score == 0.0
    assert not detector.predict("")


def test_tail_tables_load_only_for_fisher_voting():
    code = """
import sys
from pygarble import EnsembleDetector
for profile in ("english", "english_extended", "legacy", "corruption",
                "spoofing", "llm_output"):
    EnsembleDetector(profile=profile).analyze("hello world qxzjkwpv")
EnsembleDetector(
    strategies=["markov_chain", "word_lookup"], voting="average"
).analyze("hello world")
assert 'pygarble.data.calibration' not in sys.modules
EnsembleDetector(strategies=["word_lookup"], voting="fisher")
assert 'pygarble.data.calibration' in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
