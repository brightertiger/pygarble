"""Non-parametric strategies: windowed English-reference statistics."""

import pytest

from pygarble import GarbleDetector, Strategy
from pygarble.preprocessing import TextFeatures
from pygarble.registry import STRATEGY_MAP

NONPARAMETRIC = [
    Strategy.CROSS_PARSING,
    Strategy.PRIMED_COMPRESSION,
    Strategy.NGRAM_RANK,
    Strategy.PERMUTATION_TEST,
]

ENGLISH = [
    "Thank you for the quick reply.",
    "The meeting has been moved to Thursday afternoon.",
    "Please send me the report before the end of the week so that I can "
    "review it.",
    "We walked along the river after dinner and talked about the summer, "
    "the new house and what the children would do when school started "
    "again.",
]

INVENTED = (
    "glorb biga nuba sinja dabba horp minbo samboz sillinth manj hoopi "
    "alganth morphobon nila napt morja hoppa norla sappa nubi sanos"
)

# Letter order barely differs from its own shuffles in pronounceable
# invented text, so the permutation test needs less English-like pairs:
# it scores INVENTED (128 characters) at about 0.26.
INVENTED_DOUBLED = (
    "ukka tavo pleem zirra gonfu yaxel mibbo tuzza kwelo farn dibbo zeeka "
    "loppu vint oggu pazzi reemo kuvva zolla yubbe teevo naxxa"
)


def _invented(strategy):
    if strategy is Strategy.PERMUTATION_TEST:
        return INVENTED_DOUBLED
    return INVENTED


MASH = (
    "asdkfj qwpeoriu zxmcnv lkjhsdf poiuqwe mnbvzx asdlkfj qwerpoiu " "hjklgf"
)

LONG_ENGLISH = (
    "The library opens at nine in the morning and closes at six in the "
    "evening. Most visitors come to read the newspapers, borrow a few "
    "books for the week, or use the computers to look for work. On "
    "Saturdays the children's room is full of families, and a volunteer "
    "reads stories to the youngest ones while their parents choose books "
    "of their own. The staff are friendly and will help anyone who asks, "
    "whether they need a novel, a map of the town or advice on writing a "
    "letter. "
)

REASONS = {
    Strategy.CROSS_PARSING: "cross_parsing_rate",
    Strategy.PRIMED_COMPRESSION: "primed_compression_ratio",
    Strategy.NGRAM_RANK: "ngram_rank_distance",
    Strategy.PERMUTATION_TEST: "permutation_gap",
}


def _repeat(text, length):
    return (text * (length // len(text) + 1))[:length].rsplit(" ", 1)[0]


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
@pytest.mark.parametrize("text", ENGLISH)
def test_clean_english_scores_low(strategy, text):
    detector = GarbleDetector(strategy)
    assert detector.score(text) < 0.5
    assert detector.predict(text) is False
    analysis = detector.analyze(text)
    assert analysis.status == "clean"
    assert analysis.signals[0].reason == REASONS[strategy]


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
def test_pronounceable_invented_text_scores_high(strategy):
    detector = GarbleDetector(strategy)
    text = _invented(strategy)
    assert detector.score(text) > 0.5
    assert detector.predict(text) is True


@pytest.mark.parametrize(
    "strategy",
    [
        Strategy.CROSS_PARSING,
        Strategy.PRIMED_COMPRESSION,
        Strategy.NGRAM_RANK,
    ],
    ids=lambda s: s.value,
)
def test_sixty_characters_of_invented_text_score_high(strategy):
    text = INVENTED[:65]
    assert len(text) == 65 and text.endswith("hoopi")
    assert GarbleDetector(strategy).score(text) > 0.5


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
def test_keyboard_mash_scores_high(strategy):
    assert len(MASH) >= 60
    assert GarbleDetector(strategy).score(MASH) > 0.5


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
@pytest.mark.parametrize("text", ["hi you", "a b c d", "Zork!"])
def test_short_text_is_insufficient_evidence(strategy, text):
    analysis = GarbleDetector(strategy).analyze(text)
    assert analysis.status == "insufficient_evidence"
    assert analysis.garbled is False
    assert analysis.signals[0].reason == "insufficient_text"


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
def test_structured_only_text_is_not_applicable(strategy):
    detector = GarbleDetector(strategy)
    text = "https://example.com/a1b2 12345 v2.0 1.4.2 2024-01-31"
    assert detector.applicable(text) is False
    assert detector.analyze(text).status == "insufficient_evidence"
    assert STRATEGY_MAP[strategy]().applicable(text) is False
    assert STRATEGY_MAP[strategy]().applicable(ENGLISH[1]) is True


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
def test_allowlisted_word_does_not_contribute(strategy):
    text = "please read the notes on xqzvkjwp before the meeting on friday"
    without = "please read the notes on before the meeting on friday"
    allowed = GarbleDetector(strategy, allowlist=["xqzvkjwp"])
    plain = GarbleDetector(strategy)
    assert allowed.score(text) == plain.score(without)
    assert plain.score(text) > allowed.score(text)


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
def test_long_text_is_scored_through_windows(strategy):
    detector = GarbleDetector(strategy)
    english = _repeat(LONG_ENGLISH, 2000)
    invented = _repeat(_invented(strategy) + " ", 2000)
    assert len(english) > 1900 and len(invented) > 1900
    assert detector.score(english) < 0.5
    assert detector.score(invented) > 0.5
    # The median over windows ignores one invented window among English.
    mixed = _repeat(LONG_ENGLISH, 1000) + " " + _invented(strategy)
    assert detector.score(mixed) < 0.5


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
def test_scores_are_bounded_and_deterministic(strategy):
    for text in ENGLISH + [INVENTED, MASH, _repeat(INVENTED + " ", 2000)]:
        first = GarbleDetector(strategy).score(text)
        second = GarbleDetector(strategy).score(text)
        assert 0.0 <= first <= 1.0
        assert first == second


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
def test_evaluate_matches_predict_proba(strategy):
    instance = STRATEGY_MAP[strategy]()
    evidence = instance.evaluate(TextFeatures(INVENTED))
    assert evidence.applicable is True
    assert evidence.reason == REASONS[strategy]
    assert instance.predict_proba(INVENTED) == evidence.score


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
def test_midpoint_and_scale_shape_the_score(strategy):
    base = GarbleDetector(strategy).score(MASH)
    assert base > 0.5
    assert GarbleDetector(strategy, midpoint=50.0).score(MASH) < base
    steep = GarbleDetector(strategy, scale=8.0).score(MASH)
    assert steep >= base


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
def test_min_length_controls_applicability(strategy):
    text = "the cat sat on the mat"
    assert GarbleDetector(strategy).applicable(text) is True
    assert GarbleDetector(strategy, min_length=40).applicable(text) is False


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
@pytest.mark.parametrize(
    "options",
    [
        {"scale": 0},
        {"scale": -1},
        {"scale": float("nan")},
        {"midpoint": float("inf")},
        {"min_length": 0},
        {"min_length": 2.5},
    ],
)
def test_options_validate(strategy, options):
    with pytest.raises(ValueError):
        GarbleDetector(strategy, strategy_kwargs=options)


@pytest.mark.parametrize("strategy", NONPARAMETRIC, ids=lambda s: s.value)
def test_unknown_settings_warn(strategy):
    with pytest.warns(FutureWarning, match="shufles"):
        GarbleDetector(strategy, shufles=4)


@pytest.mark.parametrize(
    "strategy",
    [
        Strategy.CROSS_PARSING,
        Strategy.PRIMED_COMPRESSION,
        Strategy.NGRAM_RANK,
    ],
    ids=lambda s: s.value,
)
def test_only_the_permutation_test_accepts_shuffles(strategy):
    with pytest.warns(FutureWarning, match="shuffles"):
        GarbleDetector(strategy, shuffles=4)


@pytest.mark.parametrize("shuffles", [0, -1, 2.0, True])
def test_permutation_shuffles_validate(shuffles):
    with pytest.raises(ValueError):
        GarbleDetector(Strategy.PERMUTATION_TEST, shuffles=shuffles)


def test_permutation_shuffles_change_the_score():
    default = GarbleDetector(Strategy.PERMUTATION_TEST).score(INVENTED)
    fewer = GarbleDetector(Strategy.PERMUTATION_TEST, shuffles=1)
    assert 0.0 <= fewer.score(INVENTED) <= 1.0
    assert fewer.score(INVENTED) != default


def test_primed_compression_documents_zlib_dependence():
    doc = STRATEGY_MAP[Strategy.PRIMED_COMPRESSION].__doc__
    assert "zlib" in doc


def test_strategy_names_and_order():
    assert [s.value for s in list(Strategy)[-4:]] == [
        "cross_parsing",
        "primed_compression",
        "ngram_rank",
        "permutation_test",
    ]
    assert [STRATEGY_MAP[s].__name__ for s in NONPARAMETRIC] == [
        "CrossParsingStrategy",
        "PrimedCompressionStrategy",
        "NGramRankStrategy",
        "PermutationTestStrategy",
    ]
