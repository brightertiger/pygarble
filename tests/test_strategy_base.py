"""Direct BaseStrategy use goes through the same evaluate() contract."""

from pygarble.strategies import (
    FunctionWordDensityStrategy,
    PronouncabilityStrategy,
    PronounceabilityStrategy,
)


def test_legacy_predict_proba_respects_applicability():
    strategy = FunctionWordDensityStrategy()
    assert strategy.applicable("hi") is False
    assert strategy.predict_proba("hi") == 0.0
    assert strategy.predict("hi") is False


def test_legacy_predict_agrees_with_evaluate():
    from pygarble.preprocessing import TextFeatures

    strategy = FunctionWordDensityStrategy()
    text = "xkrf plmq bvzt nwsd jghc trbn mkpl qwer asdf zxcv poiu lkjh mnbv"
    assert (
        strategy.predict_proba(text)
        == strategy.evaluate(TextFeatures(text)).score
    )


def test_pronounceability_alias_is_same_class():
    assert PronounceabilityStrategy is PronouncabilityStrategy
