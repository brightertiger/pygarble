"""Code written against pygarble 0.8.0 keeps importing and running."""

import os
import pickle
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy
from pygarble.screening import PIIDetector, ProfanityDetector, SecretsDetector

ROOT = Path(__file__).resolve().parent.parent
# Every strategy class pygarble.core exported in 0.8.0.
STRATEGIES_080 = [
    name + "Strategy"
    for name in (
        "AffixDetection BigramProbability ConsonantSequence EntropyBased "
        "FunctionWordDensity HexString KeyboardAdjacency KeyboardPattern "
        "LetterFrequency LetterPosition LogLikelihoodRatio MarkovChain "
        "Mojibake NGramFrequency PatternMatching Pronouncability "
        "RareTrigram Repetition SymbolRatio UnicodeScript VowelPattern "
        "VowelRatio WordAnomaly WordCollocation WordLookup ZipfConformity"
    ).split()
]


def _removed_080_imports():
    strategies = import_module("pygarble.gibberish.strategies")
    data = import_module("pygarble.data")
    cases = [
        ("pygarble.core", name, getattr(strategies, name), strategies)
        for name in ["BaseStrategy"] + STRATEGIES_080
    ]
    for module in ("markov_chain", "word_anomaly", "log_likelihood_ratio"):
        for name in ("BIGRAM_LOG_PROBS", "DEFAULT_LOG_PROB"):
            cases.append(
                (
                    "pygarble.strategies." + module,
                    name,
                    getattr(data, name),
                    data,
                )
            )
    cases.append(
        (
            "pygarble.strategies.zipf_conformity",
            "FunctionWordDensityStrategy",
            strategies.FunctionWordDensityStrategy,
            strategies,
        )
    )
    return cases


def test_080_import_paths_resolve_with_a_deprecation_warning():
    assert len(STRATEGIES_080) == 26
    for module, name, expected, home in _removed_080_imports():
        with pytest.warns(DeprecationWarning, match=home.__name__):
            namespace = {}
            exec(f"from {module} import {name} as value", namespace)
        assert namespace["value"] is expected, (module, name)
    # Importing never warns; only touching an old name does.
    code = (
        "import pygarble, pygarble.core, pygarble.strategies.markov_chain\n"
        "pygarble.GarbleDetector('markov_chain').predict('Hello world')\n"
    )
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_names_leaked_by_080_imports_still_resolve():
    import concurrent.futures
    import typing
    import unicodedata

    cases = [
        ("pygarble.core", name, getattr(typing, name))
        for name in ("Callable", "Dict", "Optional", "Type", "Union")
    ]
    cases.append(("pygarble.core", "concurrent", concurrent))
    cases.append(("pygarble.strategies.base", "unicodedata", unicodedata))
    for module, name, expected in cases:
        with pytest.warns(DeprecationWarning, match=f"{module}.{name}"):
            namespace = {}
            exec(f"from {module} import {name} as value", namespace)
        assert namespace["value"] is expected, (module, name)
    with pytest.warns(DeprecationWarning):
        pool = import_module("pygarble.core").concurrent.futures
    assert pool.ThreadPoolExecutor is concurrent.futures.ThreadPoolExecutor
    with pytest.raises(AttributeError):
        import_module("pygarble.core").NoSuchName


def test_weighted_default_ensemble_error_points_to_the_legacy_profile():
    with pytest.raises(ValueError) as error:
        EnsembleDetector(voting="weighted", weights=[1, 1, 1])
    message = str(error.value)
    assert message.startswith("weights must have same length as strategies")
    assert "got 3 weights for 6 strategies" in message
    assert "profile='legacy'" in message
    legacy = EnsembleDetector(
        profile="legacy", voting="weighted", weights=[1, 1, 1]
    )
    assert legacy.predict("qxzjkwpv bnmqwer zzxqv") is True
    assert legacy.predict("The weather is lovely today") is False


def test_detectors_pickled_by_080_still_predict():
    path = Path(__file__).parent / "fixtures" / "pickled_080.pkl"
    with path.open("rb") as handle:
        detector, ensemble, strategy = pickle.load(handle)
    assert strategy is Strategy.WORD_LOOKUP
    fresh_detector = GarbleDetector(Strategy.MARKOV_CHAIN)
    fresh_ensemble = EnsembleDetector(profile="legacy")
    texts = ["Hello world", "qxzjkwpv bnmqwer zzxqv", "x" * 50]
    assert detector.predict(texts) == fresh_detector.predict(texts)
    assert ensemble.predict(texts) == fresh_ensemble.predict(texts)
    assert ensemble.predict_proba(texts) == fresh_ensemble.predict_proba(texts)
    assert ensemble.analyze("Hello world") == fresh_ensemble.analyze(
        "Hello world"
    )
    restored = pickle.loads(pickle.dumps(ensemble))
    assert restored.predict(texts) == fresh_ensemble.predict(texts)


def test_strategy_map_is_a_dict_again():
    from pygarble.core import STRATEGY_MAP

    assert isinstance(STRATEGY_MAP, dict)
    assert list(STRATEGY_MAP) == list(Strategy)
    strategies = import_module("pygarble.gibberish.strategies")
    assert STRATEGY_MAP[Strategy.MARKOV_CHAIN] is (
        strategies.MarkovChainStrategy
    )
    copy = dict(STRATEGY_MAP)
    assert copy == STRATEGY_MAP and len(copy) == len(Strategy)
    for module in (
        "pygarble.detector",
        "pygarble.ensemble",
        "pygarble.gibberish.detector",
        "pygarble.gibberish.ensemble",
    ):
        assert import_module(module).STRATEGY_MAP is STRATEGY_MAP, module
    for module in ("pygarble.gibberish.registry", "pygarble.gibberish.core"):
        assert "STRATEGY_MAP" in dir(import_module(module)), module


def test_patched_strategy_map_entry_is_used_by_detectors(monkeypatch):
    from pygarble.core import STRATEGY_MAP
    from pygarble.gibberish.analysis import Evidence

    class AlwaysGarbled(STRATEGY_MAP[Strategy.VOWEL_RATIO]):
        def evaluate(self, features):
            return Evidence(1.0)

    text = "Hello world"
    assert GarbleDetector(Strategy.VOWEL_RATIO).predict(text) is False
    monkeypatch.setitem(STRATEGY_MAP, Strategy.VOWEL_RATIO, AlwaysGarbled)
    assert GarbleDetector(Strategy.VOWEL_RATIO).predict(text) is True
    ensemble = EnsembleDetector(strategies=[Strategy.VOWEL_RATIO])
    assert ensemble.predict(text) is True


def test_wrong_strategy_and_name_arguments_say_what_to_pass():
    for given in ([None], (s for s in [None])):
        with pytest.raises(ValueError, match="None"):
            EnsembleDetector(strategies=given)
    with pytest.raises(TypeError, match="pass a list"):
        EnsembleDetector("markov_chain")
    for build in (
        lambda: SecretsDetector(kinds="jwt"),
        lambda: SecretsDetector(exclude_kinds="jwt"),
        lambda: ProfanityDetector(allowlist="damn"),
        lambda: PIIDetector(kinds="email"),
    ):
        with pytest.raises(TypeError, match="pass a list") as error:
            build()
        # Callers that caught the former ValueError keep working.
        assert isinstance(error.value, ValueError)
