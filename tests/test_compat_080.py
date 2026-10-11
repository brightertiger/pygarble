"""Code written against pygarble 0.8.0 keeps importing and running."""

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

from pygarble import Strategy

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
    # Incidental 0.8.0 names stay gone.
    with pytest.raises(AttributeError):
        import_module("pygarble.core").Callable
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
