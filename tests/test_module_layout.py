"""Canonical modules and old pointers must share behavior and state."""

import base64
import json
import os
import pickle
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

import pygarble
from pygarble import gibberish, screening
from pygarble.gibberish.registry import STRATEGY_MAP, Strategy
from pygarble.gibberish.strategies import _EXPORTS

ROOT = Path(__file__).resolve().parent.parent
_GIBBERISH_MODULES = (
    "analysis",
    "calibration",
    "core",
    "detector",
    "ensemble",
    "options",
    "preprocessing",
    "registry",
    "scoring",
)
_SCREENING_MODULES = (
    "pii",
    "pii.checksums",
    "pii.patterns",
    "profanity",
    "profanity.normalize",
    "profanity.wordlist",
    "secrets",
    "secrets.entropy",
    "secrets.patterns",
)
_MODULE_PAIRS = (
    [
        ("pygarble." + name, "pygarble.gibberish." + name)
        for name in _GIBBERISH_MODULES
    ]
    + [
        ("pygarble." + name, "pygarble.screening." + name)
        for name in _SCREENING_MODULES
    ]
    + [
        (
            "pygarble.strategies." + name,
            "pygarble.gibberish.strategies." + name,
        )
        for name in sorted(set(_EXPORTS.values()))
    ]
)


def _run_in_checkout(code):
    # Import this checkout's pygarble, never an installed copy.
    check = (
        "\nimport pathlib, pygarble\n"
        f"root = pathlib.Path({str(ROOT)!r})\n"
        "assert root in pathlib.Path(pygarble.__file__).resolve().parents, "
        "pygarble.__file__\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code + check],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
    )


@pytest.mark.parametrize("canonical_first", [False, True])
def test_old_and_new_modules_are_identical_in_either_import_order(
    canonical_first,
):
    # Fresh processes catch duplicate modules hidden by test collection.
    code = f"""
from importlib import import_module
pairs = {_MODULE_PAIRS!r}
for old, new in pairs:
    first, second = (new, old) if {canonical_first!r} else (old, new)
    a = import_module(first)
    b = import_module(second)
    assert a is b, (old, new)
    parent, _, child = old.rpartition('.')
    assert getattr(import_module(parent), child) is a, old
"""
    result = _run_in_checkout(code)
    assert result.returncode == 0, result.stderr


def test_public_exports_and_strategy_registry_share_classes():
    for name in gibberish.__all__:
        if name != "STRATEGY_MAP":
            assert getattr(pygarble, name) is getattr(gibberish, name)
        assert name in dir(gibberish)
    for name in ("PIIDetector", "ProfanityDetector", "SecretsDetector"):
        assert getattr(pygarble, name) is getattr(screening, name)
    legacy = import_module("pygarble.strategies")
    canonical = import_module("pygarble.gibberish.strategies")
    for name in canonical.__all__:
        assert getattr(legacy, name) is getattr(canonical, name)
        assert name in dir(legacy)
    old_registry = import_module("pygarble.registry")
    assert old_registry.Strategy is Strategy
    assert old_registry.STRATEGY_MAP is STRATEGY_MAP
    for strategy, cls in STRATEGY_MAP.items():
        detector = gibberish.GarbleDetector(strategy)
        assert type(detector._strategy_instance) is cls


def test_patching_legacy_module_changes_detector_globals(monkeypatch):
    old = import_module("pygarble.pii")
    canonical = import_module("pygarble.screening.pii")
    assert old._COMPILED is canonical._COMPILED
    monkeypatch.setattr(old, "_validate", lambda *_: None)
    assert canonical.PIIDetector().detect("mail jane@example.com") == ()


def test_pickles_created_before_the_move_still_work():
    # Produced from main at 7a3ff893a26659e7b15763e3d2427a93c6383d39.
    path = Path(__file__).parent / "fixtures" / "legacy_layout_pickles.json"
    objects = {
        name: pickle.loads(base64.b64decode(value))
        for name, value in json.loads(path.read_text()).items()
    }
    assert objects["strategy"] is Strategy.VOWEL_RATIO
    assert objects["span"] == gibberish.Span(0, 3, "example")
    for name, current in (
        ("detector", gibberish.GarbleDetector(Strategy.VOWEL_RATIO)),
        ("ensemble", gibberish.EnsembleDetector()),
    ):
        old = objects[name]
        assert type(old) is type(current)
        for text in (
            "Hello world",
            "qxzjkwpv bnmqwer zzxqv",
            "hello\x00world",
        ):
            assert old.predict(text) == current.predict(text)
            assert old.analyze(text) == current.analyze(text)
    for name, cls in (
        ("pii", screening.PIIDetector),
        ("profanity", screening.ProfanityDetector),
        ("secrets", screening.SecretsDetector),
    ):
        old = objects[name]
        assert type(old) is cls
        text = "mail jane@example.com; damn; key AKIAIOSFODNN7EXAMPLE"
        assert old.detect(text) == cls().detect(text)
    for obj in objects.values():
        assert type(pickle.loads(pickle.dumps(obj))) is type(obj)


@pytest.mark.parametrize("module", ["pygarble", "pygarble.gibberish"])
def test_gibberish_exports_keep_strategies_lazy(module):
    code = f"""
import sys
from importlib import import_module
api = import_module({module!r})
assert 'pygarble.gibberish.ensemble' not in sys.modules
assert 'pygarble.gibberish.strategies' not in sys.modules
d = api.GarbleDetector('vowel_ratio')
assert not d.predict('Hello world')
assert 'pygarble.data.words' not in sys.modules
assert 'pygarble.data.reference' not in sys.modules
assert 'pygarble.data.ngram_ranks' not in sys.modules
assert 'pygarble.data.calibration' not in sys.modules
assert 'pygarble.gibberish.strategies.markov_chain' not in sys.modules
"""
    result = _run_in_checkout(code)
    assert result.returncode == 0, result.stderr
