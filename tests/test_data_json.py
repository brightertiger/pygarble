"""JSON data tables must equal the Python tables byte-for-byte in content."""

import json
from pathlib import Path

import pytest

from pygarble.data import (
    BIGRAM_LOG_PROBS,
    COMMON_TRIGRAMS,
    DEFAULT_LOG_PROB,
    ENGLISH_WORDS,
)

DATA = Path(__file__).resolve().parent.parent / "pygarble" / "data"


def load(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not present (wheel install)")
    return json.loads(path.read_text(encoding="utf-8"))


def test_words_json_matches():
    words = load("words.json")
    assert words == sorted(words)
    assert set(words) == set(ENGLISH_WORDS)


def test_bigrams_json_matches():
    table = load("bigrams.json")
    assert table["default_log_prob"] == DEFAULT_LOG_PROB
    assert table["log_probs"] == dict(BIGRAM_LOG_PROBS)


def test_trigrams_json_matches():
    trigrams = load("trigrams.json")
    assert trigrams == sorted(trigrams)
    assert set(trigrams) == set(COMMON_TRIGRAMS)


def test_manifest_lists_json_files():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert {"words.json", "bigrams.json", "trigrams.json"} <= set(
        manifest["files"]
    )


def test_rule_tables_match_python_sources():
    from pygarble.pii.patterns import export as pii_export
    from pygarble.profanity.wordlist import export as profanity_export
    from pygarble.secrets.patterns import export as secrets_export

    for name, export in (
        ("secrets.json", secrets_export),
        ("pii.json", pii_export),
        ("profanity.json", profanity_export),
    ):
        path = DATA / name
        if not path.is_file():
            pytest.skip(f"{name} is not shipped in wheels")
        assert json.loads(path.read_text(encoding="utf-8")) == export()


def test_manifest_hashes_rule_tables():
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name in ("secrets.json", "pii.json", "profanity.json"):
        if (DATA / name).is_file():
            assert name in manifest["files"]
