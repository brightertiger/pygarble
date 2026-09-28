"""JSON data tables must equal the Python tables byte-for-byte in content."""

import json
from pathlib import Path

import pytest

from pygarble.data import (
    BIGRAM_LOG_PROBS,
    COMMON_TRIGRAMS,
    DEFAULT_LOG_PROB,
    ENGLISH_WORDS,
    NGRAM_RANKS,
    REFERENCE_WORDS,
    SCORE_NULL_TAILS,
    STATISTIC_NULL,
    TAIL_GRID,
)

STATISTICS = (
    "cross_parsing",
    "primed_compression",
    "ngram_rank",
    "permutation_test",
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


def test_reference_json_matches():
    assert load("reference.json") == list(REFERENCE_WORDS)


def test_reference_words_fit_dictionary_and_are_english():
    assert len(" ".join(REFERENCE_WORDS).encode("ascii")) <= 32768
    assert len(set(REFERENCE_WORDS)) == len(REFERENCE_WORDS)
    assert set(REFERENCE_WORDS) <= ENGLISH_WORDS
    assert REFERENCE_WORDS[0] == "the"


def test_ngram_ranks_json_matches():
    assert load("ngram_ranks.json") == list(NGRAM_RANKS)


def test_ngram_ranks_shape():
    assert len(NGRAM_RANKS) == 1000
    assert len(set(NGRAM_RANKS)) == 1000
    assert all(1 <= len(gram) <= 3 for gram in NGRAM_RANKS)
    assert NGRAM_RANKS[0] == " "


def test_calibration_json_matches():
    table = load("calibration.json")
    assert table["tail_grid"] == list(TAIL_GRID)
    assert table["statistic_null"] == {
        name: [list(pair) for pair in rows]
        for name, rows in STATISTIC_NULL.items()
    }
    assert table["score_null_tails"] == {
        name: list(row) for name, row in SCORE_NULL_TAILS.items()
    }


def test_statistic_null_shape():
    assert set(STATISTIC_NULL) == set(STATISTICS)
    for rows in STATISTIC_NULL.values():
        assert len(rows) == 4
        for median, q99 in rows:
            assert q99 > median
    assert TAIL_GRID == (
        0.5,
        0.25,
        0.1,
        0.05,
        0.025,
        0.01,
        0.005,
        0.0025,
        0.001,
    )


def test_manifest_lists_json_files():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert {
        "words.json",
        "bigrams.json",
        "trigrams.json",
        "reference.json",
        "ngram_ranks.json",
        "calibration.json",
    } <= set(manifest["files"])
    counts = manifest["counts"]
    assert counts["reference_words"] == len(REFERENCE_WORDS)
    assert counts["ngram_ranks"] == len(NGRAM_RANKS)
    assert counts["score_null_texts"] == 12000


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
