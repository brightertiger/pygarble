"""The committed golden corpus must reproduce exactly."""

import json
import unicodedata

import pytest


def test_golden_corpus_reproduces():
    golden = pytest.importorskip(
        "paper.regression.golden",
        reason="paper/regression/ is not shipped in sdist",
    )
    assert golden.render() == golden.OUTPUT.read_text(encoding="utf-8")


def test_golden_rows_cover_every_profile_and_edge_input():
    golden = pytest.importorskip("paper.regression.golden")
    from pygarble.ensemble import PROFILES

    rows = [
        json.loads(line)
        for line in golden.OUTPUT.read_text(encoding="utf-8").splitlines()
    ]
    assert {r["profile"] for r in rows} == set(PROFILES)
    texts = {r["text"] for r in rows}
    assert set(golden.EDGE_INPUTS) <= texts
    assert len(rows) == len(texts) * len(PROFILES)


def has_decomposed_latin(text):
    """A combining mark directly after an ASCII letter, as in e + U+0301."""
    return any(
        unicodedata.combining(mark) and base.isascii() and base.isalpha()
        for base, mark in zip(text, text[1:])
    )


def test_golden_texts_include_decomposed_combining_marks():
    # Devanagari viramas elsewhere are combining too; a Latin base is where
    # code-point and grapheme offsets of a port most often diverge.
    golden = pytest.importorskip("paper.regression.golden")
    texts = {
        json.loads(line)["text"]
        for line in golden.OUTPUT.read_text(encoding="utf-8").splitlines()
    }
    assert any(has_decomposed_latin(t) for t in texts)


def test_golden_pins_spans_across_a_combining_mark():
    # A span ending after e + U+0301 fixes whether a port counts the mark
    # as its own code point.
    golden = pytest.importorskip("paper.regression.golden")
    rows = [
        json.loads(line)
        for line in golden.OUTPUT.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 918
    assert any("\u0301" in r["text"] and r["spans"] for r in rows)
