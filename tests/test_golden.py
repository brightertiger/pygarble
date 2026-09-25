"""The committed golden corpus must reproduce exactly."""

import pytest


def test_golden_corpus_reproduces():
    golden = pytest.importorskip(
        "regression.golden", reason="regression/ is not shipped in sdist"
    )
    assert golden.render() == golden.OUTPUT.read_text(encoding="utf-8")


def test_golden_rows_cover_every_profile_and_edge_input():
    golden = pytest.importorskip("regression.golden")
    from pygarble.ensemble import PROFILES

    rows = [
        __import__("json").loads(line)
        for line in golden.OUTPUT.read_text(encoding="utf-8").splitlines()
    ]
    assert {r["profile"] for r in rows} == set(PROFILES)
    texts = {r["text"] for r in rows}
    assert set(golden.EDGE_INPUTS) <= texts
    assert len(rows) == len(texts) * len(PROFILES)
