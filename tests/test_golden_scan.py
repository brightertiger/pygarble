"""The committed golden scan corpus must reproduce exactly."""

import hashlib
import json

import pytest


def test_golden_scan_corpus_reproduces():
    golden_scan = pytest.importorskip(
        "regression.golden_scan", reason="regression/ is not shipped in sdist"
    )
    assert golden_scan.render() == golden_scan.OUTPUT.read_text(
        encoding="utf-8"
    )


def test_golden_scan_checksum_matches():
    golden_scan = pytest.importorskip("regression.golden_scan")
    content = golden_scan.OUTPUT.read_text(encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert digest == golden_scan.CHECKSUM.read_text().strip()


def test_golden_scan_rows_carry_no_text_and_record_options():
    golden_scan = pytest.importorskip("regression.golden_scan")
    rows = [
        json.loads(line)
        for line in golden_scan.OUTPUT.read_text(encoding="utf-8").splitlines()
    ]
    keys = {"source", "sha256", "length", "options", "flagged", "findings"}
    assert all(set(r) == keys for r in rows)
    assert len(rows) == len(golden_scan.texts())
    assert any(r["options"] for r in rows)
    assert {r["source"] for r in rows} >= {"vector", "golden", "prose.txt"}
