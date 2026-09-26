"""Every vector reproduces exactly; every kind has positives and negatives."""

import json
from pathlib import Path

import pytest

from pygarble import Scanner
from pygarble.scanner import _all_kinds

VECTORS = (
    Path(__file__).resolve().parent.parent
    / "paper"
    / "regression"
    / "scan_vectors.json"
)

if not VECTORS.is_file():
    pytest.skip(
        "paper/regression/scan_vectors.json not present",
        allow_module_level=True,
    )

ROWS = json.loads(VECTORS.read_text(encoding="utf-8"))


@pytest.mark.parametrize("row", ROWS, ids=[r["note"] for r in ROWS])
def test_vector_reproduces(row):
    # "options" carries Scanner keywords a row needs, e.g. high_entropy_string
    # only fires with secrets_without_context=True.
    report = Scanner(
        categories=row["categories"], **row.get("options", {})
    ).scan(row["text"])
    assert [f.to_dict() for f in report.findings] == row["expected"]


def test_every_kind_has_three_positives_and_three_negatives():
    universe = set().union(*_all_kinds().values())
    positives = {kind: 0 for kind in universe}
    negatives = {kind: 0 for kind in universe}
    for row in ROWS:
        kinds = {f["kind"] for f in row["expected"]}
        for kind in universe:
            category = next(c for c, ks in _all_kinds().items() if kind in ks)
            if category not in row["categories"]:
                continue
            if kind in kinds:
                positives[kind] += 1
            else:
                negatives[kind] += 1
    short = {
        k: (positives[k], negatives[k])
        for k in universe
        if positives[k] < 3 or negatives[k] < 3
    }
    assert not short, short
