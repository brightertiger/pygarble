"""Immutable finding and report contracts."""

import dataclasses

import pytest

from pygarble.findings import CATEGORIES, Finding, ScanReport, sort_key


def finding(**overrides):
    base = dict(
        category="pii",
        kind="email",
        start=3,
        end=10,
        confidence=0.9,
        reason="structure",
    )
    base.update(overrides)
    return Finding(**base)


def test_categories_are_fixed():
    assert CATEGORIES == ("secrets", "pii", "profanity", "gibberish")


def test_finding_is_frozen_and_has_no_text_field():
    f = finding()
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.kind = "phone"
    assert set(f.to_dict()) == {
        "category",
        "kind",
        "start",
        "end",
        "confidence",
        "reason",
    }


def test_sort_key_orders_by_start_end_category_kind():
    items = [
        finding(start=5, end=9, category="pii", kind="phone"),
        finding(start=5, end=9, category="pii", kind="email"),
        finding(start=1, end=2, category="secrets", kind="jwt"),
        finding(start=5, end=7, category="secrets", kind="jwt"),
    ]
    ordered = sorted(items, key=sort_key)
    assert [(f.start, f.end, f.category, f.kind) for f in ordered] == [
        (1, 2, "secrets", "jwt"),
        (5, 7, "secrets", "jwt"),
        (5, 9, "pii", "email"),
        (5, 9, "pii", "phone"),
    ]


def test_report_helpers():
    report = ScanReport(
        findings=(
            finding(kind="email"),
            finding(category="secrets", kind="jwt", start=20, end=40),
            finding(kind="email", start=50, end=60),
        ),
        flagged=True,
        length=80,
    )
    assert report.kinds() == ("email", "jwt")
    by = report.by_category()
    assert set(by) == {"pii", "secrets"}
    assert len(by["pii"]) == 2
    payload = report.to_dict()
    assert payload["flagged"] is True
    assert payload["length"] == 80
    assert payload["findings"][1]["kind"] == "jwt"
    assert "text" not in payload
