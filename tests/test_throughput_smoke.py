"""Loose timing guards: Scanner stays fast and linear on long input."""

import time

import pytest

from pygarble import Scanner

RULES = ["secrets", "pii", "profanity"]


def test_rules_only_throughput_smoke():
    throughput = pytest.importorskip(
        "regression.throughput", reason="regression/ is not shipped in sdist"
    )
    lines = throughput.build_corpus(200_000)
    result = throughput.measure(RULES, lines)
    assert result["seconds"] < 10
    assert result["findings"] > 0


def test_build_corpus_is_deterministic():
    throughput = pytest.importorskip("regression.throughput")
    assert throughput.build_corpus(20_000) == throughput.build_corpus(20_000)


def _scan_seconds(scanner, size):
    sentence = (
        "The quarterly report covers revenue, churn and the hiring plan "
        "for the next two quarters. "
    )
    text = sentence * (size // len(sentence) + 1)
    start = time.perf_counter()
    scanner.scan(text)
    return time.perf_counter() - start


def test_one_megabyte_line_scans_in_linear_time():
    # Ratio, not an absolute bound, so slow CI runners and coverage do not
    # flake; ten times the input must cost well under 20 times the time.
    scanner = Scanner(categories=RULES)
    small = _scan_seconds(scanner, 100_000)
    large = _scan_seconds(scanner, 1_000_000)
    assert large / max(small, 1e-3) < 20
    assert large < 30
