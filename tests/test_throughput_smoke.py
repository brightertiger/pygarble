"""Loose timing guards: Scanner stays fast and linear on long input."""

import time

import pytest

from pygarble import Scanner

RULES = ["secrets", "pii", "profanity"]


def test_dense_nhs_phone_overlap_filter_scales_linearly():
    scanner = Scanner(categories=["pii"], locales=["uk"])
    line = "943 476 5919; +14155550123\n"
    scanner.scan(line * 100)
    timings = []
    for count in (1000, 4000):
        text = line * count
        start = time.perf_counter()
        report = scanner.scan(text)
        timings.append(time.perf_counter() - start)
        assert len(report.findings) == 2 * count
    assert timings[1] / max(timings[0], 1e-3) < 8


def test_rules_only_throughput_smoke():
    throughput = pytest.importorskip(
        "paper.regression.throughput",
        reason="paper/regression/ is not shipped in sdist",
    )
    lines = throughput.build_corpus(200_000)
    result = throughput.measure(RULES, lines)
    assert result["seconds"] < 10
    assert result["findings"] > 0


def test_build_corpus_is_deterministic():
    throughput = pytest.importorskip("paper.regression.throughput")
    assert throughput.build_corpus(20_000) == throughput.build_corpus(20_000)


def test_chunked_corpus_joins_the_same_lines():
    throughput = pytest.importorskip("paper.regression.throughput")
    lines = throughput.build_corpus(50_000)
    chunks = throughput.build_corpus(50_000, chunk_bytes=4096)
    assert "\n".join(chunks) == "\n".join(lines)
    assert len(chunks) < len(lines)
    assert all(len(chunk.encode("utf-8")) < 4096 + 200 for chunk in chunks)


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
    _scan_seconds(scanner, 10_000)  # untimed warm-up: imports and caches
    small = _scan_seconds(scanner, 100_000)
    large = _scan_seconds(scanner, 1_000_000)
    assert large / max(small, 1e-3) < 20
    assert large < 30


def _many_findings(kind, lines):
    import hashlib

    if kind == "keyword":
        token = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
        return "\n".join(
            f"token={token} other_secret=q8Zt3vP2xL9mK4nR{i}"
            for i in range(lines)
        )
    return "\n".join(
        hashlib.sha256(str(i).encode()).hexdigest() for i in range(lines)
    )


@pytest.mark.parametrize("kind", ["keyword", "hashes"])
def test_many_secret_findings_scale_linearly(kind):
    # Overlap filtering between finding lists must not be O(n * m).
    scanner = Scanner(categories=["secrets"], secrets_without_context=True)
    scanner.scan(_many_findings(kind, 200))  # untimed warm-up
    timings = []
    for lines in (2_000, 8_000):
        text = _many_findings(kind, lines)
        start = time.perf_counter()
        report = scanner.scan(text)
        timings.append(time.perf_counter() - start)
        assert len(report.findings) >= lines
    small, large = timings
    assert large / max(small, 1e-3) < 8
    assert large < 5
