"""Scanner composition, gibberish category, caching and validation."""

import os
import sys

import pytest

import pygarble
from pygarble import Finding, Redaction, Scanner, ScanReport, redact, scan
from pygarble.scanner import DEFAULT_CATEGORIES

AWS = "AKIAIOSFODNN7EXAMPLE"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_default_categories():
    assert DEFAULT_CATEGORIES == ("secrets", "pii", "profanity", "gibberish")


def test_scan_secret_and_gibberish_together():
    report = Scanner().scan(f"key {AWS} qxzjkwpv bnmqwer zzxqv")
    assert isinstance(report, ScanReport)
    kinds = [f.kind for f in report.findings]
    assert "aws_access_key_id" in kinds
    assert "garbled" in kinds
    garbled = [f for f in report.findings if f.kind == "garbled"][0]
    assert garbled.category == "gibberish"
    assert (garbled.start, garbled.end) == (0, report.length)
    assert 0.0 <= garbled.confidence <= 1.0
    assert report.flagged


def test_gibberish_absent_when_clean():
    report = Scanner(categories=["gibberish"]).scan("hello world again")
    assert report.findings == () and report.flagged is False


def test_scan_empty_and_whitespace():
    for text in ["", "   ", "\n\t"]:
        report = scan(text)
        assert report.findings == ()
        assert report.flagged is False
        assert report.length == len(text)
        assert redact(text).text == text


def test_min_confidence_controls_flagged_not_findings():
    text = "password = 'q8Zt3vP2xL9mK4nR'"
    strict = Scanner(categories=["secrets"], min_confidence=0.7).scan(text)
    assert strict.findings and strict.flagged is False
    loose = Scanner(categories=["secrets"], min_confidence=0.6).scan(text)
    assert loose.flagged is True


def test_redact_skips_gibberish_and_low_confidence():
    text = f"token {AWS} password = 'q8Zt3vP2xL9mK4nR' qxzjkwpv"
    out = Scanner(min_confidence=0.7).redact(text)
    assert isinstance(out, Redaction)
    assert "[AWS_ACCESS_KEY_ID]" in out.text
    assert "q8Zt3vP2xL9mK4nR" in out.text
    assert "qxzjkwpv" in out.text


def test_redact_modes_and_category_subset():
    text = f"token {AWS}"
    assert Scanner().redact(text, mode="mask").text == "token " + "*" * 20
    assert Scanner().redact(text, categories=["pii"]).text == text
    with pytest.raises(ValueError, match="mode"):
        Scanner().redact(text, mode="shred")


def test_batch_and_iter_match_single():
    texts = [f"a {AWS}", "hello world", ""]
    scanner = Scanner(categories=["secrets", "gibberish"])
    singles = [scanner.scan(t) for t in texts]
    assert scanner.scan_batch(texts) == singles
    assert list(scanner.iter_scan(iter(texts))) == singles


def test_validation_errors():
    with pytest.raises(ValueError, match="unknown category"):
        Scanner(categories=["toxicity"])
    with pytest.raises(ValueError, match="unknown kind"):
        Scanner(kinds=["nope"])
    with pytest.raises(ValueError, match="min_confidence"):
        Scanner(min_confidence=1.5)
    with pytest.raises(ValueError, match="at least one category"):
        Scanner(categories=[])
    with pytest.raises(TypeError):
        Scanner().scan(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Scanner().scan_batch("not a list")  # type: ignore[arg-type]


def test_kinds_filter_restricts_across_categories():
    report = Scanner(kinds=["garbled"]).scan(f"{AWS} qxzjkwpv bnmqwer")
    assert [f.kind for f in report.findings] == ["garbled"]


def test_max_input_length_applies():
    with pytest.raises(ValueError, match="max_input_length"):
        Scanner(max_input_length=5).scan("abcdefgh")


def test_module_level_cache_reuses_scanner():
    from pygarble import scanner as module

    module._CACHE.clear()
    scan("x", categories=["secrets"])
    scan("y", categories=["secrets"])
    assert len(module._CACHE) == 1
    scan("z", categories=["secrets"], min_confidence=0.9)
    assert len(module._CACHE) == 2


def test_gibberish_options_forwarded():
    strict = Scanner(categories=["gibberish"], threshold=0.01)
    lax = Scanner(categories=["gibberish"], threshold=0.99)
    text = "hello wrld frbl"
    assert strict.scan(text).flagged is True
    assert lax.scan(text).flagged is False
    allow = Scanner(categories=["gibberish"], allowlist=["qxzjkwpv"])
    assert allow.scan("qxzjkwpv").flagged is False


def test_gibberish_flagged_matches_ensemble_predict():
    from pygarble import EnsembleDetector

    scanner = Scanner(categories=["gibberish"], threshold=0.3)
    ensemble = EnsembleDetector(threshold=0.3)
    for text in [
        "hello world again",
        "hello wrld frbl",
        "qxzjkwpv bnmqwer zzxqv",
    ]:
        assert scanner.scan(text).flagged is ensemble.predict(text)


def test_public_exports_are_lazy():
    for name in ("Scanner", "scan", "redact", "Finding", "SecretsDetector"):
        assert name in pygarble.__all__
    code = (
        "import sys, pygarble; "
        "print(sorted(m for m in sys.modules "
        "if m in ('pygarble.scanner', 'pygarble.secrets')))"
    )
    import subprocess

    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": "."},
        cwd=REPO_ROOT,
    )
    assert out.stdout.strip() == "[]"
    assert isinstance(Finding("pii", "email", 0, 1, 0.9, "x"), Finding)


def test_star_import_and_hasattr_do_not_raise():
    namespace: dict = {}
    exec("from pygarble import *", namespace)
    assert namespace["Scanner"] is Scanner
    for name in ("PIIDetector", "ProfanityDetector"):
        assert isinstance(hasattr(pygarble, name), bool)
    with pytest.raises(AttributeError):
        pygarble.no_such_name  # type: ignore[attr-defined]


def test_bare_string_arguments_rejected():
    for kwargs in (
        {"categories": "secrets"},
        {"kinds": "garbled"},
        {"exclude_kinds": "garbled"},
    ):
        with pytest.raises(ValueError, match="not a string"):
            Scanner(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        list(Scanner().iter_scan("abc"))  # type: ignore[arg-type]


def test_secrets_without_context_forwarded():
    blob = "blob " + "9f86d081884c7d659a2feaa0c55ad015" * 2
    plain = Scanner(categories=["secrets"]).scan(blob)
    assert "high_entropy_string" not in [f.kind for f in plain.findings]
    loose = Scanner(categories=["secrets"], secrets_without_context=True)
    kinds = [f.kind for f in loose.scan(blob).findings]
    assert "high_entropy_string" in kinds


def test_exclude_kinds_removes_kind():
    report = Scanner(
        categories=["secrets"], exclude_kinds=["aws_access_key_id"]
    ).scan(f"key {AWS}")
    assert "aws_access_key_id" not in [f.kind for f in report.findings]


def test_redact_rejects_unknown_category_before_scanning():
    scanner = Scanner(categories=["secrets"])
    with pytest.raises(ValueError, match="unknown category"):
        scanner.redact(f"key {AWS}", categories=["nope"])
    with pytest.raises(ValueError, match="unknown category"):
        scanner.redact(None, categories=["nope"])  # type: ignore[arg-type]


def test_unknown_locale_rejected_even_without_pii():
    with pytest.raises(ValueError, match="unknown locale"):
        Scanner(categories=["secrets"], locales=["zz"])
    with pytest.raises(ValueError, match="not a string"):
        Scanner(categories=["secrets"], locales="us")
