"""Standalone screening, compatibility, custom detectors and failures."""

import subprocess
import sys

import pytest

from pygarble import Scanner as LegacyScanner
from pygarble.screening import BackendError, Finding, Scanner, redact, scan


def test_screening_import_and_scan_do_not_load_gibberish_or_extras():
    code = """
import sys
from pygarble.screening import Scanner
s = Scanner()
assert s.scan('mail jane@example.com').kinds() == ('email',)
assert not s.scan('qxzjkwpv bnmqwer zzxqv').flagged
for name in ('pygarble.ensemble', 'pygarble.registry',
             'pygarble.gibberish.ensemble', 'pygarble.gibberish.registry',
             'pygarble.gibberish.strategies', 'pygarble.data.words',
             'phonenumbers',
             'stdnum', 'detect_secrets', 'spacy', 'numpy'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_screening_and_legacy_rules_agree():
    text = "key AKIAIOSFODNN7EXAMPLE; mail a@b.co; damn"
    legacy = LegacyScanner(categories=["secrets", "pii", "profanity"])
    current = Scanner()
    assert current.scan(text) == legacy.scan(text)
    assert current.redact(text) == legacy.redact(text)
    assert LegacyScanner().scan("qxzjkwpv bnmqwer zzxqv").flagged


def test_convenience_batch_and_iter():
    texts = ["hello", "mail a@b.co", "damn"]
    scanner = Scanner()
    expected = [scan(text) for text in texts]
    assert scanner.scan_batch(texts) == expected
    assert list(scanner.iter_scan(iter(texts))) == expected
    assert redact("mail a@b.co").text == "mail [EMAIL]"


def test_screening_filters_and_profanity_tiers():
    assert not Scanner(profanity_tiers=["strong"]).scan("damn").flagged
    assert not Scanner(profanity_allowlist=["damn"]).scan("damn").flagged
    assert Scanner(kinds=["email"]).scan("a@b.co damn").kinds() == ("email",)
    report = Scanner(min_confidence=1).scan("a@b.co")
    assert report.findings and not report.flagged
    assert Scanner(min_confidence=1).redact("a@b.co").text == "a@b.co"


@pytest.mark.parametrize(
    "options",
    [
        {"categories": ["gibberish"]},
        {"categories": []},
        {"categories": "pii"},
        {"backends": "stdnum"},
        {"backends": ["unknown"]},
        {"kinds": ["unknown"]},
        {"kinds": []},
        {"locales": ["unknown"]},
        {"builtin": False},
        {"backend_options": {"stdnum": {}}},
        {"categories": ["profanity"], "backends": ["stdnum"]},
        {"kinds": ["high_entropy_string"]},
    ],
)
def test_bad_configuration(options):
    with pytest.raises(ValueError):
        Scanner(**options)


def test_input_limit_and_type():
    scanner = Scanner(max_input_length=4)
    with pytest.raises(ValueError, match="max_input_length"):
        scanner.scan("hello")
    with pytest.raises(TypeError):
        scanner.scan(None)


class CustomDetector:
    category = "pii"
    kinds = frozenset({"customer_id"})

    def detect(self, text):
        start = text.find("ID123")
        if start < 0:
            return ()
        return (
            Finding("pii", "customer_id", start, start + 5, 0.9, "customer"),
        )


def test_custom_detector_unicode_redaction_and_filters():
    scanner = Scanner(detectors=[CustomDetector()], builtin=False)
    text = "é😀 ID123 end"
    assert scanner.redact(text).text == "é😀 [CUSTOMER_ID] end"
    assert scanner.redact(text, mode="mask").text == "é😀 ***** end"
    with pytest.raises(ValueError, match="nothing to scan"):
        Scanner(
            detectors=[CustomDetector()],
            builtin=False,
            exclude_kinds=["customer_id"],
        )


def test_backend_failures_do_not_leak_or_return_clean():
    class Broken(CustomDetector):
        def detect(self, text):
            raise RuntimeError("secret=" + text)

    scanner = Scanner(detectors=[Broken()])
    with pytest.raises(BackendError) as caught:
        scanner.scan("sensitive-value")
    assert "sensitive-value" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize(
    "finding",
    [
        Finding("pii", "customer_id", -1, 2, 0.9, "custom"),
        Finding("pii", "customer_id", 0, 999, 0.9, "custom"),
        Finding("pii", "customer_id", 0, 0, 0.9, "custom"),
        Finding("secrets", "customer_id", 0, 2, 0.9, "custom"),
        Finding("pii", "customer_id", 0, 2, float("nan"), "custom"),
    ],
)
def test_invalid_backend_spans_and_scores_fail_closed(finding):
    class Broken(CustomDetector):
        def detect(self, text):
            return (finding,)

    with pytest.raises(BackendError):
        Scanner(detectors=[Broken()]).redact("ID123")


def test_duplicate_same_kind_spans_keep_stronger_evidence():
    class Weak(CustomDetector):
        def detect(self, text):
            return (Finding("pii", "customer_id", 0, 5, 0.6, "weak"),)

    scanner = Scanner(builtin=False, detectors=[Weak(), CustomDetector()])
    findings = scanner.scan("ID123").findings
    assert len(findings) == 1
    assert findings[0].confidence == 0.9
