"""Real optional packages, offline matching and exact redaction spans."""

import os
import shutil
import socket

import pytest

from pygarble.screening import Scanner
from pygarble.screening.backends import (
    DetectSecretsDetector,
    GitleaksDetector,
    PhoneNumbersDetector,
    StdnumDetector,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("screening attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket, "getaddrinfo", fail)


def test_international_phone_spans_and_redaction():
    pytest.importorskip("phonenumbers")
    scanner = Scanner(builtin=False, backends=["phonenumbers"])
    text = "é😀 Call +33 1 42 68 53 00 or +49 30 901820"
    assert scanner.redact(text).text == "é😀 Call [PHONE] or [PHONE]"
    assert [text[f.start : f.end] for f in scanner.scan(text).findings] == [
        "+33 1 42 68 53 00",
        "+49 30 901820",
    ]
    assert not scanner.scan("+9991234567890").flagged


def test_phone_region_and_missing_number_metadata():
    pytest.importorskip("phonenumbers")
    detector = PhoneNumbersDetector(region="gb")
    assert detector.detect("Call 020 8366 1177")
    assert PhoneNumbersDetector().detect("Call 020 8366 1177") == ()
    with pytest.raises(ValueError, match="region"):
        PhoneNumbersDetector(region="not-a-region")


def test_stdnum_curated_formats_and_checksums():
    pytest.importorskip("stdnum")
    scanner = Scanner(builtin=False, backends=["stdnum"])
    text = "é😀 CPF 390.533.447-05; IBAN GB82 WEST 1234 5698 7654 32"
    assert scanner.redact(text).text == "é😀 CPF [CPF_BR]; IBAN [IBAN]"
    assert not scanner.scan("CPF 390.533.447-06").flagged
    assert not scanner.scan("order 39053344705").flagged
    assert scanner.scan("CPF: 39053344705").kinds() == ("cpf_br",)


def test_stdnum_formats_and_kind_filters():
    pytest.importorskip("stdnum")
    scanner = Scanner(
        builtin=False,
        backends=["stdnum"],
        backend_options={"stdnum": {"formats": ["br.cpf"]}},
        kinds=["cpf_br"],
    )
    assert scanner.scan("CPF 390.533.447-05").flagged
    assert not scanner.scan("GB82 WEST 1234 5698 7654 32").flagged
    with pytest.raises(ValueError, match="formats"):
        StdnumDetector(formats=["missing"])
    with pytest.raises(ValueError, match="nothing to scan"):
        Scanner(
            builtin=False,
            backends=["stdnum"],
            kinds=["iban"],
            backend_options={"stdnum": {"formats": ["br.cpf"]}},
        )


def test_detect_secrets_quoted_passwords_repetitions_and_global_state(
    monkeypatch,
):
    pytest.importorskip("detect_secrets")
    from detect_secrets.plugins.base import BasePlugin
    from detect_secrets.settings import get_settings

    def fail(*args, **kwargs):
        raise AssertionError("verification must never run")

    monkeypatch.setattr(BasePlugin, "analyze_line", fail)
    before = get_settings().json()
    scanner = Scanner(
        builtin=False,
        backends=["detect-secrets"],
        backend_options={"detect-secrets": {"plugins": ["KeywordDetector"]}},
    )
    text = 'é😀 password="hunter2"; repeat hunter2\r\npassword="letmein123"'
    assert scanner.redact(text).text == (
        'é😀 password="[DETECT_SECRETS_SECRET]"; repeat '
        '[DETECT_SECRETS_SECRET]\r\npassword="[DETECT_SECRETS_SECRET]"'
    )
    assert get_settings().json() == before


def test_detect_secrets_default_vendor_plugins():
    pytest.importorskip("detect_secrets")
    detector = DetectSecretsDetector()
    value = "AKIA" + "QWERTYUIOPASDFGH"
    assert detector.detect("key " + value)
    assert detector.detect("ordinary clean sentence") == ()


def test_detect_secrets_entropy_threshold_is_preserved():
    pytest.importorskip("detect_secrets")
    detector = DetectSecretsDetector(plugins=["Base64HighEntropyString"])
    assert detector.detect('value="' + "a" * 48 + '"') == ()
    value = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH"
    assert detector.detect('value="' + value + '"')
    with pytest.raises(ValueError):
        DetectSecretsDetector(plugins=["PrivateKeyDetector"])
    with pytest.raises(ValueError):
        DetectSecretsDetector(plugins=["unknown"])


@pytest.fixture
def gitleaks():
    executable = os.environ.get("PYGARBLE_GITLEAKS") or shutil.which(
        "gitleaks"
    )
    if executable is None:
        pytest.skip("Gitleaks executable is not installed")
    return executable


def test_gitleaks_real_unicode_repeated_values_crlf_and_redaction(gitleaks):
    token = "ghp_" + ("a1B2c3D4" * 5)[:36]
    text = 'é😀 token="' + token + '"\r\nnext="' + token + '"'
    scanner = Scanner(
        builtin=False,
        backends=["gitleaks"],
        backend_options={"gitleaks": {"executable": gitleaks}},
    )
    assert scanner.redact(text).text == (
        'é😀 token="[GITLEAKS_SECRET]"\r\nnext="[GITLEAKS_SECRET]"'
    )
    assert len(scanner.scan(text).findings) == 2
    assert not scanner.scan("ordinary clean sentence").flagged


def test_gitleaks_large_document_and_multiline_key(gitleaks):
    detector = GitleaksDetector(executable=gitleaks)
    token = "ghp_" + ("a1B2c3D4" * 5)[:36]
    text = "clean text\n" * 10000 + token
    (finding,) = detector.detect(text)
    assert text[finding.start : finding.end] == token
    key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        + "aBCdEF0123456789/" * 32
        + "\n-----END RSA PRIVATE KEY-----"
    )
    text = "first\n" + key + "\nlast"
    (finding,) = detector.detect(text)
    assert text[finding.start : finding.end] == key


def test_gitleaks_ignores_ambient_config_and_inline_allow_comments(
    gitleaks, monkeypatch
):
    monkeypatch.setenv("GITLEAKS_CONFIG", "/missing-config.toml")
    monkeypatch.setenv("GITLEAKS_CONFIG_TOML", "invalid toml")
    detector = GitleaksDetector(executable=gitleaks)
    token = "ghp_" + ("a1B2c3D4" * 5)[:36]
    assert detector.detect('token="' + token + '" # gitleaks:allow')
