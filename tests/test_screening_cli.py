"""The screening CLI processes whole documents without source-text logs."""

import json
import subprocess
import sys

import pytest


def run(*args, text=""):
    return subprocess.run(
        [sys.executable, "-m", "pygarble.screening", *args],
        input=text,
        text=True,
        capture_output=True,
    )


def test_scan_is_findings_only_unless_explicitly_requested():
    text = "mail jane@example.com\n"
    result = run("scan", text=text)
    assert result.returncode == 1
    assert "jane@example.com" not in result.stdout + result.stderr
    assert json.loads(result.stdout)["findings"][0]["kind"] == "email"
    result = run("scan", "--include-text", text=text)
    assert json.loads(result.stdout)["text"] == text


def test_multiline_key_body_is_redacted_from_stdin_and_file(tmp_path):
    text = (
        "before\n-----BEGIN PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA1234567890abcdef\n"
        "-----END PRIVATE KEY-----\nafter\n"
    )
    result = run("redact", text=text)
    assert result.returncode == 0
    assert result.stdout == "before\n[PRIVATE_KEY]\nafter\n"
    source = tmp_path / "input.txt"
    source.write_text(text)
    assert run("redact", str(source)).stdout == result.stdout
    assert (
        run("redact", "--categories", "secrets", str(source)).stdout
        == result.stdout
    )


def test_gibberish_is_not_part_of_screening_cli():
    result = run("scan", text="qxzjkwpv bnmqwer zzxqv")
    assert result.returncode == 0
    assert json.loads(result.stdout)["findings"] == []


@pytest.mark.parametrize(
    "args",
    [
        ("--max-input-length", "4"),
        ("--categories", "gibberish"),
        ("--backends", "unknown"),
        ("--backend-options", "[]"),
        ("--backend-options", "invalid"),
        ("--categories", "pii,"),
        (
            "--backends",
            "gitleaks",
            "--backend-options",
            '{"gitleaks":{"executable":"/missing/gitleaks"}}',
        ),
    ],
)
def test_errors_do_not_emit_clean_results_or_source(args):
    result = run("scan", *args, text="secret-sensitive-value")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "secret-sensitive-value" not in result.stderr


def test_empty_document():
    result = run("scan")
    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "flagged": False,
        "length": 0,
        "findings": [],
    }
