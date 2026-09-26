"""Gitleaks transport errors and hostile/malformed result handling."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pygarble.screening import BackendError
from pygarble.screening.backends import GitleaksDetector


def install_result(monkeypatch, rows, returncode=10):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        report = Path(command[command.index("--report-path") + 1])
        report.write_text(json.dumps(rows))
        return subprocess.CompletedProcess(
            command, returncode, b"secret", b"secret"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def row(**kwargs):
    result = {
        "StartLine": 1,
        "EndLine": 1,
        "StartColumn": 8,
        "EndColumn": 13,
        "Secret": "secret",
        "RuleID": "example-rule",
    }
    result.update(kwargs)
    return result


def test_byte_offsets_unicode_temporary_cleanup_and_no_shell(monkeypatch):
    calls = install_result(monkeypatch, [row()])
    detector = GitleaksDetector(executable=sys.executable)
    text = "é😀 secret"
    (finding,) = detector.detect(text)
    assert text[finding.start : finding.end] == "secret"
    command, kwargs = calls[0]
    assert kwargs["input"] == text.encode("utf-8")
    assert not kwargs.get("shell", False)
    assert kwargs["timeout"] == 10
    assert "--ignore-gitleaks-allow" in command
    assert not Path(kwargs["cwd"]).exists()
    assert "secret" not in finding.to_dict().values()


@pytest.mark.parametrize(
    "rows,code",
    [
        ([row(StartLine=0)], 10),
        ([row(StartColumn=True)], 10),
        ([row(EndColumn=99)], 10),
        ([row(Secret="changed")], 10),
        ([row(RuleID="bad rule")], 10),
        ([row()], 0),
        ([], 10),
        ({"findings": []}, 0),
        ([], 1),
    ],
)
def test_invalid_results_are_errors_not_clean_scans(monkeypatch, rows, code):
    install_result(monkeypatch, rows, code)
    detector = GitleaksDetector(executable=sys.executable)
    with pytest.raises(BackendError) as caught:
        detector.detect("é😀 secret")
    assert "secret" not in str(caught.value)


def test_timeout_does_not_echo_subprocess_input(monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command, 0.1, output=b"sensitive-value"
        )

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(BackendError) as caught:
        GitleaksDetector(executable=sys.executable, timeout=0.1).detect(
            "secret"
        )
    assert "sensitive-value" not in str(caught.value)


def test_missing_executable_and_invalid_timeout():
    with pytest.raises(ImportError, match="install Gitleaks"):
        GitleaksDetector(executable="/missing/gitleaks")
    with pytest.raises(ValueError, match="timeout"):
        GitleaksDetector(executable=sys.executable, timeout=0)
