"""Packaging invariants that CI must keep true."""

import re
from pathlib import Path

import pytest

import pygarble

ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_has_no_static_version():
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^version\s*=\s*"', text, re.M) is None
    assert 'dynamic = ["version"]' in text
    assert 'version = {attr = "pygarble.__version__"}' in text


def test_docs_conf_uses_package_version():
    conf = ROOT / "docs" / "conf.py"
    if not conf.exists():
        pytest.skip("docs/ is not shipped in sdist")
    text = conf.read_text()
    assert "pygarble.__version__" in text
    assert "version = '0." not in text


def test_author_email_is_not_placeholder():
    text = (ROOT / "pyproject.toml").read_text()
    assert "example.com" not in text
    assert pygarble.__email__ in text


JSON_TABLES = (
    "words.json",
    "bigrams.json",
    "trigrams.json",
    "secrets.json",
    "pii.json",
    "profanity.json",
    "reference.json",
    "ngram_ranks.json",
    "calibration.json",
)


def test_json_tables_excluded_from_wheel():
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(
        r"^\[tool\.setuptools\.exclude-package-data\]\s*\n"
        r'"pygarble\.data"\s*=\s*\[([^\]]*)\]',
        text,
        re.M,
    )
    assert match is not None
    excluded = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert set(JSON_TABLES) <= excluded


def test_json_tables_included_in_sdist():
    manifest_in = ROOT / "MANIFEST.in"
    if not manifest_in.exists():
        pytest.skip("MANIFEST.in is not installed")
    lines = manifest_in.read_text().splitlines()
    assert "recursive-include pygarble *.py *.json" in lines
