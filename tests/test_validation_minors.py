"""Invalid options fail the same way regardless of call order or path."""

import sys

import pytest

import pygarble
from pygarble import GarbleDetector
from pygarble.screening.base import optional_module


def test_scan_cache_does_not_mask_invalid_or_unhashable_options():
    # F7: True is not accepted just because 1 was cached first.
    pygarble.scan("hello", min_confidence=1)
    with pytest.raises(ValueError, match="min_confidence"):
        pygarble.scan("hello", min_confidence=True)
    # A dict-valued option behaves as Scanner(**kwargs) does, not "unhashable".
    expected = pygarble.Scanner(allowlist={"asdkjh": [1]}).scan("asdkjh")
    assert pygarble.scan("asdkjh", allowlist={"asdkjh": [1]}) == expected


def test_control_characters_flags_text_made_only_of_controls():
    # F8: VT, FF, U+001C-U+001F and NEL count as whitespace for str.strip().
    detector = GarbleDetector("control_characters")
    for text in ("\x1c\x1d\x1e\x1f", "\x85\x85", "\x0b\x0c"):
        analysis = detector.analyze(text)
        assert analysis.garbled and analysis.status == "garbled", repr(text)
    assert detector.analyze("  \t ").status == "insufficient_evidence"


def test_applicable_enforces_max_input_length():
    # F10: same limit as analyze/predict.
    detector = GarbleDetector("markov_chain", max_input_length=10)
    with pytest.raises(ValueError, match="max_input_length"):
        detector.applicable("a fairly long sentence of text")


def test_pattern_matching_options_raise_value_error():
    # F6: every invalid option is a ValueError naming the problem.
    with pytest.raises(ValueError, match=r"unterminated|missing \)"):
        GarbleDetector("pattern_matching", patterns={"x": "("})
    with pytest.raises(ValueError, match="override_defaults"):
        GarbleDetector("pattern_matching", override_defaults="yes")


def test_optional_module_names_missing_setuptools(tmp_path, monkeypatch):
    # H4: python-stdnum on 3.8 imports pkg_resources; say so, not the extra.
    package = tmp_path / "fake_stdnum_backend"
    package.mkdir()
    (package / "__init__.py").write_text("import pkg_resources\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "pkg_resources", None)
    with pytest.raises(ImportError, match="setuptools") as info:
        optional_module("fake_stdnum_backend", "stdnum")
    assert "pygarble[stdnum]" not in str(info.value)
    with pytest.raises(ImportError, match=r"pygarble\[stdnum\]"):
        optional_module("no_such_backend_module", "stdnum")
