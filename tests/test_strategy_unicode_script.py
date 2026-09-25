"""UnicodeScript: only confusable-script mixing inside a word is spoofing."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "私はiPhoneを使います",
        "我用Python写代码",
        "The α-helix is stable",
        "cells were 5 μm wide",
        "E=mc² with Δt",
        "Hello, World! 123",
    ],
)
def test_natural_script_mixing_is_clean(text):
    assert GarbleDetector(Strategy.UNICODE_SCRIPT).score(text) < 0.5


@pytest.mark.parametrize("text", ["pаypal", "gооgle login", "аpple"])
def test_confusable_mixing_inside_a_word_fires(text):
    assert GarbleDetector(Strategy.UNICODE_SCRIPT).score(text) >= 0.5
