"""Contracts every registered strategy must satisfy."""

import inspect
import re

from pygarble.preprocessing import ascii_alpha_words
from pygarble.strategies import rare_trigram


def test_ascii_alpha_words_matches_legacy_regex():
    text = "Hello, wörld! it's 42 x-ray"
    assert ascii_alpha_words(text) == [
        "hello",
        "w",
        "rld",
        "it",
        "s",
        "x",
        "ray",
    ]


def test_rare_trigram_source_lists_each_entry_once():
    # IMPOSSIBLE_TRIGRAMS is a set literal, so duplicates collapse at
    # runtime; guard the source text instead.
    source = inspect.getsource(rare_trigram)
    entries = re.findall(r'"([a-z]{3})"', source)
    dupes = {e for e in entries if entries.count(e) > 1}
    assert not dupes
