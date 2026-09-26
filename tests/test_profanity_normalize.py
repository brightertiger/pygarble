"""Token normalisation is deterministic and never widens to non-letters."""

from pygarble.profanity.normalize import (
    LEET_MAP,
    TOKEN_RE,
    collapse_runs,
    has_long_run,
    normalize_token,
)


def test_leet_map_is_single_chars():
    assert all(len(k) == 1 and len(v) == 1 for k, v in LEET_MAP.items())
    assert (
        LEET_MAP["@"] == "a" and LEET_MAP["$"] == "s" and LEET_MAP["0"] == "o"
    )


def test_normalize_token():
    assert normalize_token("Sh1T") == "shit"
    assert normalize_token("@ss") == "ass"
    assert normalize_token("don't") == "dont"
    assert normalize_token("café") == "cafe"
    assert normalize_token("b00k") == "book"
    assert normalize_token("138") == "ieb"  # digits map through leet
    assert normalize_token("") == ""


def test_collapse_runs_and_detection():
    assert collapse_runs("shiiit", 1) == "shit"
    assert collapse_runs("shiiit", 2) == "shiit"
    assert collapse_runs("book", 1) == "bok"
    assert has_long_run("shiiit") and not has_long_run("book")
    assert not has_long_run("")


def test_token_regex_keeps_symbols_inside_tokens():
    text = "f*ck sh!t $5 email@example.com a.b"
    assert [m.group() for m in TOKEN_RE.finditer(text)] == [
        "f*ck",
        "sh!t",
        "$5",
        "email@example",
        "com",
        "a",
        "b",
    ]
