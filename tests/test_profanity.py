"""Each matching rule, in order, with its confidence and reason."""

import time

import pytest

from pygarble.profanity import ALL_KINDS, ProfanityDetector, detect


def hits(text, **kwargs):
    return [
        (text[f.start : f.end], f.confidence, f.reason)
        for f in detect(text, **kwargs)
    ]


def test_kinds():
    assert ALL_KINDS == frozenset({"profanity"})
    (f,) = detect("what the fuck")
    assert f.category == "profanity" and f.kind == "profanity"


def test_exact_strong_and_mild_with_leet_and_case():
    assert hits("What the FUCK") == [("FUCK", 1.0, "strong")]
    assert hits("sh1t happens") == [("sh1t", 1.0, "strong")]
    assert hits("well, damn.") == [("damn", 0.7, "mild")]
    assert hits("@ss") == [("@ss", 1.0, "strong")]


def test_scunthorpe_and_ordinary_words_are_clean():
    for text in [
        "Scunthorpe United",
        "assassin",
        "classic",
        "shiitake",
        "cocktail",
        "as",
        "pass the salt",
        "Dick Whittington",
        "hello world",
        "the analyst assessed the assets",
        "bass guitar",
        "Essex",
    ]:
        assert hits(text) == [], text


def test_ordinary_prose_uses_of_mild_words_are_only_mild():
    for text, word in [
        ("graduated magna cum laude", "cum"),
        ("a chink in the armour", "chink"),
        ("blue tits at the feeder", "tits"),
        ("a Maine Coon kitten", "Coon"),
        ("pissed off", "pissed"),
    ]:
        assert hits(text) == [(word, 0.7, "mild")], text
        assert hits(text, tiers=["strong"]) == [], text


def test_digit_only_tokens_never_match():
    for text in ["455", "7175", "room 455", "call 7175 now", "7175's"]:
        assert hits(text) == [], text
    # A digit token between words does not bridge a phrase.
    assert hits("son of 4 bitch") == [("bitch", 1.0, "strong")]
    # Symbol masks are not digits-only, so they still go through the rules.
    assert hits("@$$") == [("@$$", 1.0, "strong")]
    assert hits("$h1t") == [("$h1t", 1.0, "strong")]


def test_possessives_are_stripped_for_lookup():
    for text in ["Philip K. Dick's novel", "Dick's Sporting Goods"]:
        assert hits(text) == [], text
    assert hits("that bitch's car") == [("bitch's", 1.0, "strong")]
    assert hits("the BITCH’S car") == [("BITCH’S", 1.0, "strong")]


def test_elongation_rule():
    assert hits("shiiiit") == [("shiiiit", 0.8, "elongated")]
    assert hits("fuuuuck") == [("fuuuuck", 0.8, "elongated")]
    # A run of three collapses to two first, so "asss" reads as "ass".
    assert hits("asss") == [("asss", 0.8, "elongated")]
    assert hits("sooo good") == []  # "so" and "soo" are not listed
    assert hits("boooook") == []
    # Obfuscated mild words never score above the plain mild 0.7.
    assert hits("daaamn") == [("daaamn", 0.7, "elongated")]


def test_embedded_rule():
    assert hits("fuckwit") == [("fuckwit", 1.0, "strong")]
    # Only "fuck" is embedded; compounds of other words are listed.
    assert hits("shitposting") == [("shitposting", 1.0, "strong")]
    assert hits("unfuckingbelievable") == [
        ("unfuckingbelievable", 0.8, "embedded")
    ]
    assert hits("fuckwittery") == [("fuckwittery", 0.8, "embedded")]
    assert hits("fuckwittery", allowlist=["fuck"]) == []


def test_wildcard_rule():
    assert hits("f*cking") == [("f*cking", 0.9, "masked")]
    assert hits("b#stard") == [("b#stard", 0.9, "masked")]
    # "f*ck" also fits "fick" in the English word list, so it is ambiguous.
    assert hits("f*ck") == [("f*ck", 0.6, "masked_ambiguous")]
    assert hits("f#ck you") == [("f#ck", 0.6, "masked_ambiguous")]
    # "sh*t" also fits ordinary words (shot, shut), so it is ambiguous.
    assert hits("sh*t") == [("sh*t", 0.6, "masked_ambiguous")]
    assert hits("c*nt") == [("c*nt", 0.6, "masked_ambiguous")]
    assert hits("d*ck") == []  # "dick" is not listed (it is a name)
    assert hits("$5 and 50% off") == []
    assert hits("c*") == []
    assert hits("f*ck", obfuscation=False) == []
    assert hits("f*cking", allowlist=["fucking"]) == []


def test_spaced_rule():
    assert hits("s h i t") == [("s h i t", 0.8, "spaced")]
    assert hits("f.u.c.k off") == [("f.u.c.k", 0.8, "spaced")]
    assert hits("a b c") == []
    assert hits("i am a") == []
    assert hits("s h i t", obfuscation=False) == []
    assert hits("d a m n") == [("d a m n", 0.7, "spaced")]


def test_spaced_rule_after_a_or_i():
    assert hits("what a s h i t show") == [("s h i t", 0.8, "spaced")]
    assert hits("I s h i t you not") == [("s h i t", 0.8, "spaced")]
    assert hits("He's a p.r.i.c.k") == [("p.r.i.c.k", 0.7, "spaced")]
    assert hits("s h i t a") == [("s h i t", 0.8, "spaced")]
    assert hits("a b c d") == []


def test_trailing_exclamation_marks():
    assert hits("Damn!") == [("Damn!", 0.7, "mild")]
    assert hits("Shit!") == [("Shit!", 1.0, "strong")]
    assert hits("Fuck!!!") == [("Fuck!!!", 1.0, "strong")]
    assert hits("You ass!") == [("ass!", 1.0, "strong")]
    assert hits("sh!t") == [("sh!t", 1.0, "strong")]
    assert hits("b!tch") == [("b!tch", 1.0, "strong")]
    assert hits("Wow!!! Great!") == []


def test_many_masked_tokens_scan_quickly():
    text = "f*ck " * 5000
    started = time.perf_counter()
    found = detect(text)
    elapsed = time.perf_counter() - started
    assert len(found) == 5000
    assert elapsed < 0.5, elapsed


def test_phrase_rule():
    assert hits("you son of a bitch") == [("son of a bitch", 1.0, "strong")]
    assert hits("piece of SHIT, honestly") == [
        ("piece of SHIT", 1.0, "strong")
    ]
    assert hits("son of a bitch", allowlist=["son of a bitch"]) == [
        ("bitch", 1.0, "strong")
    ]
    assert hits("piece of shit", tiers=["mild"]) == []


def test_allowlist_and_tiers():
    assert hits("damn", allowlist=["damn"]) == []
    assert hits("Fuck", allowlist=["fuck"]) == []
    assert hits("damn it", tiers=["strong"]) == []
    with pytest.raises(ValueError, match="tiers"):
        ProfanityDetector(tiers=["extreme"])
    with pytest.raises(ValueError, match="tiers"):
        ProfanityDetector(tiers=[])
    with pytest.raises(TypeError):
        detect(None)  # type: ignore[arg-type]


def test_offsets_with_unicode_and_punctuation():
    text = "café — “fuck” — fin"
    (f,) = detect(text)
    assert text[f.start : f.end] == "fuck"


def test_deterministic_and_sorted():
    text = "shit fuck shit"
    first = detect(text)
    assert first == detect(text)
    assert [f.start for f in first] == [0, 5, 10]
