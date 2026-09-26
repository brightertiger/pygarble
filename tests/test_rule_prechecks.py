"""The throughput prefilters, fast paths and caches change no result."""

import json
import random
import re
import unicodedata
from pathlib import Path

import pytest

from pygarble import pii, secrets
from pygarble.pii import PIIDetector
from pygarble.profanity import ProfanityDetector
from pygarble.profanity.normalize import LEET_MAP, normalize_token

VECTORS = (
    Path(__file__).resolve().parent.parent
    / "paper"
    / "regression"
    / "scan_vectors.json"
)
PIECES = (
    "aA sS kK ſ K 0 1 2 3 4 5 6 7 8 9 ١ ３ - . : @ + _ / ="
    " \n ' ’ * # $ ! | ssn SSN social security nhs NHS aadhaar uidai"
    " bearer BEARER aws AWS key KEY token pwd passwd secret AKIA sk- sk-ant-"
    " eyJ :// GB82 WEST 1234 5698 7654 32 ABCPD1234E QQ123456C 4111 1111"
    " +44 +91 192.168.0.1 fe80::1 ::ffff: cafe é İ ß"
    " sh!t f*ck b u l l s h i t son of a bitch"
).split(" ")


def _texts():
    texts = []
    if VECTORS.is_file():
        rows = json.loads(VECTORS.read_text(encoding="utf-8"))
        texts.extend(row["text"] for row in rows)
    for entry in secrets.KNOWN_PATTERNS:
        texts.extend(entry["vectors"]["positive"])
        texts.extend(entry["vectors"]["negative"])
    rng = random.Random(5)
    for _ in range(3000):
        texts.append(
            "".join(
                rng.choice(PIECES) + rng.choice(["", " ", "-", "\n", ":"])
                for _ in range(rng.randint(1, 25))
            )
        )
    return texts


TEXTS = _texts()


def _pii_rules():
    for scope in ("generic",) + pii.LOCALES:
        for (rule, pattern), checks in zip(
            pii._compiled(scope), pii._PRECHECKS[scope]
        ):
            yield scope, rule, pattern, checks


def test_every_pii_rule_has_a_precheck():
    assert len(pii._PRECHECKS["generic"]) == len(pii.GENERIC)
    for name, rules in pii.LOCALE_RULES.items():
        assert len(pii._PRECHECKS[name]) == len(rules)


def test_pii_prechecks_hold_wherever_the_rule_matches():
    for text in TEXTS:
        facts = pii._Facts(text)
        for scope, rule, pattern, checks in _pii_rules():
            if pattern.search(text) is not None:
                assert facts.passes(checks), (scope, rule[0], text)


def test_pii_ascii_digit_tests_agree_with_the_regex():
    for text in TEXTS:
        fast = pii._Facts(text)
        if fast.digits is None:
            continue
        slow = pii._Facts(text)
        slow.digits = None
        for check in (("run", k) for k in range(1, 14)):
            assert fast._test(check) == slow._test(check), (check, text)
        for check in (("count", k) for k in (1, 13)):
            assert fast._test(check) == slow._test(check), (check, text)


def test_pii_rules_follow_kind_changes():
    detector = PIIDetector()
    text = "mail jane.doe@example.com or call +442079460958"
    assert {f.kind for f in detector.detect(text)} >= {"email", "phone"}
    detector.kinds = frozenset({"phone"})
    assert {f.kind for f in detector.detect(text)} == {"phone"}


def test_pii_rules_follow_in_place_kind_changes():
    detector = PIIDetector()
    text = "mail jane.doe@example.com or call +442079460958"
    detector.kinds = {"phone"}
    assert {f.kind for f in detector.detect(text)} == {"phone"}
    detector.kinds.add("email")
    assert {f.kind for f in detector.detect(text)} == {"phone", "email"}


def test_pii_rules_follow_in_place_locale_changes():
    detector = PIIDetector()
    text = "PAN ABCPD1234E"
    detector.locales = ["us"]
    assert detector.detect(text) == ()
    detector.locales.append("in")
    assert {f.kind for f in detector.detect(text)} == {"pan"}


def test_every_known_secret_pattern_has_an_anchor():
    assert len(secrets._ANCHORS) == len(secrets.KNOWN_PATTERNS)


def test_secret_anchors_hold_wherever_the_pattern_matches():
    for text in TEXTS:
        live = secrets._live_patterns(text, secrets._lowered(text))
        for index, part in enumerate(secrets._PARTS):
            if re.search(part, text) is not None:
                assert index in live, (index, text)


def test_secret_subset_alternation_matches_the_full_one():
    for text in TEXTS:
        live = secrets._live_patterns(text, secrets._lowered(text))
        subset = secrets._known_subset(live) if live else None
        full = [(m.span(), m.lastgroup) for m in secrets._KNOWN.finditer(text)]
        got = (
            [(m.span(), m.lastgroup) for m in subset.finditer(text)]
            if subset is not None
            else []
        )
        assert got == full, text


def test_secret_keyword_stems_hold_wherever_the_keyword_matches():
    for text in TEXTS:
        if secrets._KEYWORD.search(text) is not None:
            assert secrets._has_keyword_stem(text, secrets._lowered(text))
            assert secrets._has_keyword_stem(text, None)


def test_case_insensitive_literals_on_ascii_text():
    rng = random.Random(9)
    alphabet = "awsAWSbearerBEARERkeyKEYtokenTOKEN_- "
    for _ in range(5000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        for literal in ("aws", "bearer", "key", "token", "passw"):
            expected = re.search(f"(?i:{literal})", text) is not None
            assert (literal in text.lower()) == expected, (literal, text)


def _normalize_reference(token):
    folded = unicodedata.normalize("NFKD", token.casefold())
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    mapped = stripped.translate(str.maketrans(LEET_MAP))
    return "".join(c for c in mapped if "a" <= c <= "z")


def test_normalize_token_matches_the_reference():
    rng = random.Random(3)
    ascii_chars = [chr(i) for i in range(128)]
    tokens = ascii_chars + ["Sh1T", "@ss", "café", "Key", "x" * 100]
    for _ in range(5000):
        tokens.append(
            "".join(rng.choice(ascii_chars) for _ in range(rng.randint(0, 9)))
        )
    for token in tokens:
        assert normalize_token(token) == _normalize_reference(token), token
        assert normalize_token(token) == _normalize_reference(token), token


@pytest.mark.parametrize("obfuscation", [True, False])
def test_profanity_verdict_cache_changes_nothing(obfuscation):
    cached = ProfanityDetector(obfuscation=obfuscation)
    for text in TEXTS:
        fresh = ProfanityDetector(obfuscation=obfuscation)
        fresh.allowlist = set(fresh.allowlist)  # a set disables the cache
        assert cached.detect(text) == fresh.detect(text), text


def test_profanity_verdict_cache_follows_attribute_changes():
    detector = ProfanityDetector()
    assert detector.detect("well shit")
    detector.allowlist = frozenset({"shit"})
    assert detector.detect("well shit") == ()
    detector.allowlist = frozenset()
    detector.strong = frozenset()
    assert detector.detect("well shit") == ()
