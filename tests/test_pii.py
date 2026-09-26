"""Structural PII with checksums; locale packs; overlap rules."""

import pytest

from pygarble.pii import ALL_KINDS, LOCALES, PIIDetector, detect
from pygarble.pii.patterns import export


def spans(text, findings):
    return [(f.kind, text[f.start : f.end], f.confidence) for f in findings]


def test_kinds_locales_and_export():
    assert LOCALES == ("us", "uk", "in")
    assert ALL_KINDS == frozenset(
        {
            "email",
            "phone",
            "credit_card",
            "iban",
            "ipv4",
            "ipv6",
            "ssn_us",
            "nino",
            "nhs_number",
            "aadhaar",
            "pan",
        }
    )
    payload = export()
    assert set(payload) >= {"generic", "locales", "iban_lengths", "iin"}


def test_email():
    text = "Write to jane.doe+news@example.co.uk. Not at@x or @handle."
    assert spans(text, detect(text)) == [
        ("email", "jane.doe+news@example.co.uk", 0.9)
    ]
    assert detect("user@localhost") == ()


def test_phone_formats():
    cases = {
        "+14155552671": ("e164", 0.9),
        "(415) 555-2671": ("national_us", 0.8),
        "415-555-2671": ("national_us", 0.8),
        "020 7946 0958": ("national_uk", 0.8),
        "07911 123456": ("national_uk", 0.8),
        "+91 98765 43210": ("national_in", 0.8),
        "9876543210": ("national_in", 0.8),
    }
    for text, (reason, confidence) in cases.items():
        found = [f for f in detect(f"call {text} now") if f.kind == "phone"]
        assert found, text
        assert found[0].reason == reason and found[0].confidence == confidence
    for text in ["2026-09-26", "1695700000", "order 123456", "v1.2.3.4"]:
        assert not [f for f in detect(text) if f.kind == "phone"], text


def test_phone_locale_gating():
    only_us = PIIDetector(locales=["us"])
    assert [f.kind for f in only_us.detect("07911 123456")] == []
    assert only_us.detect("+14155552671")[0].reason == "e164"


def test_credit_card_luhn_and_brand():
    text = "pay with 4111 1111 1111 1111 or 3782-822463-10005"
    found = [f for f in detect(text) if f.kind == "credit_card"]
    assert [
        (text[f.start : f.end], f.reason, f.confidence) for f in found
    ] == [
        ("4111 1111 1111 1111", "luhn_visa", 1.0),
        ("3782-822463-10005", "luhn_amex", 1.0),
    ]
    assert not [
        f for f in detect("4111 1111 1111 1112") if f.kind == "credit_card"
    ]
    assert not [
        f for f in detect("1234 5678 9012 3452") if f.kind == "credit_card"
    ]


def test_card_beats_phone_on_identical_span():
    # 13 digits, Luhn-valid Visa; also shaped like a long phone number.
    text = "id 4222222222222"
    found = detect(text)
    assert [f.kind for f in found] == ["credit_card"]


def test_iban():
    text = "IBAN GB82 WEST 1234 5698 7654 32 and DE89370400440532013000"
    found = [f for f in detect(text) if f.kind == "iban"]
    assert [text[f.start : f.end] for f in found] == [
        "GB82 WEST 1234 5698 7654 32",
        "DE89370400440532013000",
    ]
    assert all(f.confidence == 1.0 and f.reason == "mod97" for f in found)
    assert not [
        f for f in detect("GB82WEST12345698765433") if f.kind == "iban"
    ]
    assert not [
        f for f in detect("XX82WEST12345698765432") if f.kind == "iban"
    ]


def test_ip_addresses():
    text = "from 192.168.1.10 to 2001:db8::8a2e:370:7334 at 12:30:45"
    found = detect(text)
    assert spans(text, found) == [
        ("ipv4", "192.168.1.10", 0.7),
        ("ipv6", "2001:db8::8a2e:370:7334", 0.7),
    ]
    for negative in ["v1.2.3.4", "256.1.1.1", "1.2.3", "de:ad:be:ef:00:01"]:
        assert not [f for f in detect(negative) if f.kind.startswith("ip")]


def test_ssn_us():
    assert spans("ssn 123-45-6789", detect("ssn 123-45-6789")) == [
        ("ssn_us", "123-45-6789", 0.8)
    ]
    assert detect("123 45 6789")[0].kind == "ssn_us"
    (ctx,) = detect("SSN: 123456789")
    assert ctx.kind == "ssn_us" and ctx.confidence == 0.6
    for negative in [
        "000-45-6789",
        "666-45-6789",
        "900-45-6789",
        "123-00-6789",
        "123-45-0000",
        "id 123456789",
    ]:
        assert not [
            f for f in detect(negative) if f.kind == "ssn_us"
        ], negative


def test_uk_pack():
    assert spans("NI AB 12 34 56 C", detect("NI AB 12 34 56 C")) == [
        ("nino", "AB 12 34 56 C", 0.9)
    ]
    assert not [f for f in detect("BG 12 34 56 C") if f.kind == "nino"]
    assert not [f for f in detect("AB 12 34 56 E") if f.kind == "nino"]
    text = "NHS number 943 476 5919"
    assert spans(text, detect(text)) == [("nhs_number", "943 476 5919", 0.9)]
    (bare,) = detect("nhs: 9434765919")
    assert bare.kind == "nhs_number"
    assert not [f for f in detect("943 476 5918") if f.kind == "nhs_number"]
    assert not [f for f in detect("ref 9434765919") if f.kind == "nhs_number"]


def test_india_pack():
    text = "Aadhaar 2345 6789 0124"
    found = [f for f in detect(text) if f.kind == "aadhaar"]
    assert (
        found and found[0].confidence == 1.0 and found[0].reason == "verhoeff"
    )
    assert not [f for f in detect("1345 6789 0124") if f.kind == "aadhaar"]
    assert not [f for f in detect("ref 234567890124") if f.kind == "aadhaar"]
    assert spans("PAN ABCPE1234F", detect("PAN ABCPE1234F")) == [
        ("pan", "ABCPE1234F", 0.9)
    ]
    assert not [f for f in detect("ABCDE1234F") if f.kind == "pan"]


def test_identical_spans_keep_higher_confidence_only():
    text = "9434765919"
    found = PIIDetector(locales=["uk", "in"]).detect("nhs " + text)
    assert [f.kind for f in found] == ["nhs_number"]


def test_kind_and_locale_validation():
    with pytest.raises(ValueError, match="unknown pii kind"):
        PIIDetector(kinds=["passport"])
    with pytest.raises(ValueError, match="unknown locale"):
        PIIDetector(locales=["fr"])
    assert PIIDetector(kinds=["email"]).detect("+14155552671") == ()
    assert PIIDetector(exclude_kinds=["phone"]).detect("+14155552671") == ()
    with pytest.raises(TypeError):
        detect(123)  # type: ignore[arg-type]


def test_deterministic_and_sorted():
    text = "a@b.com 4111111111111111 +14155552671 a@b.com"
    first = detect(text)
    assert first == detect(text)
    assert [f.start for f in first] == sorted(f.start for f in first)


def test_bare_string_names_are_rejected():
    for key in ("kinds", "exclude_kinds", "locales"):
        with pytest.raises(ValueError, match="not a string"):
            PIIDetector(**{key: "email" if "kinds" in key else "us"})


def test_phone_outranks_keywordless_nhs_on_identical_span():
    # 943 476 5919 passes NHS mod-11 and is also a US-shaped phone.
    for text in ("415 555 2671", "ref 943 476 5919"):
        found = detect(text)
        assert [(f.kind, f.confidence) for f in found] == [("phone", 0.8)]
    (only_uk,) = PIIDetector(locales=["uk"]).detect("ref 943 476 5919")
    assert (only_uk.kind, only_uk.confidence, only_uk.reason) == (
        "nhs_number",
        0.8,
        "mod11",
    )
    (keyword,) = detect("NHS number 943 476 5919")
    assert keyword.reason == "mod11_keyword"


def test_digit_rules_stop_inside_identifiers():
    for text in (
        "abc9876543210def",
        "a4111111111111111b",
        "tok_2345 6789 0124",
        "classname: 123456789",
        "id_123-45-6789",
    ):
        assert detect(text) == (), text
    found = detect("xnhs 9434765919")
    assert not [f for f in found if f.kind == "nhs_number"]
    for text in ("+14155552671", "(415) 555-2671", "+91 98765 43210"):
        assert [f.kind for f in detect(text)][:1] == ["phone"], text


def test_email_rejects_file_extension_tld():
    assert spans("file name@2x.png", detect("file name@2x.png")) == []
    assert spans("logo@3x.WEBP", detect("logo@3x.WEBP")) == []
    assert spans("jane@example.com", detect("jane@example.com")) == [
        ("email", "jane@example.com", 0.9)
    ]
    assert spans("a@b.co", detect("a@b.co")) == [("email", "a@b.co", 0.9)]
    assert "png" in export()["email_excluded_tlds"]
    assert "co" not in export()["email_excluded_tlds"]


def test_us_phone_requires_nanp_area_and_exchange():
    found = detect("401 023 2137")
    assert [(f.kind, f.confidence, f.reason) for f in found] == [
        ("nhs_number", 0.8, "mod11")
    ]
    assert spans("(415) 555-2671", detect("(415) 555-2671")) == [
        ("phone", "(415) 555-2671", 0.8)
    ]
    assert detect("012 345 6789", locales=["us"]) == ()
    assert detect("415 055 2671", locales=["us"]) == ()


def test_keywordless_nhs_nested_in_phone_is_dropped():
    # 4155552604 passes NHS mod-11; the +1 phone span strictly contains it.
    text = "+1 415 555 2604"
    assert spans(text, detect(text)) == [("phone", text, 0.8)]
    (keyword,) = [
        f for f in detect("NHS number 415 555 2604") if f.kind == "nhs_number"
    ]
    assert keyword.reason == "mod11_keyword"


@pytest.mark.parametrize(
    "text,kind,value",
    [
        ("NHS-9434765919", "nhs_number", "9434765919"),
        ("nhs-943 476 5919", "nhs_number", "943 476 5919"),
        ("SSN-123456789", "ssn_us", "123456789"),
        ("SSN123456789", "ssn_us", "123456789"),
        ("aadhaar-234123412346", "aadhaar", "234123412346"),
    ],
)
def test_keyword_rules_accept_a_hyphen_after_the_keyword(text, kind, value):
    (finding,) = detect(text)
    assert finding.kind == kind
    assert text[finding.start : finding.end] == value


@pytest.mark.parametrize(
    "text", ["ssn 1234567890", "nhs 94347659190", "aadhaar-2341234123467"]
)
def test_keyword_values_still_need_whole_digit_runs(text):
    assert detect(text) == ()
