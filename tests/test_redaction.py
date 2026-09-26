"""Redaction merges regions and renders every mode deterministically."""

import pytest

from pygarble.findings import Finding, Redaction
from pygarble.redaction import MODES, REVEAL_LAST_FOUR, render


def f(kind, start, end, confidence=0.9, category="pii"):
    return Finding(category, kind, start, end, confidence, "test")


TEXT = "mail a@b.co card 4111 1111 1111 1111 now"


def test_modes_and_reveal_set():
    assert MODES == ("placeholder", "mask", "partial")
    assert "credit_card" in REVEAL_LAST_FOUR and "email" not in (
        REVEAL_LAST_FOUR
    )


def test_placeholder_mode_default_template():
    out = render(TEXT, [f("email", 5, 11)], "placeholder", "[{KIND}]", "*")
    assert isinstance(out, Redaction)
    assert out.text == "mail [EMAIL] card 4111 1111 1111 1111 now"
    assert out.count == 1
    assert out.findings == (f("email", 5, 11),)


def test_placeholder_template_fields():
    out = render(
        TEXT, [f("email", 5, 11)], "placeholder", "<{category}:{kind}>", "*"
    )
    assert out.text.startswith("mail <pii:email> card")


def test_placeholder_unknown_field_is_value_error():
    with pytest.raises(ValueError, match="placeholder"):
        render(TEXT, [f("email", 5, 11)], "placeholder", "[{nope}]", "*")
    for template in ("[{kind.x}]", "[{kind[x]}]"):
        with pytest.raises(ValueError, match="placeholder"):
            render(TEXT, [f("email", 5, 11)], "placeholder", template, "*")


def test_mask_mode_preserves_length():
    out = render(TEXT, [f("email", 5, 11)], "mask", "[{KIND}]", "#")
    assert out.text == "mail ###### card 4111 1111 1111 1111 now"
    assert len(out.text) == len(TEXT)


def test_partial_mode_reveals_last_four_only_for_listed_kinds():
    card = f("credit_card", 17, 36, 1.0)
    out = render(TEXT, [card, f("email", 5, 11)], "partial", "[{KIND}]", "*")
    assert out.text == "mail ****** card ***************1111 now"


def test_redact_merges_nested_and_touching_regions():
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc"
    assert len(text) == 62
    bearer = Finding("secrets", "bearer_token", 22, 62, 0.8, "bearer")
    jwt = Finding("secrets", "jwt", 22, 62, 1.0, "jwt_header")
    touching = Finding("pii", "email", 62, 62, 0.9, "x")
    out = render(text, [touching, bearer, jwt], "placeholder", "[{KIND}]", "*")
    assert out.text == "Authorization: Bearer [JWT]"
    assert out.count == 1
    assert out.findings[0] == bearer


def test_merge_ties_go_to_earliest_finding():
    a = f("phone", 0, 4, 0.8)
    b = f("ssn_us", 2, 6, 0.8)
    out = render("0123456789", [b, a], "placeholder", "[{KIND}]", "*")
    assert out.text == "[PHONE]6789"


def test_no_findings_returns_input_unchanged():
    out = render(TEXT, [], "mask", "[{KIND}]", "*")
    assert out.text == TEXT and out.count == 0 and out.findings == ()


def test_invalid_mode_and_mask_char():
    with pytest.raises(ValueError, match="mode"):
        render(TEXT, [], "shred", "[{KIND}]", "*")
    with pytest.raises(ValueError, match="mask_char"):
        render(TEXT, [], "mask", "[{KIND}]", "**")


def test_placeholder_is_idempotent():
    once = render(TEXT, [f("email", 5, 11)], "placeholder", "[{KIND}]", "*")
    again = render(once.text, [], "placeholder", "[{KIND}]", "*")
    assert again.text == once.text


def test_partial_never_reveals_tail_of_non_revealable_finding():
    text = "555-1234bob@x.com"
    phone = f("phone", 0, 8, 0.9)
    email = f("email", 8, 17, 0.8)
    out = render(text, [phone, email], "partial", "[{KIND}]", "*")
    assert out.text == "*" * len(text)


def test_partial_reveals_when_region_ends_with_revealable_finding():
    text = "bob@x.com4111111111111111"
    email = f("email", 0, 9, 0.9)
    card = f("credit_card", 9, 25, 0.8)
    out = render(text, [email, card], "partial", "[{KIND}]", "*")
    assert out.text == "*" * 21 + "1111"


def test_partial_hides_tail_covered_by_non_revealable_finding():
    text = "4111111111111111"
    card = f("credit_card", 0, 16, 1.0)
    email = f("email", 12, 16, 0.5)
    out = render(text, [card, email], "partial", "[{KIND}]", "*")
    assert out.text == "*" * 16


def test_bad_template_raises_even_without_findings():
    with pytest.raises(ValueError, match="placeholder"):
        render("clean", [], "placeholder", "[{nope}]", "*")


@pytest.mark.parametrize(
    "template",
    [
        "[{kind[0]}]",
        "[{KIND.__class__}]",
        "[{}]",
        "[{0}]",
        "[{kind!r}]",
        "[{kind:>9}]",
        "[{kind",
        "[kind}]",
    ],
)
def test_template_rejects_access_conversion_and_spec(template):
    with pytest.raises(ValueError, match="placeholder may use only"):
        render(TEXT, [], "placeholder", template, "*")


def test_template_fields_all_render():
    out = render(
        TEXT,
        [f("email", 5, 11)],
        "placeholder",
        "{{{KIND}|{kind}|{category}}}",
        "*",
    )
    assert out.text.startswith("mail {EMAIL|email|pii} card")


def test_template_is_not_validated_outside_placeholder_mode():
    out = render(TEXT, [f("email", 5, 11)], "mask", "[{nope}]", "*")
    assert out.text.startswith("mail ****** card")
