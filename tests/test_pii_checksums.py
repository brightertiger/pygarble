"""Checksum functions are pure and never raise."""

from pygarble.pii.checksums import iban_mod97, luhn, nhs_mod11, verhoeff


def test_luhn():
    assert luhn("4111111111111111")
    assert luhn("378282246310005")
    assert luhn("6011111111111117")
    assert not luhn("4111111111111112")
    assert not luhn("")
    assert not luhn("12a4")


def test_iban_mod97():
    assert iban_mod97("GB82WEST12345698765432")
    assert iban_mod97("DE89370400440532013000")
    assert iban_mod97("FR7630006000011234567890189")
    assert not iban_mod97("GB82WEST12345698765433")
    assert not iban_mod97("GB82")
    assert not iban_mod97("gb82west12345698765432")


def test_verhoeff():
    assert verhoeff("236")
    assert verhoeff("2363")  # published worked example
    assert verhoeff("123451")
    assert not verhoeff("12345")
    assert not verhoeff("235")
    assert not verhoeff("")
    assert not verhoeff("12x45")


def test_nhs_mod11():
    assert nhs_mod11("9434765919")
    assert nhs_mod11("4010232137")
    assert not nhs_mod11("9434765918")
    assert not nhs_mod11("123456789")
    assert not nhs_mod11("943476591x")
    # Weighted remainder 10 means no valid check digit exists.
    for check in "0123456789":
        assert not nhs_mod11("100000001" + check)


def test_non_ascii_digits_return_false_without_raising():
    for value in ("\u00b2" * 10, "\u0661\u0662\u0663\u0664\u0665" * 2):
        assert not luhn(value)
        assert not verhoeff(value)
        assert not nhs_mod11(value)
        assert not iban_mod97(value)
    assert not iban_mod97("GB82\u00b2EST12345698765432")
