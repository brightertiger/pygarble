"""Check digits that keep the false-positive rate low."""

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _is_ascii_digits(text: str) -> bool:
    return text.isascii() and text.isdigit()


def luhn(digits: str) -> bool:
    """Luhn check on separator-free ASCII digits; never raises."""
    if not _is_ascii_digits(digits):
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_mod97(compact: str) -> bool:
    """ISO 13616 mod-97 on a separator-free uppercase IBAN; never raises."""
    if (
        len(compact) < 5
        or not compact.isascii()
        or not compact.isalnum()
        or compact != compact.upper()
    ):
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(
        str(ord(char) - 55) if char.isalpha() else char for char in rearranged
    )
    return int(number) % 97 == 1


def verhoeff(digits: str) -> bool:
    """Verhoeff check on separator-free ASCII digits; never raises."""
    if not _is_ascii_digits(digits):
        return False
    check = 0
    for index, char in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[index % 8][ord(char) - 48]]
    return check == 0


def nhs_mod11(digits: str) -> bool:
    """NHS mod-11 on ten separator-free ASCII digits; never raises."""
    if len(digits) != 10 or not _is_ascii_digits(digits):
        return False
    total = sum(
        (10 - index) * (ord(char) - 48)
        for index, char in enumerate(digits[:9])
    )
    remainder = 11 - (total % 11)
    if remainder == 11:
        remainder = 0
    if remainder == 10:
        return False
    return remainder == ord(digits[9]) - 48
