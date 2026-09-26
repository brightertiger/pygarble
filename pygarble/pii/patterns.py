"""PII shapes. Source of truth for pii.json."""

from typing import Any, Dict, Tuple

# (kind, regex, confidence, reason, validator)
Rule = Tuple[str, str, float, str, str]

EMAIL = (
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+"
    r"(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}(?![A-Za-z0-9\-])"
)
E164 = r"(?<![\w+])\+[1-9]\d{6,14}(?!\d)"
CARD = r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])"
IBAN = (
    r"(?<![A-Z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?"
    r"(?![A-Z0-9])"
)
IPV4 = (
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.])"
)
_H = r"[0-9A-Fa-f]{1,4}"
IPV6 = (
    r"(?<![\w:])(?:"
    rf"(?:{_H}:){{7}}{_H}|(?:{_H}:){{1,7}}:|(?:{_H}:){{1,6}}:{_H}|"
    rf"(?:{_H}:){{1,5}}(?::{_H}){{1,2}}|(?:{_H}:){{1,4}}(?::{_H}){{1,3}}|"
    rf"(?:{_H}:){{1,3}}(?::{_H}){{1,4}}|(?:{_H}:){{1,2}}(?::{_H}){{1,5}}|"
    rf"{_H}:(?::{_H}){{1,6}}|:(?::{_H}){{1,7}}|"
    r"::(?:[Ff]{4}:)?(?:\d{1,3}\.){3}\d{1,3}"
    r")(?![\w:])"
)

GENERIC: Tuple[Rule, ...] = (
    ("email", EMAIL, 0.9, "structure", ""),
    ("phone", E164, 0.9, "e164", "phone_digits"),
    ("credit_card", CARD, 1.0, "luhn", "card"),
    ("iban", IBAN, 1.0, "mod97", "iban"),
    ("ipv4", IPV4, 0.7, "structure", ""),
    ("ipv6", IPV6, 0.7, "structure", ""),
)

LOCALE_RULES: Dict[str, Tuple[Rule, ...]] = {
    "us": (
        (
            "phone",
            r"(?<![\d\-])(?:\+?1[\s.\-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.\-])"
            r"\d{3}[\s.\-]\d{4}(?![\d\-])",
            0.8,
            "national_us",
            "phone_digits",
        ),
        (
            "ssn_us",
            r"(?<![\d\-])(?!000|666|9\d\d)\d{3}([\-\s])(?!00)\d{2}\1"
            r"(?!0000)\d{4}(?![\d\-])",
            0.8,
            "structure",
            "",
        ),
        (
            "ssn_us",
            r"(?i:ssn|social security)[^\d\n]{0,30}"
            r"(?P<v>(?!000|666|9)\d{3}(?!00)\d{2}(?!0000)\d{4})(?!\d)",
            0.6,
            "keyword_context",
            "",
        ),
    ),
    "uk": (
        (
            "phone",
            r"(?<![\d\-])(?:\+44\s?|0)(?:\d{4}\s?\d{6}|\d{3}\s?\d{3}\s?\d{4}"
            r"|\d{2}\s?\d{4}\s?\d{4})(?![\d\-])",
            0.8,
            "national_uk",
            "phone_digits",
        ),
        (
            "nino",
            r"(?<![A-Z0-9])(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z]"
            r"[A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D](?![A-Z0-9])",
            0.9,
            "structure",
            "",
        ),
        (
            "nhs_number",
            r"(?<!\d)\d{3} \d{3} \d{4}(?!\d)",
            0.9,
            "mod11",
            "nhs",
        ),
        (
            "nhs_number",
            r"(?i:nhs)[^\d\n]{0,30}(?P<v>\d{10})(?!\d)",
            0.9,
            "mod11",
            "nhs",
        ),
    ),
    "in": (
        (
            "phone",
            r"(?<![\d+\-])(?:\+91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?![\d\-])",
            0.8,
            "national_in",
            "phone_digits",
        ),
        (
            "aadhaar",
            r"(?<!\d)[2-9]\d{3} \d{4} \d{4}(?!\d)",
            1.0,
            "verhoeff",
            "verhoeff",
        ),
        (
            "aadhaar",
            r"(?i:aadhaar|aadhar|uidai)[^\d\n]{0,30}(?P<v>[2-9]\d{11})(?!\d)",
            1.0,
            "verhoeff",
            "verhoeff",
        ),
        (
            "pan",
            r"(?<![A-Z0-9])[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z](?![A-Z0-9])",
            0.9,
            "structure",
            "",
        ),
    ),
}

IBAN_LENGTHS = {
    "AL": 28,
    "AD": 24,
    "AT": 20,
    "AZ": 28,
    "BH": 22,
    "BE": 16,
    "BA": 20,
    "BR": 29,
    "BG": 22,
    "CR": 22,
    "HR": 21,
    "CY": 28,
    "CZ": 24,
    "DK": 18,
    "DO": 28,
    "EE": 20,
    "FO": 18,
    "FI": 18,
    "FR": 27,
    "GE": 22,
    "DE": 22,
    "GI": 23,
    "GR": 27,
    "GL": 18,
    "GT": 28,
    "HU": 28,
    "IS": 26,
    "IE": 22,
    "IL": 23,
    "IT": 27,
    "JO": 30,
    "KZ": 20,
    "KW": 30,
    "LV": 21,
    "LB": 28,
    "LI": 21,
    "LT": 20,
    "LU": 20,
    "MK": 19,
    "MT": 31,
    "MR": 27,
    "MU": 30,
    "MC": 27,
    "MD": 24,
    "ME": 22,
    "NL": 18,
    "NO": 15,
    "PK": 24,
    "PS": 29,
    "PL": 28,
    "PT": 25,
    "QA": 29,
    "RO": 24,
    "SM": 27,
    "SA": 24,
    "RS": 22,
    "SK": 24,
    "SI": 19,
    "ES": 24,
    "SE": 24,
    "CH": 21,
    "TN": 24,
    "TR": 26,
    "AE": 23,
    "GB": 22,
    "VG": 24,
    "XK": 20,
}

# (brand, prefix regex over the digit string, allowed lengths)
IIN = (
    ("visa", r"4", (13, 16, 19)),
    (
        "mastercard",
        r"5[1-5]|2(?:22[1-9]|2[3-9]\d|[3-6]\d\d|7[01]\d|720)",
        (16,),
    ),
    ("amex", r"3[47]", (15,)),
    ("discover", r"6011|65|64[4-9]", (16, 19)),
    ("jcb", r"35", (16, 19)),
    ("diners", r"3(?:0[0-5]|6|8)", (14, 16, 19)),
    ("rupay", r"60|81|82", (16,)),
    (
        "maestro",
        r"5018|5020|5038|6304|6759|676[1-3]",
        (12, 13, 14, 15, 16, 17, 18, 19),
    ),
)


def export() -> Dict[str, Any]:
    def rows(rules: Tuple[Rule, ...]) -> Any:
        return [
            {
                "kind": kind,
                "regex": regex,
                "confidence": confidence,
                "reason": reason,
                "validator": validator or None,
            }
            for kind, regex, confidence, reason, validator in rules
        ]

    return {
        "generic": rows(GENERIC),
        "locales": {name: rows(rules) for name, rules in LOCALE_RULES.items()},
        "iban_lengths": IBAN_LENGTHS,
        "iin": [
            {"brand": brand, "prefix": prefix, "lengths": list(lengths)}
            for brand, prefix, lengths in IIN
        ],
    }
