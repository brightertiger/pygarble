"""Shannon entropy with charset-specific limits (after detect-secrets)."""

import math
import re
from collections import Counter

HEX = re.compile(r"^[0-9a-fA-F]+$")
BASE64 = re.compile(r"^[A-Za-z0-9+/=_\-]+$")
HEX_LIMIT = 3.0
BASE64_LIMIT = 4.5
OTHER_LIMIT = 3.5
# Values shorter than SHORT_MAX_LENGTH + 1 cannot reach the charset limits
# (a string of n characters has at most log2(n) bits per character), so
# they need SHORT_ENTROPY bits plus a character-class mix instead.
SHORT_MIN_LENGTH = 8
SHORT_MAX_LENGTH = 22
SHORT_ENTROPY = 3.0
SHORT_CLASSES = 3
PLACEHOLDER_WORDS = frozenset(
    {
        "changeme",
        "password",
        "secret",
        "example",
        "null",
        "none",
        "true",
        "false",
        "todo",
        "redacted",
    }
)
PLACEHOLDER_SHAPE = re.compile(r"^(?:[<{$%].*|x+|\*+|\.+|-+)$", re.I)
PLACEHOLDER_HINTS = ("example", "placeholder", "your-", "your_")


def shannon(value: str) -> float:
    """Bits per character; 0.0 for empty or single-symbol strings."""
    if not value:
        return 0.0
    total = len(value)
    return -sum(
        count / total * math.log2(count / total)
        for count in Counter(value).values()
    )


def charset_limit(value: str) -> float:
    if HEX.match(value):
        return HEX_LIMIT
    if BASE64.match(value):
        return BASE64_LIMIT
    return OTHER_LIMIT


def is_placeholder(value: str) -> bool:
    lowered = value.lower()
    if lowered in PLACEHOLDER_WORDS or PLACEHOLDER_SHAPE.match(value):
        return True
    return any(hint in lowered for hint in PLACEHOLDER_HINTS)


def _class_mix_ok(value: str) -> bool:
    if HEX.match(value):
        return any(c.isdigit() for c in value) and any(
            c.isalpha() for c in value
        )
    classes = {
        (
            "lower"
            if c.islower()
            else (
                "upper" if c.isupper() else "digit" if c.isdigit() else "other"
            )
        )
        for c in value
    }
    return len(classes) >= SHORT_CLASSES


def looks_secret(value: str) -> bool:
    if len(value) < SHORT_MIN_LENGTH or is_placeholder(value):
        return False
    if len(value) <= SHORT_MAX_LENGTH:
        return shannon(value) >= SHORT_ENTROPY and _class_mix_ok(value)
    return shannon(value) >= charset_limit(value)
