"""Shannon entropy with charset-specific limits (after detect-secrets)."""

import math
import re
from collections import Counter

HEX = re.compile(r"^[0-9a-fA-F]+$")
BASE64 = re.compile(r"^[A-Za-z0-9+/=_\-]+$")
HEX_LIMIT = 3.0
BASE64_LIMIT = 4.5
OTHER_LIMIT = 3.5
# A string of n characters has at most log2(n) bits per character, so the
# charset limit is capped at this fraction of that maximum for short values.
LENGTH_FRACTION = 0.85
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


def looks_secret(value: str) -> bool:
    if not value or is_placeholder(value):
        return False
    limit = min(charset_limit(value), LENGTH_FRACTION * math.log2(len(value)))
    return shannon(value) >= limit
