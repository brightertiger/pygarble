"""Fold case, diacritics and leetspeak so obfuscated tokens compare equal."""

import re
import unicodedata

LEET_MAP = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
    "+": "t",
}
TOKEN_RE = re.compile(r"[\w*#@$!|+'’]+")
_LONG_RUN = re.compile(r"([a-z])\1\1")
_RUN = re.compile(r"([a-z])\1+")
_TRANSLATE = str.maketrans(LEET_MAP)


def normalize_token(token: str) -> str:
    folded = unicodedata.normalize("NFKD", token.casefold())
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    mapped = stripped.translate(_TRANSLATE)
    return "".join(c for c in mapped if "a" <= c <= "z")


def has_long_run(token: str) -> bool:
    return _LONG_RUN.search(token) is not None


def collapse_runs(token: str, to: int) -> str:
    if to < 1:
        raise ValueError("to must be >= 1")
    return _RUN.sub(lambda m: m.group(1) * min(to, len(m.group())), token)
