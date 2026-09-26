"""Fold case, diacritics and leetspeak so obfuscated tokens compare equal."""

import re
import unicodedata
from functools import lru_cache

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
# ASCII in, ASCII out: drop every ASCII character that is not a-z.
_KEEP_AZ = str.maketrans(
    {chr(i): None for i in range(128) if not "a" <= chr(i) <= "z"}
)
# Tokens up to this length are cached; longer ones are rare and would pin
# large strings in the cache.
_CACHE_MAX_LENGTH = 64


def _normalize(token: str) -> str:
    if token.isascii():
        # For ASCII, casefold() is lower(), NFKD changes nothing and no
        # character is combining; LEET_MAP maps ASCII to ASCII.
        return token.lower().translate(_TRANSLATE).translate(_KEEP_AZ)
    folded = unicodedata.normalize("NFKD", token.casefold())
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    mapped = stripped.translate(_TRANSLATE)
    return "".join(c for c in mapped if "a" <= c <= "z")


_normalize_cached = lru_cache(maxsize=65536)(_normalize)


def normalize_token(token: str) -> str:
    """Casefold, strip diacritics, map leetspeak and keep only a-z."""
    if len(token) > _CACHE_MAX_LENGTH:
        return _normalize(token)
    return _normalize_cached(token)


def has_long_run(token: str) -> bool:
    return _LONG_RUN.search(token) is not None


def collapse_runs(token: str, to: int) -> str:
    if to < 1:
        raise ValueError("to must be >= 1")
    return _RUN.sub(lambda m: m.group(1) * min(to, len(m.group())), token)
