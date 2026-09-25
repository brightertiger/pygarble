"""Request-local English features with offsets into the original text."""

import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property
from typing import FrozenSet, List, Tuple


def fold_diacritics(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


@dataclass(frozen=True)
class Token:
    text: str
    folded: str
    start: int
    end: int
    structured: bool = False


@dataclass(frozen=True)
class TextFeatures:
    text: str
    allowlist: FrozenSet[str] = frozenset()

    @cached_property
    def folded(self) -> str:
        return fold_diacritics(self.text).lower()

    @cached_property
    def scrubbed(self) -> str:
        """Text with allowlisted words blanked out, offsets preserved.

        Strategies that scan raw text (keyboard rows, phonotactics,
        regex patterns) receive this instead of ``text`` so an allowlisted
        token can never contribute evidence.
        """
        if not self.allowlist:
            return self.text
        chars = list(self.text)
        for token in self.tokens:
            if token.folded in self.allowlist:
                for index in range(token.start, token.end):
                    chars[index] = " "
        return "".join(chars)

    @cached_property
    def ascii_words(self) -> Tuple[str, ...]:
        return tuple(
            word
            for token in self.tokens
            if not token.structured and token.folded not in self.allowlist
            for word in re.findall(r"[a-z]+", token.folded)
        )

    @cached_property
    def tokens(self) -> Tuple[Token, ...]:
        tokens: List[Token] = []
        for chunk in re.finditer(r"\S+", self.text):
            raw = chunk.group()
            lower = raw.lower()
            structured = (
                any(c.isdigit() for c in raw)
                or "://" in lower
                or "@" in lower
                or lower.startswith(("www.", "data:", "/", "./", "../"))
                or "\\" in raw
                or re.search(r"[a-z][A-Z]", raw) is not None
            )
            start = None
            for i, char in enumerate(raw + " "):
                if char.isalpha() or (
                    start is not None and unicodedata.combining(char)
                ):
                    if start is None:
                        start = i
                elif start is not None:
                    word = raw[start:i]
                    tokens.append(
                        Token(
                            word,
                            fold_diacritics(word).lower(),
                            chunk.start() + start,
                            chunk.start() + i,
                            structured,
                        )
                    )
                    start = None
        return tuple(tokens)

    @cached_property
    def novel(self) -> Tuple[Token, ...]:
        from .data import ENGLISH_WORDS

        return tuple(
            token
            for token in self.tokens
            if not token.structured
            and not (token.text.isupper() and len(token.folded) <= 6)
            and token.folded not in ENGLISH_WORDS
            and token.folded not in self.allowlist
        )

    @cached_property
    def bigram_stats(self) -> Tuple[float, int]:
        from .scoring import bigram_stats

        return bigram_stats(self.ascii_words)
