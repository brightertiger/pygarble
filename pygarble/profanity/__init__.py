"""Rule-based profanity detection with obfuscation handling."""

import re
from functools import lru_cache
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..findings import Finding, sort_key
from .normalize import TOKEN_RE, collapse_runs, has_long_run, normalize_token
from .wordlist import EMBEDDED, PHRASES, PROFANITY_MILD, PROFANITY_STRONG

CATEGORY = "profanity"
KIND = "profanity"
ALL_KINDS: FrozenSet[str] = frozenset({KIND})
TIERS = ("strong", "mild")
_STRONG = frozenset(PROFANITY_STRONG)
_MILD = frozenset(PROFANITY_MILD)
_WILD_CHARS = frozenset("*#@$!")
_WILD_TOKEN = re.compile(r"^[\w*#@$!]+$")
_GAP = re.compile(r"^[ .\-]{1,3}$")
_POSSESSIVE = re.compile(r"['’][sS]$")
_SPACED_FILLERS = frozenset({"a", "i"})  # one-letter words before a run
_PHRASE_MAX = max(len(p) for p in PHRASES)
_BY_LENGTH: Dict[int, Tuple[str, ...]] = {}
for _word in PROFANITY_STRONG:
    _BY_LENGTH[len(_word)] = _BY_LENGTH.get(len(_word), ()) + (_word,)
del _word

Token = Tuple[int, int, str, str]  # start, end, raw lookup form, normalised


@lru_cache(maxsize=None)
def _clean_words(length: int) -> Tuple[str, ...]:
    """Unlisted English words of one length, loaded on first use."""
    from ..data import ENGLISH_WORDS

    return tuple(
        sorted(
            word
            for word in ENGLISH_WORDS
            if len(word) == length and word not in _STRONG | _MILD
        )
    )


def _fits(masked: str, word: str) -> bool:
    return all(c in _WILD_CHARS or c == w for c, w in zip(masked, word))


@lru_cache(maxsize=4096)
def _masked_candidates(masked: str) -> Tuple[Tuple[str, ...], bool]:
    """Strong words a lowercase masked token fits, and whether a clean
    English word fits it too. Independent of any allowlist."""
    if not _WILD_TOKEN.match(masked) or not (set(masked) & _WILD_CHARS):
        return ((), False)
    if sum(1 for c in masked if c.isalpha()) < 2:
        return ((), False)
    strong = tuple(
        word for word in _BY_LENGTH.get(len(masked), ()) if _fits(masked, word)
    )
    if not strong:
        return ((), False)
    clean = any(_fits(masked, word) for word in _clean_words(len(masked)))
    return (strong, clean)


class ProfanityDetector:
    category = CATEGORY

    def __init__(
        self,
        allowlist: Optional[Iterable[str]] = None,
        tiers: Iterable[str] = TIERS,
        obfuscation: bool = True,
    ) -> None:
        chosen = tuple(dict.fromkeys(tiers))
        bad = [t for t in chosen if t not in TIERS]
        if bad or not chosen:
            raise ValueError(
                f"tiers must be a non-empty subset of {', '.join(TIERS)}"
            )
        self.strong = _STRONG if "strong" in chosen else frozenset()
        self.mild = _MILD if "mild" in chosen else frozenset()
        self.allowlist = frozenset(
            " ".join(normalize_token(word) for word in entry.split())
            for entry in (allowlist or ())
        )
        self.obfuscation = bool(obfuscation)

    def _tier(self, word: str) -> Optional[Tuple[float, str]]:
        if word in self.strong:
            return (1.0, "strong")
        if word in self.mild:
            return (0.7, "mild")
        return None

    def _single(self, token: Token) -> Optional[Tuple[float, str]]:
        raw, norm = token[2], token[3]
        if not norm or norm in self.allowlist:
            return None
        tier = self._tier(norm)
        if tier is not None:
            return tier
        if has_long_run(norm):
            for width in (2, 1):
                collapsed = collapse_runs(norm, width)
                if collapsed in self.allowlist:
                    return None
                collapsed_tier = self._tier(collapsed)
                if collapsed_tier is not None:
                    return (min(0.8, collapsed_tier[0]), "elongated")
        if self.strong and any(
            w in norm and w not in self.allowlist for w in EMBEDDED
        ):
            return (0.8, "embedded")
        if self.obfuscation and self.strong:
            return self._wildcard(raw)
        return None

    def _wildcard(self, raw: str) -> Optional[Tuple[float, str]]:
        strong, clean = _masked_candidates(raw.lower())
        matches = [word for word in strong if word not in self.allowlist]
        if not matches:
            return None
        if len(matches) > 1 or clean:
            return (0.6, "masked_ambiguous")
        return (0.9, "masked")

    def _phrases(
        self, tokens: Sequence[Token], used: Set[int], out: List[Finding]
    ) -> None:
        if not self.strong:
            return
        norms = [t[3] for t in tokens]
        for size in range(_PHRASE_MAX, 1, -1):
            for i in range(0, len(tokens) - size + 1):
                if any(j in used for j in range(i, i + size)):
                    continue
                window = tuple(norms[i : i + size])
                if window not in PHRASES:
                    continue
                joined = "".join(window)
                if " ".join(window) in self.allowlist or (
                    joined in self.allowlist
                ):
                    continue
                confidence, reason = self._tier(joined) or (1.0, "strong")
                out.append(
                    Finding(
                        CATEGORY,
                        KIND,
                        tokens[i][0],
                        tokens[i + size - 1][1],
                        confidence,
                        reason,
                    )
                )
                used.update(range(i, i + size))

    def _spaced(
        self,
        text: str,
        tokens: Sequence[Token],
        used: Set[int],
        out: List[Finding],
    ) -> None:
        i = 0
        while i < len(tokens):
            j = i
            while (
                j + 1 < len(tokens)
                and j + 1 not in used
                and len(tokens[j][2]) == 1
                and len(tokens[j + 1][2]) == 1
                and tokens[j][2].isalpha()
                and tokens[j + 1][2].isalpha()
                and _GAP.match(text[tokens[j][1] : tokens[j + 1][0]])
            ):
                j += 1
            if j - i + 1 >= 3 and i not in used:
                hit = self._spaced_run(tokens, i, j)
                if hit is not None:
                    first, last, confidence = hit
                    out.append(
                        Finding(
                            CATEGORY,
                            KIND,
                            tokens[first][0],
                            tokens[last][1],
                            confidence,
                            "spaced",
                        )
                    )
                    used.update(range(first, last + 1))
                i = j + 1
            else:
                i += 1

    def _spaced_run(
        self, tokens: Sequence[Token], i: int, j: int
    ) -> Optional[Tuple[int, int, float]]:
        """The listed word spelled by tokens i..j, or by that run without a
        leading or trailing "a"/"I" ("what a s h i t show")."""
        lead = tokens[i][3] in _SPACED_FILLERS
        trail = tokens[j][3] in _SPACED_FILLERS
        spans = [(i, j), (i + 1, j) if lead else None]
        spans += [(i, j - 1) if trail else None]
        spans += [(i + 1, j - 1) if lead and trail else None]
        for span in spans:
            if span is None or span[1] - span[0] + 1 < 3:
                continue
            first, last = span
            joined = "".join(t[3] for t in tokens[first : last + 1])
            tier = None if joined in self.allowlist else self._tier(joined)
            if tier is not None:
                return (first, last, min(0.8, tier[0]))
        return None

    def detect(self, text: str) -> Tuple[Finding, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: List[Token] = []
        for match in TOKEN_RE.finditer(text):
            # Strip "!" off the end ("Shit!") but keep it inside ("sh!t");
            # the span ends where the stripped token ends.
            bare = match.group().rstrip("!")
            raw = _POSSESSIVE.sub("", bare)
            # A digits-only token ("455") never matches: no "ass".
            norm = "" if raw.isdigit() else normalize_token(raw)
            end = match.start() + len(bare)
            tokens.append((match.start(), end, raw, norm))
        out: List[Finding] = []
        used: Set[int] = set()
        self._phrases(tokens, used, out)
        if self.obfuscation:
            self._spaced(text, tokens, used, out)
        for index, token in enumerate(tokens):
            if index in used:
                continue
            verdict = self._single(token)
            if verdict is not None:
                out.append(
                    Finding(
                        CATEGORY,
                        KIND,
                        token[0],
                        token[1],
                        verdict[0],
                        verdict[1],
                    )
                )
        return tuple(sorted(out, key=sort_key))


def detect(text: str, **kwargs: Any) -> Tuple[Finding, ...]:
    return ProfanityDetector(**kwargs).detect(text)


__all__ = ["ProfanityDetector", "detect", "ALL_KINDS", "TIERS"]
