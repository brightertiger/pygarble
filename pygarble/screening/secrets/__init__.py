"""Deterministic secret detection: known prefixes and keyword entropy."""

import base64
import binascii
import json
import re
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from ...findings import Finding, sort_key
from .entropy import (
    BASE64_LIMIT,
    HEX_LIMIT,
    is_placeholder,
    looks_secret,
    shannon,
)
from .patterns import ALL_KINDS, KEYWORDS, KNOWN_PATTERNS, LEFT, RIGHT

CATEGORY = "secrets"
_VALUE_GROUP = re.compile(r"\(\?P<v>")


def _assemble() -> Tuple[List[str], List[Dict[str, Any]]]:
    parts = []
    table = []
    for index, entry in enumerate(KNOWN_PATTERNS):
        body = (
            entry["regex"] if entry["raw"] else LEFT + entry["regex"] + RIGHT
        )
        body = _VALUE_GROUP.sub(f"(?P<v{index}>", body)
        parts.append(f"(?P<k{index}>{body})")
        table.append(entry)
    return parts, table


_PARTS, _TABLE = _assemble()
_KNOWN = re.compile("|".join(_PARTS))

# Anchors: for each known pattern, literals of which every match contains
# at least one ("cs": case-sensitive; "ci": lowercase ASCII matched under
# the same (?i:...) folding the pattern uses). A pattern whose anchor is
# absent from a text cannot match anywhere in it, so dropping it from the
# alternation leaves every match (position, alternative, groups) unchanged.
_ANCHORS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("cs", ("AKIA", "ASIA", "ABIA", "ACCA")),  # aws_access_key_id
    ("ci", ("aws",)),  # aws_secret_access_key: (?i:aws)
    ("cs", ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")),
    ("cs", ("glpat-",)),
    ("cs", ("xox",)),  # xox[abprs]-
    ("cs", ("https://hooks.slack.com/services/",)),
    ("cs", ("_live_",)),  # (?:sk|rk)_live_
    ("cs", ("_test_",)),  # (?:sk|rk)_test_
    ("cs", ("AIza",)),
    ("cs", ("sk-",)),  # openai sk-(?:proj-|svcacct-)?...T3BlbkFJ
    ("cs", ("sk-",)),  # openai sk-[A-Za-z0-9]{48}
    ("cs", ("sk-ant-",)),
    ("cs", ("hf_",)),
    ("cs", ("npm_",)),
    ("cs", ("pypi-AgEIcHlwaS5vcmc",)),
    ("cs", ("SG.",)),
    ("cs", ("eyJ",)),
    ("cs", ("-----BEGIN ",)),
    ("cs", ("://",)),  # url_credentials: scheme://user:pass@
    ("ci", ("bearer",)),  # bearer_token: (?i:bearer)
)
if len(_ANCHORS) != len(KNOWN_PATTERNS):
    raise RuntimeError("_ANCHORS must have one entry per known pattern")
_CI_ANCHORS = {
    index: re.compile("(?i:" + "|".join(literals) + ")")
    for index, (mode, literals) in enumerate(_ANCHORS)
    if mode == "ci"
}
_CS_LITERALS = tuple(
    (literal, index)
    for index, (mode, literals) in enumerate(_ANCHORS)
    if mode == "cs"
    for literal in literals
)
_CI_LITERALS = tuple(
    (index, literals)
    for index, (mode, literals) in enumerate(_ANCHORS)
    if mode == "ci"
)
# Every keyword holds one of these stems ("password" and "passwd" hold
# "passw"; the compound keywords end in "key", "token" or "secret").
_KEYWORD_STEMS = ("passw", "pwd", "secret", "token", "key")
_KEYWORD_STEM_RE = re.compile("(?i:" + "|".join(_KEYWORD_STEMS) + ")")
# The whole identifier (e.g. DB_PASSWORD, MY_API_KEY) must contain a keyword.
# It is matched inside a lookahead, which Python treats as atomic, so each
# identifier is scanned once and a failed separator never backtracks into it.
_KEYWORD = re.compile(
    r"(?<![A-Za-z0-9_])(?=(?P<name>[A-Za-z0-9_]*?(?i:"
    + "|".join(KEYWORDS)
    + r")[A-Za-z0-9_]*))(?P=name)"
    r"[\"']?\s*(?:=>|[:=])\s*[\"']?(?P<v>[^\s\"',;&()]{8,})"
)
_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=_\-])[A-Za-z0-9+/=_\-]{32,}")
_HEX_TOKEN = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{32,}(?![0-9A-Fa-f])")


def _lowered(text: str) -> Optional[str]:
    """text.lower() for ASCII text, where a (?i:...) literal matches
    exactly where its lowercase form occurs in the lowered text; None for
    other text, which falls back to the (?i:...) regex."""
    return text.lower() if text.isascii() else None


def _live_patterns(text: str, lowered: Optional[str]) -> Tuple[int, ...]:
    """Indices of the known patterns whose anchor occurs in text."""
    live = set()
    for literal, index in _CS_LITERALS:
        if literal in text:
            live.add(index)
    for index, literals in _CI_LITERALS:
        if lowered is not None:
            if any(literal in lowered for literal in literals):
                live.add(index)
        elif _CI_ANCHORS[index].search(text) is not None:
            live.add(index)
    return tuple(sorted(live))


@lru_cache(maxsize=256)
def _known_subset(indices: Tuple[int, ...]) -> "re.Pattern[str]":
    """The known-pattern alternation restricted to indices, in order."""
    if len(indices) == len(_PARTS):
        return _KNOWN
    return re.compile("|".join(_PARTS[i] for i in indices))


def _has_keyword_stem(text: str, lowered: Optional[str]) -> bool:
    if lowered is not None:
        return any(stem in lowered for stem in _KEYWORD_STEMS)
    return _KEYWORD_STEM_RE.search(text) is not None


def _jwt_header_ok(token: str) -> bool:
    head = token.split(".", 1)[0]
    padded = head + "=" * (-len(head) % 4)
    try:
        header = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        return isinstance(header, dict) and "alg" in header
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False


_NESTING = frozenset({"bearer_token", "url_credentials"})
_KNOWN_KINDS = frozenset(entry["kind"] for entry in KNOWN_PATTERNS)


def _located(match: "re.Match[str]") -> Tuple[Dict[str, Any], int, int]:
    """The table entry of a known-alternation match and its value span."""
    index = int(str(match.lastgroup)[1:])
    group = f"v{index}"
    if group in match.groupdict():
        start, end = match.span(group)
    else:
        start, end = match.span()
    return _TABLE[index], start, end


def _finding(
    entry: Dict[str, Any], start: int, end: int, text: str
) -> Optional[Finding]:
    value = text[start:end]
    # Bearer values are not entropy-gated, only placeholder-gated.
    if entry.get("filter") == "placeholder" and is_placeholder(value):
        return None
    confidence = entry["confidence"]
    if entry.get("verify") == "jwt_header" and not _jwt_header_ok(value):
        confidence = 0.8
    return Finding(
        CATEGORY, entry["kind"], start, end, confidence, entry["reason"]
    )


class _Coverage:
    """Overlap test against fixed spans for queries in ascending start
    order: one forward sweep over the merged spans, linear overall."""

    def __init__(self, spans: Iterable[Tuple[int, int]]) -> None:
        merged: List[List[int]] = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self.spans = merged
        self.index = 0

    def __call__(self, start: int, end: int) -> bool:
        spans = self.spans
        while self.index < len(spans) and spans[self.index][1] <= start:
            self.index += 1
        return self.index < len(spans) and spans[self.index][0] < end


def _validate_kinds(
    kinds: Optional[Iterable[str]], exclude: Iterable[str]
) -> FrozenSet[str]:
    for value in (kinds, exclude):
        if isinstance(value, str):
            raise ValueError(
                "kinds must be an iterable of kind names, not a string"
            )
    chosen = frozenset(ALL_KINDS if kinds is None else kinds)
    excluded = frozenset(exclude)
    unknown = sorted((chosen | excluded) - ALL_KINDS)
    if unknown:
        raise ValueError(
            f"unknown secrets kind(s): {', '.join(unknown)}; "
            f"valid kinds: {', '.join(sorted(ALL_KINDS))}"
        )
    return chosen - excluded


class SecretsDetector:
    category = CATEGORY

    def __init__(
        self,
        kinds: Optional[Iterable[str]] = None,
        exclude_kinds: Iterable[str] = (),
        without_context: bool = False,
    ) -> None:
        self.kinds = _validate_kinds(kinds, exclude_kinds)
        self.without_context = bool(without_context)

    def _known(
        self, text: str, lowered: Optional[str], kinds: FrozenSet[str]
    ) -> List[Finding]:
        """Known-shape findings of the given kinds, in text order.

        Only the selected kinds enter the alternation, so an unselected
        kind never consumes text. A bearer or URL-credentials value is
        searched again for vendor tokens and JWTs, which are reported as
        additional findings nested inside it.
        """
        found: List[Finding] = []
        live = tuple(
            i
            for i in _live_patterns(text, lowered)
            if _TABLE[i]["kind"] in kinds
        )
        if not live:
            return found
        inner = tuple(i for i in live if _TABLE[i]["kind"] not in _NESTING)
        inner_pattern = _known_subset(inner) if inner else None
        for match in _known_subset(live).finditer(text):
            entry, start, end = _located(match)
            finding = _finding(entry, start, end, text)
            if finding is not None:
                found.append(finding)
            if inner_pattern is None or entry["kind"] not in _NESTING:
                continue
            # Nested search runs even when the outer value is rejected as
            # a placeholder: the outer match consumed the text either way.
            for sub in inner_pattern.finditer(text, start, end):
                nested = _finding(*_located(sub), text)
                if nested is not None:
                    found.append(nested)
        return found

    def _keyword(self, text: str, lowered: Optional[str]) -> List[Finding]:
        found: List[Finding] = []
        if not _has_keyword_stem(text, lowered):
            return found
        for match in _KEYWORD.finditer(text):
            value = match.group("v")
            if looks_secret(value):
                start, end = match.span("v")
                found.append(
                    Finding(
                        CATEGORY,
                        "generic_secret",
                        start,
                        end,
                        0.6,
                        "keyword_entropy",
                    )
                )
        return found

    def _standalone(
        self, text: str, taken: List[Tuple[int, int]]
    ) -> List[Finding]:
        found: List[Finding] = []
        for pattern, limit in (
            (_HEX_TOKEN, HEX_LIMIT),
            (_BASE64_TOKEN, BASE64_LIMIT),
        ):
            # Matches of one pattern never overlap each other, so checking
            # them against the spans taken before this pattern suffices.
            covered = _Coverage(taken)
            for match in pattern.finditer(text):
                start, end = match.span()
                if covered(start, end) or shannon(match.group()) < limit:
                    continue
                found.append(
                    Finding(
                        CATEGORY,
                        "high_entropy_string",
                        start,
                        end,
                        0.5,
                        "entropy",
                    )
                )
            taken = taken + [(f.start, f.end) for f in found]
        return found

    def detect(self, text: str) -> Tuple[Finding, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        lowered = _lowered(text)
        known_kinds = self.kinds & _KNOWN_KINDS
        findings = self._known(text, lowered, known_kinds)
        standalone = self.without_context and (
            "high_entropy_string" in self.kinds
        )
        if "generic_secret" not in self.kinds and not standalone:
            return tuple(sorted(set(findings), key=sort_key))
        # The context kinds defer to every known finding, selected or not,
        # so selecting a kind only filters the default output.
        if known_kinds == _KNOWN_KINDS:
            every_known = findings
        else:
            every_known = self._known(text, lowered, _KNOWN_KINDS)
        # A known-prefix finding already names the value; drop keyword
        # findings that overlap one so a value is reported once.
        covered = _Coverage([(f.start, f.end) for f in every_known])
        keyword = [
            f
            for f in self._keyword(text, lowered)
            if not covered(f.start, f.end)
        ]
        if "generic_secret" in self.kinds:
            findings.extend(keyword)
        if standalone:
            taken = [(f.start, f.end) for f in every_known + keyword]
            findings.extend(self._standalone(text, taken))
        return tuple(sorted(set(findings), key=sort_key))


def detect(text: str, **kwargs: Any) -> Tuple[Finding, ...]:
    return SecretsDetector(**kwargs).detect(text)


__all__ = ["SecretsDetector", "detect", "ALL_KINDS"]
