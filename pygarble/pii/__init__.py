"""Structural PII detection with checksums and locale packs."""

import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from ..findings import Finding, sort_key
from .checksums import iban_mod97, luhn, nhs_mod11, verhoeff
from .patterns import (
    EMAIL_EXCLUDED_TLDS,
    GENERIC,
    IBAN_LENGTHS,
    IIN,
    LOCALE_RULES,
    Rule,
)

CATEGORY = "pii"
LOCALES: Tuple[str, ...] = ("us", "uk", "in")
ALL_KINDS: FrozenSet[str] = frozenset(
    [rule[0] for rule in GENERIC]
    + [rule[0] for rules in LOCALE_RULES.values() for rule in rules]
)
_GENERIC_KINDS = frozenset(rule[0] for rule in GENERIC)
_SEPARATORS = re.compile(r"[ \-]")
_IIN = [(brand, re.compile(prefix), lengths) for brand, prefix, lengths in IIN]
_COMPILED: Dict[str, List[Tuple[Rule, "re.Pattern[str]"]]] = {}


def _compiled(scope: str) -> List[Tuple[Rule, "re.Pattern[str]"]]:
    if scope not in _COMPILED:
        rules = GENERIC if scope == "generic" else LOCALE_RULES[scope]
        _COMPILED[scope] = [(rule, re.compile(rule[1])) for rule in rules]
    return _COMPILED[scope]


def _card_reason(value: str) -> Optional[str]:
    digits = _SEPARATORS.sub("", value)
    if not 13 <= len(digits) <= 19 or not luhn(digits):
        return None
    for brand, prefix, lengths in _IIN:
        if prefix.match(digits) and len(digits) in lengths:
            return f"luhn_{brand}"
    return None


def _validate(validator: str, value: str) -> Optional[str]:
    """Return a reason suffix override, "" to keep the rule's reason, or
    None to reject the match."""
    if validator == "":
        return ""
    if validator == "email":
        tld = value.rsplit(".", 1)[-1].lower()
        return None if tld in EMAIL_EXCLUDED_TLDS else ""
    if validator == "phone_digits":
        digits = re.sub(r"\D", "", value)
        return "" if 7 <= len(digits) <= 15 else None
    if validator == "card":
        return _card_reason(value)
    if validator == "iban":
        compact = value.replace(" ", "")
        expected = IBAN_LENGTHS.get(compact[:2])
        if expected != len(compact) or not iban_mod97(compact):
            return None
        return ""
    if validator == "nhs":
        return "" if nhs_mod11(re.sub(r"\D", "", value)) else None
    if validator == "verhoeff":
        return "" if verhoeff(re.sub(r"\D", "", value)) else None
    raise ValueError(f"unknown validator {validator!r}")


def _names(label: str, value: Iterable[str]) -> Iterable[str]:
    if isinstance(value, str):
        raise ValueError(f"{label} must be an iterable of names, not a string")
    return value


def _outranks(finding: Finding, current: Finding) -> bool:
    """Higher confidence wins; on a tie a generic kind beats a locale kind."""
    if finding.confidence != current.confidence:
        return finding.confidence > current.confidence
    return (
        finding.kind in _GENERIC_KINDS and current.kind not in _GENERIC_KINDS
    )


class PIIDetector:
    category = CATEGORY

    def __init__(
        self,
        kinds: Optional[Iterable[str]] = None,
        exclude_kinds: Iterable[str] = (),
        locales: Iterable[str] = LOCALES,
    ) -> None:
        chosen = frozenset(
            ALL_KINDS if kinds is None else _names("kinds", kinds)
        )
        excluded = frozenset(_names("exclude_kinds", exclude_kinds))
        unknown = sorted((chosen | excluded) - ALL_KINDS)
        if unknown:
            raise ValueError(
                f"unknown pii kind(s): {', '.join(unknown)}; "
                f"valid kinds: {', '.join(sorted(ALL_KINDS))}"
            )
        self.kinds = chosen - excluded
        wanted = tuple(dict.fromkeys(_names("locales", locales)))
        bad = sorted(set(wanted) - set(LOCALES))
        if bad:
            raise ValueError(
                f"unknown locale(s): {', '.join(bad)}; "
                f"valid locales: {', '.join(LOCALES)}"
            )
        self.locales = wanted

    def _run(self, text: str, scope: str, found: List[Finding]) -> None:
        for rule, pattern in _compiled(scope):
            kind, _, confidence, reason, validator = rule
            if kind not in self.kinds:
                continue
            for match in pattern.finditer(text):
                if "v" in match.groupdict() and match.group("v") is not None:
                    start, end = match.span("v")
                else:
                    start, end = match.span()
                verdict = _validate(validator, text[start:end])
                if verdict is None:
                    continue
                found.append(
                    Finding(
                        CATEGORY,
                        kind,
                        start,
                        end,
                        confidence,
                        verdict or reason,
                    )
                )

    def detect(self, text: str) -> Tuple[Finding, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        found: List[Finding] = []
        self._run(text, "generic", found)
        for locale in self.locales:
            self._run(text, locale, found)
        best: Dict[Tuple[int, int], Finding] = {}
        for finding in sorted(found, key=sort_key):
            key = (finding.start, finding.end)
            current = best.get(key)
            if current is None or _outranks(finding, current):
                best[key] = finding
        return tuple(sorted(best.values(), key=sort_key))


def detect(text: str, **kwargs: Any) -> Tuple[Finding, ...]:
    return PIIDetector(**kwargs).detect(text)


__all__ = ["PIIDetector", "detect", "ALL_KINDS", "LOCALES"]
