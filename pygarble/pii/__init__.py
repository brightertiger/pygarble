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

# Prechecks: cheap tests that every match of a rule implies, so a text
# that fails one cannot match and the rule's regex is skipped. A rule's
# tests run in order and stop at the first failure. Each test is one of:
#   ("in", s): the literal s occurs in the text;
#   ("run", k): k consecutive \d characters occur;
#   ("count", k): at least k \d characters occur;
#   ("re", r): regex r (compiled without flags) is found.
# Every test is implied by the rule: its literal, digit run or regex is a
# contiguous piece of the rule's regex with lookarounds dropped (or a
# wider form of it). \d and \s are the same Unicode classes the rules use,
# [A-Z] is ASCII as in the rules, (?i:...) folds case exactly as the rule's
# own (?i:...) does, and no rule sets a flag.
Check = Tuple[str, Any]


def _re(pattern: str) -> Check:
    return ("re", re.compile(pattern))


# One tuple per rule, in the order of GENERIC and LOCALE_RULES; each
# comment quotes the part of the rule the tests are taken from.
_PRECHECKS: Dict[str, Tuple[Tuple[Check, ...], ...]] = {
    "generic": (
        (("in", "@"),),  # email: "@"
        (("in", "+"), ("run", 7)),  # phone: \+[1-9]\d{6,14}
        (("count", 13),),  # card: (?:\d[ \-]?){12,18}\d
        (("run", 2), _re(r"[A-Z]{2}\d{2}")),  # iban: [A-Z]{2}\d{2}
        # ipv4: four octets of one to three digits joined by "."
        (("in", "."), _re(r"\d\.\d{1,3}\.\d{1,3}\.\d")),
        # ipv6: every branch starts with a hex digit and ":", or "::"
        (("in", ":"), _re(r"[0-9A-Fa-f]:|::")),
    ),
    "us": (
        # phone: [2-9]\d{2}[\s.\-]\d{4}
        (("run", 4), _re(r"\d{3}[\s.\-]\d{4}")),
        # ssn: \d{2}\1\d{4}, where \1 is [\-\s]
        (("run", 4), _re(r"\d{2}[\-\s]\d{4}")),
        # ssn keyword: (?i:ssn|social security), then \d{3}\d{2}\d{4}
        (("run", 9), _re(r"(?i:ssn|social security)")),
    ),
    "uk": (
        (("run", 4),),  # phone: every branch holds \d{4}
        (("run", 2), _re(r"[A-Z] ?\d{2}")),  # nino: [A-...] ?\d{2}
        # nhs: \d{3} \d{3} \d{4}
        (("in", " "), ("run", 4), _re(r"\d{3} \d{4}")),
        # nhs keyword: (?i:nhs), then \d{3} ?\d{3} ?\d{4}
        (("run", 4), _re(r"(?i:nhs)")),
    ),
    "in": (
        (("run", 5),),  # phone: [6-9]\d{4}[\s\-]?\d{5}
        # aadhaar: [2-9]\d{3} \d{4} \d{4}
        (("in", " "), ("run", 4), _re(r"\d{4} \d{4}")),
        # aadhaar keyword: (?i:aadhaar|aadhar|uidai), then [2-9]\d{11}
        (("run", 12), _re(r"(?i:aadhaar|aadhar|uidai)")),
        (("run", 4), _re(r"[A-Z]\d{4}[A-Z]")),  # pan: [A-Z]\d{4}[A-Z]
    ),
}
if len(_PRECHECKS["generic"]) != len(GENERIC) or any(
    len(_PRECHECKS[name]) != len(rules) for name, rules in LOCALE_RULES.items()
):
    raise RuntimeError("_PRECHECKS must have one entry per rule")
_ANY_DIGIT = re.compile(r"\d")
# ASCII bytes: each digit becomes "0", every other byte "x".
_DIGIT_MAP = bytes(
    ord("0") if chr(i) in "0123456789" else ord("x") for i in range(256)
)


class _Facts:
    r"""Precheck results for one text, each computed at most once. For
    ASCII text \d matches exactly 0-9, so digit tests run on a byte copy
    with every digit mapped to "0"; other text uses the \d regex."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.memo: Dict[Check, bool] = {}
        self.digits: Optional[bytes] = None
        if text.isascii():
            self.digits = text.encode("ascii").translate(_DIGIT_MAP)

    def _test(self, check: Check) -> bool:
        op, arg = check
        if op == "in":
            return bool(arg in self.text)
        if op == "re":
            return arg.search(self.text) is not None
        if op == "run":
            if self.digits is not None:
                return bool(b"0" * arg in self.digits)
            return re.search(r"\d{%d}" % arg, self.text) is not None
        if op == "count":
            if self.digits is not None:
                return bool(self.digits.count(b"0") >= arg)
            return bool(len(_ANY_DIGIT.findall(self.text)) >= arg)
        raise ValueError(f"unknown precheck {op!r}")

    def passes(self, checks: Tuple[Check, ...]) -> bool:
        for check in checks:
            result = self.memo.get(check)
            if result is None:
                result = self.memo[check] = self._test(check)
            if not result:
                return False
        return True


def _compiled(scope: str) -> List[Tuple[Rule, "re.Pattern[str]"]]:
    if scope not in _COMPILED:
        rules = GENERIC if scope == "generic" else LOCALE_RULES[scope]
        _COMPILED[scope] = [(rule, re.compile(rule[1])) for rule in rules]
    return _COMPILED[scope]


_Active = Tuple[Rule, "re.Pattern[str]", Tuple[Check, ...]]


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


def locale_kinds(locales: Iterable[str]) -> FrozenSet[str]:
    """The kinds some rule can report under these locales."""
    kinds = set(_GENERIC_KINDS)
    for locale in locales:
        kinds.update(rule[0] for rule in LOCALE_RULES[locale])
    return frozenset(kinds)


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


def _strictly_inside(inner: Finding, outer: Finding) -> bool:
    return (
        outer.start <= inner.start
        and inner.end <= outer.end
        and (inner.start, inner.end) != (outer.start, outer.end)
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
        self._active: List[_Active] = []
        self._active_key: Optional[Tuple[Any, ...]] = None

    def _build_rules(self) -> List[_Active]:
        active: List[_Active] = []
        for scope in ("generic",) + tuple(self.locales):
            checks = _PRECHECKS[scope]
            for (rule, pattern), check in zip(_compiled(scope), checks):
                if rule[0] in self.kinds:
                    active.append((rule, pattern, check))
        return active

    def _rules(self) -> List[_Active]:
        """The selected rules in the order the detector runs them. Cached
        only while kinds is a frozenset and locales a tuple; a mutable
        value could change in place, so it is re-read on every call."""
        if not (
            isinstance(self.kinds, frozenset)
            and isinstance(self.locales, tuple)
        ):
            return self._build_rules()
        key = (self.kinds, self.locales)
        if self._active_key != key:
            self._active = self._build_rules()
            self._active_key = key
        return self._active

    def _run(self, text: str, found: List[Finding]) -> None:
        facts = _Facts(text)
        for rule, pattern, checks in self._rules():
            if not facts.passes(checks):
                continue
            kind, _, confidence, reason, validator = rule
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
        self._run(text, found)
        best: Dict[Tuple[int, int], Finding] = {}
        for finding in sorted(found, key=sort_key):
            key = (finding.start, finding.end)
            current = best.get(key)
            if current is None or _outranks(finding, current):
                best[key] = finding
        phones = [f for f in best.values() if f.kind == "phone"]
        kept = [
            f
            for f in best.values()
            if not (f.kind == "nhs_number" and f.reason == "mod11")
            or not any(_strictly_inside(f, phone) for phone in phones)
        ]
        return tuple(sorted(kept, key=sort_key))


def detect(text: str, **kwargs: Any) -> Tuple[Finding, ...]:
    return PIIDetector(**kwargs).detect(text)


__all__ = ["PIIDetector", "detect", "ALL_KINDS", "LOCALES"]
