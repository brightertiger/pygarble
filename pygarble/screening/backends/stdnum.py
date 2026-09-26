"""Explicit identifier extraction plus python-stdnum local validation."""

import re
from typing import Callable, Dict, Iterable, List, Tuple

from ...findings import Finding, sort_key
from ...pii.patterns import IBAN
from .._engine import _names
from ..base import check_text, optional_module

# format -> (kind, candidate expression). A broad number validator alone
# cannot establish that arbitrary digits in prose are a personal identifier.
# Bare digit formats therefore require a label or distinctive punctuation.
_RULES: Dict[str, Tuple[str, str]] = {
    "iban": ("iban", IBAN),
    "in_.pan": ("pan", r"(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])"),
    "in_.aadhaar": (
        "aadhaar",
        r"\b(?i:aadhaar|aadhar|uidai)[^\d\n]{0,30}"
        r"(?P<v>[2-9][0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4})(?!\w)",
    ),
    "us.ssn": ("ssn_us", r"(?<!\w)[0-9]{3}-[0-9]{2}-[0-9]{4}(?!\w)"),
    "gb.nhs": (
        "nhs_number",
        r"\b(?i:nhs)[^\d\n]{0,30}"
        r"(?P<v>[0-9]{3} ?[0-9]{3} ?[0-9]{4})(?!\w)",
    ),
    "br.cpf": (
        "cpf_br",
        r"(?<!\w)(?:[0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2}"
        r"|(?i:cpf)[ :#=]{0,5}(?P<v>[0-9]{11}))(?!\w)",
    ),
    "de.idnr": (
        "idnr_de",
        r"\b(?i:idnr|steuer-id|steueridentifikationsnummer)"
        r"[ :#=]{0,5}(?P<v>[0-9]{11})(?!\w)",
    ),
}


class StdnumDetector:
    category = "pii"
    kinds = frozenset(rule[0] for rule in _RULES.values())

    def __init__(self, formats: Iterable[str] = tuple(_RULES)) -> None:
        chosen = tuple(dict.fromkeys(_names("formats", formats)))
        if not chosen or set(chosen) - set(_RULES):
            raise ValueError("formats must select from: " + ", ".join(_RULES))
        self.kinds = frozenset(_RULES[name][0] for name in chosen)
        self._rules: List[
            Tuple[str, str, "re.Pattern[str]", Callable[[str], bool]]
        ] = []
        for name in chosen:
            module = optional_module("stdnum." + name, "stdnum")
            kind, pattern = _RULES[name]
            self._rules.append(
                (name, kind, re.compile(pattern), module.is_valid)
            )

    def detect(self, text: str) -> Tuple[Finding, ...]:
        check_text(text)
        found = []
        for name, kind, pattern, validate in self._rules:
            for match in pattern.finditer(text):
                start, end = (
                    match.span("v")
                    if match.groupdict().get("v") is not None
                    else match.span()
                )
                if validate(text[start:end]):
                    found.append(
                        Finding(
                            self.category,
                            kind,
                            start,
                            end,
                            0.9,
                            "stdnum:" + name,
                        )
                    )
        return tuple(sorted(found, key=sort_key))
