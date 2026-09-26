"""Immutable scan results. Findings never carry matched text."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

CATEGORIES = ("secrets", "pii", "profanity", "gibberish")


@dataclass(frozen=True)
class Finding:
    category: str
    kind: str
    start: int
    end: int
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def sort_key(finding: Finding) -> Tuple[int, int, str, str]:
    return (finding.start, finding.end, finding.category, finding.kind)


@dataclass(frozen=True)
class ScanReport:
    findings: Tuple[Finding, ...]
    flagged: bool
    length: int

    def by_category(self) -> Dict[str, Tuple[Finding, ...]]:
        grouped: Dict[str, Tuple[Finding, ...]] = {}
        for finding in self.findings:
            grouped[finding.category] = grouped.get(finding.category, ()) + (
                finding,
            )
        return grouped

    def kinds(self) -> Tuple[str, ...]:
        return tuple(sorted({finding.kind for finding in self.findings}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flagged": self.flagged,
            "length": self.length,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class Redaction:
    text: str
    findings: Tuple[Finding, ...]
    count: int
