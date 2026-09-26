"""Shared execution and redaction for screening and the legacy scanner."""

from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from ..findings import CATEGORIES, Finding, Redaction, ScanReport, sort_key
from ..redaction import render
from .base import Detector


def _names(label: str, value: Iterable[str]) -> Iterable[str]:
    if isinstance(value, str):
        raise ValueError(f"{label} must be an iterable of names, not a string")
    return value


class ScanEngine:
    categories: Tuple[str, ...]
    min_confidence: float
    max_input_length: Optional[int]
    _detectors: List[Detector]

    def _check(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if (
            self.max_input_length is not None
            and len(text) > self.max_input_length
        ):
            raise ValueError("text exceeds max_input_length")

    def _collect(
        self, text: str, detectors: Sequence[Detector]
    ) -> Tuple[Finding, ...]:
        findings: List[Finding] = []
        for detector in detectors:
            findings.extend(detector.detect(text))
        return _drop_url_emails(tuple(sorted(findings, key=sort_key)))

    def _scan_one(self, text: str) -> ScanReport:
        self._check(text)
        ordered = self._collect(text, self._detectors)
        if not ordered:
            return ScanReport((), False, len(text))
        flagged = any(
            f.confidence >= self.min_confidence or f.category == "gibberish"
            for f in ordered
        )
        return ScanReport(ordered, flagged, len(text))

    def scan(self, text: str) -> ScanReport:
        return self._scan_one(text)

    def scan_batch(self, texts: Sequence[str]) -> List[ScanReport]:
        if isinstance(texts, str) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        return [self._scan_one(text) for text in texts]

    def iter_scan(self, texts: Iterable[str]) -> Iterator[ScanReport]:
        if isinstance(texts, str):
            raise TypeError("texts must be an iterable of strings")
        return (self._scan_one(text) for text in texts)

    def redact(
        self,
        text: str,
        *,
        mode: str = "placeholder",
        placeholder: str = "[{KIND}]",
        mask_char: str = "*",
        categories: Optional[Iterable[str]] = None,
    ) -> Redaction:
        allowed = (
            frozenset(c for c in self.categories if c != "gibberish")
            if categories is None
            else frozenset(_names("categories", categories))
        )
        unknown = sorted(allowed - frozenset(CATEGORIES))
        if unknown:
            raise ValueError(f"unknown category: {', '.join(unknown)}")
        # Gibberish is never redacted, so its ensemble is never run here.
        detectors = [
            d
            for d in self._detectors
            if d.category in allowed and d.category != "gibberish"
        ]
        if not detectors:
            raise ValueError(
                "nothing to redact: no rule category (secrets, pii, "
                "profanity) is both selected by this Scanner and allowed "
                "by categories"
            )
        self._check(text)
        chosen = [
            f
            for f in self._collect(text, detectors)
            if f.confidence >= self.min_confidence
        ]
        return render(text, chosen, mode, placeholder, mask_char)


def _drop_url_emails(ordered: Tuple[Finding, ...]) -> Tuple[Finding, ...]:
    """Drop email findings overlapping a url_credentials span: in
    "https://user:pass@host.tld" the "pass@host.tld" part is a password
    and a host, not an address. ordered is sorted by start; one sweep."""
    urls: List[List[int]] = []
    for f in ordered:
        if f.kind != "url_credentials":
            continue
        if urls and f.start <= urls[-1][1]:
            urls[-1][1] = max(urls[-1][1], f.end)
        else:
            urls.append([f.start, f.end])
    if not urls:
        return ordered
    kept: List[Finding] = []
    index = 0
    for f in ordered:
        if f.kind == "email":
            while index < len(urls) and urls[index][1] <= f.start:
                index += 1
            if index < len(urls) and urls[index][0] < f.end:
                continue
        kept.append(f)
    return tuple(kept)
