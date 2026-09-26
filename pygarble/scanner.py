"""One call that screens text for secrets, PII, profanity and gibberish."""

from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from .findings import CATEGORIES, Finding, Redaction, ScanReport, sort_key
from .redaction import render
from .validation import positive_int, unit_interval

DEFAULT_CATEGORIES = CATEGORIES
GIBBERISH_KINDS = frozenset({"garbled"})

Detector = Any  # object with .detect(text) -> Tuple[Finding, ...]


def _pii_detector(
    locales: Tuple[str, ...],
    kinds: Optional[FrozenSet[str]],
    exclude: FrozenSet[str],
) -> Detector:
    from .pii import PIIDetector

    return PIIDetector(kinds=kinds, exclude_kinds=exclude, locales=locales)


def _profanity_detector(allowlist: Optional[Iterable[str]]) -> Detector:
    from .profanity import ProfanityDetector

    return ProfanityDetector(allowlist=allowlist)


def _all_kinds() -> Dict[str, FrozenSet[str]]:
    from .pii import ALL_KINDS as PII_KINDS
    from .profanity import ALL_KINDS as PROFANITY_KINDS
    from .secrets.patterns import ALL_KINDS as SECRET_KINDS

    return {
        "secrets": frozenset(SECRET_KINDS),
        "pii": frozenset(PII_KINDS),
        "profanity": frozenset(PROFANITY_KINDS),
        "gibberish": GIBBERISH_KINDS,
    }


def _names(label: str, value: Iterable[str]) -> Iterable[str]:
    if isinstance(value, str):
        raise ValueError(f"{label} must be an iterable of names, not a string")
    return value


class _Gibberish:
    category = "gibberish"

    def __init__(
        self,
        profile: str,
        threshold: float,
        allowlist: Optional[Iterable[str]],
        max_input_length: Optional[int],
    ) -> None:
        from .ensemble import EnsembleDetector

        self.detector = EnsembleDetector(
            profile=profile,
            threshold=threshold,
            allowlist=allowlist,
            max_input_length=max_input_length,
        )

    def detect(self, text: str) -> Tuple[Finding, ...]:
        from .analysis import Analysis

        analysis = cast(Analysis, self.detector.analyze(text))
        if not analysis.garbled:
            return ()
        return (
            Finding(
                "gibberish",
                "garbled",
                0,
                len(text),
                min(1.0, max(0.0, float(analysis.score))),
                analysis.status,
            ),
        )


class Scanner:
    """Screen text for secrets, PII, profanity and gibberish.

    min_confidence gates the rule categories; gibberish is gated by
    threshold.
    """

    def __init__(
        self,
        categories: Iterable[str] = DEFAULT_CATEGORIES,
        *,
        min_confidence: float = 0.5,
        kinds: Optional[Iterable[str]] = None,
        exclude_kinds: Iterable[str] = (),
        locales: Iterable[str] = ("us", "uk", "in"),
        profile: str = "english",
        threshold: float = 0.5,
        allowlist: Optional[Iterable[str]] = None,
        profanity_allowlist: Optional[Iterable[str]] = None,
        secrets_without_context: bool = False,
        max_input_length: Optional[int] = None,
    ) -> None:
        chosen = tuple(dict.fromkeys(_names("categories", categories)))
        unknown = [c for c in chosen if c not in CATEGORIES]
        if unknown:
            raise ValueError(
                f"unknown category: {', '.join(unknown)}; "
                f"valid: {', '.join(CATEGORIES)}"
            )
        if not chosen:
            raise ValueError("at least one category is required")
        self.categories = chosen
        self.min_confidence = unit_interval("min_confidence", min_confidence)
        # Validated even when gibberish or profanity is not selected, so a
        # typo fails the same way whatever the categories.
        threshold = unit_interval("threshold", threshold)
        from .ensemble import PROFILES

        if profile not in PROFILES:
            raise ValueError(
                f"unknown profile: {profile}; valid: {', '.join(PROFILES)}"
            )
        if profanity_allowlist is not None:
            profanity_allowlist = tuple(
                _names("profanity_allowlist", profanity_allowlist)
            )
        self.max_input_length = (
            None
            if max_input_length is None
            else positive_int("max_input_length", max_input_length)
        )
        from .pii import LOCALES

        chosen_locales = tuple(dict.fromkeys(_names("locales", locales)))
        bad_locales = [loc for loc in chosen_locales if loc not in LOCALES]
        if bad_locales:
            raise ValueError(
                f"unknown locale: {', '.join(bad_locales)}; "
                f"valid: {', '.join(LOCALES)}"
            )
        known = _all_kinds()
        universe = frozenset().union(*known.values())
        wanted = None if kinds is None else frozenset(_names("kinds", kinds))
        excluded = frozenset(_names("exclude_kinds", exclude_kinds))
        bad = sorted(((wanted or frozenset()) | excluded) - universe)
        if bad:
            raise ValueError(
                f"unknown kind: {', '.join(bad)}; "
                f"valid: {', '.join(sorted(universe))}"
            )
        self._kinds = wanted
        self._exclude = excluded
        self._detectors: List[Detector] = []
        from .pii import locale_kinds

        for category in chosen:
            allowed = known[category]
            selected = (
                allowed if wanted is None else (allowed & wanted)
            ) - excluded
            if category == "secrets":
                live = selected
                if not secrets_without_context:
                    live = live - {"high_entropy_string"}
                if live:
                    self._detectors.append(
                        _secrets(selected, secrets_without_context)
                    )
            elif category == "pii":
                if selected & locale_kinds(chosen_locales):
                    self._detectors.append(
                        _pii_detector(
                            chosen_locales,
                            None if wanted is None else selected,
                            excluded & allowed,
                        )
                    )
            elif category == "profanity":
                if "profanity" in selected:
                    self._detectors.append(
                        _profanity_detector(profanity_allowlist)
                    )
            elif "garbled" in selected:
                self._detectors.append(
                    _Gibberish(profile, threshold, allowlist, max_input_length)
                )
        if not self._detectors:
            raise ValueError(
                "nothing to scan for: no kind of the selected categories "
                "is left after kinds, exclude_kinds, locales and "
                "secrets_without_context"
            )

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


def _secrets(selected: FrozenSet[str], without_context: bool) -> Detector:
    from .secrets import SecretsDetector

    return SecretsDetector(
        kinds=selected, exclude_kinds=(), without_context=without_context
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(v) for v in value)
    return value


_CACHE: Dict[Tuple[Tuple[str, Any], ...], Scanner] = {}
_CACHE_LIMIT = 32


def _cached(kwargs: Dict[str, Any]) -> Scanner:
    key = tuple(sorted((k, _freeze(v)) for k, v in kwargs.items()))
    scanner = _CACHE.get(key)
    if scanner is None:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()
        scanner = Scanner(**kwargs)
        _CACHE[key] = scanner
    return scanner


def scan(text: str, **kwargs: Any) -> ScanReport:
    return _cached(kwargs).scan(text)


def redact(
    text: str,
    *,
    mode: str = "placeholder",
    placeholder: str = "[{KIND}]",
    mask_char: str = "*",
    categories: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> Redaction:
    return _cached(kwargs).redact(
        text,
        mode=mode,
        placeholder=placeholder,
        mask_char=mask_char,
        categories=categories,
    )
