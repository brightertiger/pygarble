"""One call that screens text for secrets, PII, profanity and gibberish."""

from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple, cast

from .findings import CATEGORIES, Finding, Redaction, ScanReport
from .screening._engine import ScanEngine, _names
from .screening.base import Detector
from .validation import positive_int, unit_interval

DEFAULT_CATEGORIES = CATEGORIES
GIBBERISH_KINDS = frozenset({"garbled"})


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


class Scanner(ScanEngine):
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
