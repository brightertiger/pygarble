"""Compose built-in rules and explicitly selected local backends."""

from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from ..findings import Finding, Redaction, ScanReport, sort_key
from ..validation import positive_int, unit_interval
from ._engine import ScanEngine, _drop_url_emails, _names
from .base import BackendError, Detector, ScreeningDetector
from .pii import ALL_KINDS as PII_KINDS
from .pii import PIIDetector, locale_kinds
from .profanity import ALL_KINDS as PROFANITY_KINDS
from .profanity import ProfanityDetector
from .secrets import ALL_KINDS as SECRET_KINDS
from .secrets import SecretsDetector

CATEGORIES = ("secrets", "pii", "profanity")
_BUILTIN_KINDS = {
    "secrets": SECRET_KINDS,
    "pii": PII_KINDS,
    "profanity": PROFANITY_KINDS,
}


class _Selected:
    def __init__(
        self, detector: Detector, kinds: FrozenSet[str], external: bool
    ) -> None:
        self.category = detector.category
        self.detector = detector
        self.kinds = kinds
        self.declared_kinds = frozenset(getattr(detector, "kinds", kinds))
        self.external = external

    def detect(self, text: str) -> Tuple[Finding, ...]:
        try:
            findings = self.detector.detect(text)
            kept = []
            for finding in findings:
                if self.external:
                    if (
                        not isinstance(finding, Finding)
                        or finding.category != self.category
                        or finding.kind not in self.declared_kinds
                        or type(finding.start) is not int
                        or type(finding.end) is not int
                        or not 0 <= finding.start < finding.end <= len(text)
                        or not isinstance(finding.reason, str)
                    ):
                        raise BackendError("invalid backend finding")
                    unit_interval("confidence", finding.confidence)
                if finding.kind in self.kinds:
                    kept.append(finding)
            return tuple(kept)
        except Exception:
            if not self.external:
                raise
            # External exceptions may contain the scanned value. Never
            # expose them or silently convert an incomplete scan to clean.
            raise BackendError("screening backend failed") from None


class Scanner(ScanEngine):
    """Screen secrets, PII and profanity; no gibberish model is loaded.

    Backends supplement built-in rules. Set builtin=False to use only
    backends/custom detectors. Construct once and reuse for throughput.
    """

    def __init__(
        self,
        categories: Iterable[str] = CATEGORIES,
        *,
        backends: Iterable[str] = (),
        backend_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
        detectors: Iterable[ScreeningDetector] = (),
        builtin: bool = True,
        min_confidence: float = 0.5,
        kinds: Optional[Iterable[str]] = None,
        exclude_kinds: Iterable[str] = (),
        locales: Iterable[str] = ("us", "uk", "in"),
        profanity_allowlist: Optional[Iterable[str]] = None,
        profanity_tiers: Iterable[str] = ("strong", "mild"),
        secrets_without_context: bool = False,
        max_input_length: Optional[int] = None,
    ) -> None:
        from .backends import BACKENDS

        self.categories = tuple(
            dict.fromkeys(_names("categories", categories))
        )
        if not self.categories or set(self.categories) - set(CATEGORIES):
            raise ValueError(
                "categories must be a non-empty subset of "
                + ", ".join(CATEGORIES)
            )
        self.min_confidence = unit_interval("min_confidence", min_confidence)
        self.max_input_length = (
            None
            if max_input_length is None
            else positive_int("max_input_length", max_input_length)
        )
        selected_backends = tuple(dict.fromkeys(_names("backends", backends)))
        if set(selected_backends) - set(BACKENDS):
            raise ValueError("unknown backend; valid: " + ", ".join(BACKENDS))
        options = dict(backend_options or {})
        if set(options) - set(selected_backends):
            raise ValueError("backend_options must name selected backends")
        custom = tuple(detectors)
        for detector in custom:
            if detector.category not in CATEGORIES or not detector.kinds:
                raise ValueError("custom detector needs a category and kinds")
            if isinstance(detector.kinds, str) or any(
                not isinstance(kind, str) or not kind
                for kind in detector.kinds
            ):
                raise ValueError("custom detector kinds must be names")
        known = frozenset().union(*_BUILTIN_KINDS.values())
        for name in selected_backends:
            known |= BACKENDS[name].kinds
        for detector in custom:
            known |= frozenset(detector.kinds)
        wanted = known if kinds is None else frozenset(_names("kinds", kinds))
        excluded = frozenset(_names("exclude_kinds", exclude_kinds))
        if (wanted | excluded) - known:
            raise ValueError("unknown kind for the selected backends")
        selected = wanted - excluded
        self._detectors = []
        # Validate native configuration consistently, even when unselected.
        pii = PIIDetector(locales=locales)
        profanity = ProfanityDetector(
            allowlist=profanity_allowlist, tiers=profanity_tiers
        )
        if builtin:
            native: List[Tuple[Detector, FrozenSet[str]]] = [
                (
                    SecretsDetector(
                        kinds=selected & SECRET_KINDS,
                        without_context=secrets_without_context,
                    ),
                    (
                        SECRET_KINDS
                        if secrets_without_context
                        else SECRET_KINDS - {"high_entropy_string"}
                    ),
                ),
                (pii, locale_kinds(pii.locales)),
                (profanity, PROFANITY_KINDS),
            ]
            for native_detector, supported in native:
                if native_detector.category in self.categories:
                    live = selected & supported
                    if live:
                        if isinstance(native_detector, PIIDetector):
                            native_detector.kinds = live
                        self._detectors.append(
                            _Selected(native_detector, live, False)
                        )
        for name in selected_backends:
            factory = BACKENDS[name]
            if factory.category not in self.categories:
                raise ValueError("backend category is not selected: " + name)
            if selected & factory.kinds:
                backend = factory(**dict(options.get(name, {})))
                live = selected & backend.kinds
                if live:
                    self._detectors.append(_Selected(backend, live, True))
        for detector in custom:
            if detector.category not in self.categories:
                raise ValueError("custom detector category is not selected")
            live = selected & detector.kinds
            if live:
                self._detectors.append(_Selected(detector, live, True))
        if not self._detectors:
            raise ValueError("nothing to scan for")

    def _collect(
        self, text: str, detectors: Sequence[Detector]
    ) -> Tuple[Finding, ...]:
        best: Dict[Tuple[str, str, int, int], Finding] = {}
        for detector in detectors:
            for finding in detector.detect(text):
                key = (
                    finding.category,
                    finding.kind,
                    finding.start,
                    finding.end,
                )
                current = best.get(key)
                if current is None or (finding.confidence, finding.reason) > (
                    current.confidence,
                    current.reason,
                ):
                    best[key] = finding
        return _drop_url_emails(tuple(sorted(best.values(), key=sort_key)))


def scan(text: str, **kwargs: Any) -> ScanReport:
    return Scanner(**kwargs).scan(text)


def redact(
    text: str,
    *,
    mode: str = "placeholder",
    placeholder: str = "[{KIND}]",
    mask_char: str = "*",
    **kwargs: Any,
) -> Redaction:
    return Scanner(**kwargs).redact(
        text, mode=mode, placeholder=placeholder, mask_char=mask_char
    )
