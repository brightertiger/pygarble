"""detect-secrets plugins without verification or global settings changes."""

from typing import Any, Iterable, Iterator, Optional, Tuple

from ...findings import Finding, sort_key
from .._engine import _names
from ..base import BackendError, check_text, optional_module

# IPs are PII, entropy is opt-in, and PrivateKeyDetector returns only a
# header. The native secrets detector already redacts complete PEM blocks.
_EXCLUDED = frozenset(
    {
        "IPPublicDetector",
        "PrivateKeyDetector",
        "Base64HighEntropyString",
        "HexHighEntropyString",
    }
)


class DetectSecretsDetector:
    category = "secrets"
    kinds = frozenset({"detect_secrets_secret"})

    def __init__(self, plugins: Optional[Iterable[str]] = None) -> None:
        module = optional_module("detect_secrets.core.plugins.util", "secrets")
        classes = {
            cls.__name__: cls
            for cls in module.get_mapping_from_secret_type_to_class().values()
        }
        chosen = (
            tuple(sorted(set(classes) - _EXCLUDED))
            if plugins is None
            else tuple(dict.fromkeys(_names("plugins", plugins)))
        )
        if not chosen or set(chosen) - set(classes):
            raise ValueError(
                "unknown or empty detect-secrets plugin selection"
            )
        if "PrivateKeyDetector" in chosen:
            raise ValueError(
                "use the native detector for complete private keys"
            )
        self._plugins = tuple(classes[name]() for name in chosen)

    def detect(self, text: str) -> Tuple[Finding, ...]:
        check_text(text)
        found = set()
        offset = 0
        try:
            for line in text.splitlines(keepends=True):
                for plugin in self._plugins:
                    name = type(plugin).__name__
                    # analyze_line may verify credentials over the network.
                    # analyze_string only performs local matching. Entropy
                    # plugins defer their threshold check to analyze_line,
                    # so explicitly preserve that check here.
                    entropy = getattr(plugin, "entropy_limit", None)
                    for start, end in _spans(plugin, line):
                        found.add(
                            Finding(
                                self.category,
                                "detect_secrets_secret",
                                offset + start,
                                offset + end,
                                (
                                    0.6
                                    if entropy is not None
                                    or name == "KeywordDetector"
                                    else 0.9
                                ),
                                "detect_secrets:" + name,
                            )
                        )
                offset += len(line)
        except Exception:
            raise BackendError("detect-secrets scan failed") from None
        return tuple(sorted(found, key=sort_key))


def _spans(plugin: Any, line: str) -> Iterator[Tuple[int, int]]:
    values = set(plugin.analyze_string(line)) - {""}
    entropy = getattr(plugin, "entropy_limit", None)
    if entropy is not None:
        values = {
            value
            for value in values
            if plugin.calculate_shannon_entropy(value) > entropy
        }
    if not values:
        return
    patterns = getattr(plugin, "denylist", None)
    if patterns is None and entropy is not None:
        patterns = (plugin.regex,)
    if patterns is not None:
        # Recover offsets in one pass per pattern. Searching the whole line
        # once per distinct secret would be quadratic on dense documents.
        located = set()
        for pattern in patterns:
            for match in pattern.finditer(line):
                groups = (
                    range(1, pattern.groups + 1) if pattern.groups else (0,)
                )
                for group in groups:
                    value = match.group(group)
                    if value in values:
                        located.add(value)
                        yield match.span(group)
        if values - located:
            raise BackendError("backend returned a changed value")
        return
    # KeywordDetector yields at most one value per fixed upstream pattern.
    # Conservatively redact every repetition of those values on the line.
    for value in values:
        start = line.find(value)
        if start < 0:
            raise BackendError("backend returned a changed value")
        while start >= 0:
            yield start, start + len(value)
            start = line.find(value, start + len(value))
