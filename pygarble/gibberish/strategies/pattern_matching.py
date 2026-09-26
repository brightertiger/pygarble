import re
from typing import Any, Dict, Pattern

from .base import BaseStrategy


class PatternMatchingStrategy(BaseStrategy):
    DEFAULT_PATTERNS: Dict[str, str] = {
        "special_chars": r"[^a-zA-Z0-9\s]{3,}",
        "repeated_chars": r"([a-zA-Z])\1{3,}",
        "uppercase_sequence": r"[A-Z]{5,}",
        "long_numbers": r"[0-9]{8,}",
        "keyboard_row_qwerty": (
            "(?i)(qwert|werty|ertyu|rtyui|tyuio|yuiop|asdfg|sdfgh|df"
            "ghj|fghjk|ghjkl|zxcvb|xcvbn|cvbnm)"
        ),
        "keyboard_row_reverse": (
            "(?i)(poiuy|oiuyt|iuytr|uytre|ytrew|trewq|lkjhg|kjhgf|jh"
            "gfd|hgfds|gfdsa|mnbvc|nbvcx|bvcxz)"
        ),
        "consonant_cluster": r"[bcdfghjklmnpqrstvwxz]{5,}",
        "alternating_pattern": r"(?i)([a-z])(?!\1)([a-z])(\1\2){2,}",
    }

    # Weak patterns match legitimate text too often (ALL-CAPS headlines,
    # order numbers, "----" rulers, "://" in URLs) to be decisive alone;
    # they only corroborate a strong match. consonant_cluster only sees
    # novel words, which drops dictionary words and short ALL-CAPS
    # acronyms (HTTPS, JSON).
    # repeated_chars is letters-only; digit runs ("10000") are covered by
    # long_numbers (weak) only.
    WEAK_PATTERNS = {"special_chars", "uppercase_sequence", "long_numbers"}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._compiled_patterns: Dict[str, Pattern] = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, Pattern]:
        custom_patterns = self.kwargs.get("patterns", {})
        override_defaults = self.kwargs.get("override_defaults", False)

        if override_defaults:
            patterns = custom_patterns
        else:
            patterns = {**self.DEFAULT_PATTERNS, **custom_patterns}

        return {name: re.compile(regex) for name, regex in patterns.items()}

    # consonant_cluster only judges words the dictionary cannot vouch for;
    # "strengths", "catchphrase" and "worthwhile" are real English.
    NOVEL_ONLY_PATTERNS = {"consonant_cluster"}

    def _predict_proba_impl(self, text: str) -> float:
        if not self._compiled_patterns:
            return 0.0

        # URL/email tokens legitimately contain consonant runs ("https")
        # and symbol clusters ("://")
        text = " ".join(
            t
            for t in text.split()
            if "://" not in t
            and "@" not in t
            and not t.lower().startswith("www.")
        )
        novel_text = None

        strong = weak = 0
        for name, pattern in self._compiled_patterns.items():
            target = text
            if name in self.NOVEL_ONLY_PATTERNS:
                if novel_text is None:
                    novel_text = " ".join(self._novel_words(text))
                target = novel_text
            if pattern.search(target):
                if name in self.WEAK_PATTERNS:
                    weak += 1
                else:
                    strong += 1

        if strong == 0 and weak == 0:
            return 0.0
        if strong == 0:
            # Weak evidence alone never crosses the default threshold
            return min(0.3 + 0.08 * (weak - 1), 0.45)
        return min(0.65 + 0.08 * (strong - 1 + weak), 1.0)
