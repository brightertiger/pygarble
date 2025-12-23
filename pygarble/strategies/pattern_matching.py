import re
from typing import Dict, Pattern

from .base import BaseStrategy


class PatternMatchingStrategy(BaseStrategy):
    DEFAULT_PATTERNS: Dict[str, str] = {
        "special_chars": r"[^a-zA-Z0-9\s]{3,}",
        "repeated_chars": r"(.)\1{3,}",
        "uppercase_sequence": r"[A-Z]{5,}",
        "long_numbers": r"[0-9]{8,}",
        "keyboard_row_qwerty": r"(?i)(qwert|werty|ertyu|rtyui|tyuio|yuiop|asdfg|sdfgh|dfghj|fghjk|ghjkl|zxcvb|xcvbn|cvbnm)",
        "keyboard_row_reverse": r"(?i)(poiuy|oiuyt|iuytr|uytre|ytrew|trewq|lkjhg|kjhgf|jhgfd|hgfds|gfdsa|mnbvc|nbvcx|bvcxz)",
        "consonant_cluster": r"(?i)[bcdfghjklmnpqrstvwxz]{5,}",
        "alternating_pattern": r"(?i)(.)(.)(\1\2){2,}",
    }

    def __init__(self, **kwargs):
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

    def _predict_impl(self, text: str) -> bool:
        for compiled_pattern in self._compiled_patterns.values():
            if compiled_pattern.search(text):
                return True
        return False

    def _predict_proba_impl(self, text: str) -> float:
        if not self._compiled_patterns:
            return 0.0

        matches = sum(
            1 for p in self._compiled_patterns.values() if p.search(text)
        )
        return min(matches / len(self._compiled_patterns), 1.0)
