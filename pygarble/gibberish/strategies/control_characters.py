"""Detect raw control/decoding artifacts without English normalization."""

import unicodedata
from typing import Any, List

from ...validation import positive_int
from ..analysis import Evidence, Span
from ..preprocessing import TextFeatures
from .base import BaseStrategy


def _is_artifact(char: str) -> bool:
    category = unicodedata.category(char)
    return (
        (category == "Cc" and char not in "\t\r\n")
        or category == "Cs"
        or char == "\ufffd"
    )


class ControlCharactersStrategy(BaseStrategy):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_combining_run = positive_int(
            "max_combining_run", kwargs.get("max_combining_run", 8)
        )

    def evaluate(self, features: TextFeatures) -> Evidence:
        # str.strip() removes VT, FF, U+001C-U+001F and NEL, so the base
        # empty-input shortcut would hide text made only of them.
        text = features.text
        self._validate_input(text)
        if text.strip() or not any(_is_artifact(char) for char in text):
            return super().evaluate(features)
        if self._is_extremely_long_string(text):
            return Evidence(1.0, True, "explicit_length_policy")
        return self._evaluate_features(features)

    def _evaluate_features(self, features: TextFeatures) -> Evidence:
        spans: List[Span] = []
        run = 0
        for index, char in enumerate(features.text):
            if _is_artifact(char):
                spans.append(
                    Span(index, index + 1, "control_or_decoding_artifact")
                )
            if unicodedata.combining(char):
                run += 1
            else:
                if run > self.max_combining_run:
                    spans.append(
                        Span(index - run, index, "excessive_combining_marks")
                    )
                run = 0
        if run > self.max_combining_run:
            spans.append(
                Span(
                    len(features.text) - run,
                    len(features.text),
                    "excessive_combining_marks",
                )
            )
        return Evidence(
            0.9 if spans else 0.0, True, "raw_unicode_artifacts", tuple(spans)
        )

    def _predict_proba_impl(self, text: str) -> float:
        return self._evaluate_features(TextFeatures(text)).score
