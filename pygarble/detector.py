"""Single-strategy public API and shared input handling."""

from enum import Enum
from typing import Any, FrozenSet, Iterable, List, Mapping, Optional, Union

from .analysis import Analysis, Signal
from .preprocessing import TextFeatures, fold_diacritics
from .registry import STRATEGY_MAP, Strategy
from .strategies.base import BaseStrategy
from .validation import (
    finite_number,
    positive_int,
    process_input,
    unit_interval,
    validate_batch,
)


class GarbleDetector:
    def __init__(
        self,
        strategy: Union[Strategy, str],
        threshold: float = 0.5,
        threads: Optional[int] = None,
        *,
        allowlist: Optional[Iterable[str]] = None,
        max_input_length: Optional[int] = None,
        timeout_per_text: Optional[float] = None,
        strategy_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Single-strategy detector.

        timeout_per_text only bounds work submitted to the thread pool,
        which is used for batches of 10 or more when ``threads`` is 2 or
        more. Single strings and small batches run inline and cannot be
        interrupted.
        """
        self.threshold = unit_interval("threshold", threshold)
        self.threads = (
            positive_int("threads", threads) if threads is not None else None
        )
        if self.threads is not None and self.threads > 2**31 - 1:
            raise ValueError("threads is too large")
        self.max_input_length = (
            positive_int("max_input_length", max_input_length)
            if max_input_length is not None
            else None
        )
        self.timeout_per_text = timeout_per_text
        if timeout_per_text is not None:
            if finite_number("timeout_per_text", timeout_per_text) <= 0:
                raise ValueError("timeout_per_text must be positive")
        if isinstance(strategy, str):
            strategy = Strategy(strategy)
        if not isinstance(strategy, Strategy):
            if isinstance(strategy, Enum):
                raise NotImplementedError(
                    f"Strategy {strategy} is not implemented"
                )
            raise TypeError("strategy must be a Strategy enum member or name")
        if isinstance(allowlist, str):
            raise TypeError("allowlist must be an iterable of words")
        words = list(allowlist) if allowlist is not None else []
        validate_batch(words)
        self.allowlist: FrozenSet[str] = frozenset(
            fold_diacritics(word).lower() for word in words
        )
        self.strategy = strategy
        self.kwargs = dict(kwargs, **dict(strategy_kwargs or {}))
        self._strategy_instance = self._create_strategy_instance()

    def _create_strategy_instance(self) -> BaseStrategy:
        return STRATEGY_MAP[self.strategy](**self.kwargs)

    def applicable(self, text: str) -> bool:
        BaseStrategy._validate_input(text)
        return self._strategy_instance.evaluate(
            TextFeatures(text, self.allowlist)
        ).applicable

    def _signal(self, features: TextFeatures) -> Signal:
        evidence = self._strategy_instance.evaluate(features)
        return Signal(
            self.strategy.value,
            evidence.score,
            evidence.applicable,
            evidence.reason,
            evidence.spans,
        )

    def _analyze_single(self, text: str) -> Analysis:
        signal = self._signal(TextFeatures(text, self.allowlist))
        decision = signal.applicable and signal.score >= self.threshold
        return Analysis(
            decision,
            signal.score,
            (
                "garbled"
                if decision
                else (
                    "clean" if signal.applicable else "insufficient_evidence"
                )
            ),
            (signal,),
            self.strategy.value,
        )

    def analyze(
        self, X: Union[str, List[str]]
    ) -> Union[Analysis, List[Analysis]]:
        return process_input(
            X,
            self._analyze_single,
            self.threads,
            self.timeout_per_text,
            self.max_input_length,
        )

    def _predict_single(self, text: str) -> bool:
        evidence = self._strategy_instance.evaluate(
            TextFeatures(text, self.allowlist)
        )
        return evidence.applicable and evidence.score >= self.threshold

    def _score_single(self, text: str) -> float:
        return self._strategy_instance.evaluate(
            TextFeatures(text, self.allowlist)
        ).score

    def predict(self, X: Union[str, List[str]]) -> Union[bool, List[bool]]:
        return process_input(
            X,
            self._predict_single,
            self.threads,
            self.timeout_per_text,
            self.max_input_length,
        )

    def predict_proba(
        self, X: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        """Return a heuristic score, not a calibrated probability."""
        return process_input(
            X,
            self._score_single,
            self.threads,
            self.timeout_per_text,
            self.max_input_length,
        )

    score = predict_proba
