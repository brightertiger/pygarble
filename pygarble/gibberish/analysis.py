"""Immutable, serializable explanations. Scores are not probabilities."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    reason: str


@dataclass(frozen=True)
class Evidence:
    score: float
    applicable: bool = True
    reason: str = "heuristic_score"
    spans: Tuple[Span, ...] = ()


@dataclass(frozen=True)
class Signal:
    strategy: str
    score: float
    applicable: bool
    reason: str
    spans: Tuple[Span, ...] = ()


@dataclass(frozen=True)
class Analysis:
    garbled: bool
    score: float
    status: str
    signals: Tuple[Signal, ...]
    profile: str
    model_version: str = "english-v2"

    @property
    def spans(self) -> Tuple[Span, ...]:
        return tuple(
            sorted(
                {span for signal in self.signals for span in signal.spans},
                key=lambda span: (span.start, span.end, span.reason),
            )
        )
