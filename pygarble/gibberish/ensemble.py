"""Deterministic aggregation with explicit English profiles."""

import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from ..validation import finite_number, process_input
from .analysis import Analysis, Signal
from .detector import GarbleDetector
from .options import accepted_options, unknown_options, warn_unknown_options
from .preprocessing import TextFeatures
from .registry import STRATEGY_MAP, Strategy

LEGACY_STRATEGIES = (
    Strategy.MARKOV_CHAIN,
    Strategy.LOG_LIKELIHOOD_RATIO,
    Strategy.WORD_ANOMALY,
)
PROFILES = {
    "english": LEGACY_STRATEGIES
    + (
        Strategy.MOJIBAKE,
        Strategy.KEYBOARD_ADJACENCY,
        Strategy.CONTROL_CHARACTERS,
    ),
    "english_extended": LEGACY_STRATEGIES
    + (
        Strategy.MOJIBAKE,
        Strategy.KEYBOARD_ADJACENCY,
        Strategy.CONTROL_CHARACTERS,
        Strategy.PATTERN_MATCHING,
        Strategy.LOCAL_ANOMALY,
        Strategy.REPETITION,
    ),
    "legacy": LEGACY_STRATEGIES,
    "corruption": (Strategy.MOJIBAKE, Strategy.CONTROL_CHARACTERS),
    "spoofing": (Strategy.UNICODE_SCRIPT,),
    # Degenerate model output: loops, encoding damage, dense token salad.
    # Deliberately excludes the Markov/word-anomaly members so technical
    # prose, code and product names stay quiet.
    "llm_output": (
        Strategy.REPETITION,
        Strategy.CONTROL_CHARACTERS,
        Strategy.MOJIBAKE,
        Strategy.LOCAL_ANOMALY,
    ),
}


class EnsembleDetector:
    def __init__(
        self,
        strategies: Optional[List[Union[Strategy, str]]] = None,
        threshold: float = 0.5,
        voting: Optional[str] = None,
        weights: Optional[List[float]] = None,
        threads: Optional[int] = None,
        *,
        profile: Optional[str] = None,
        strategy_kwargs: Optional[Mapping[Strategy, Mapping[str, Any]]] = None,
        allowlist: Optional[Iterable[str]] = None,
        max_input_length: Optional[int] = None,
        timeout_per_text: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if profile is not None and strategies is not None:
            raise ValueError("use either profile or strategies")
        self.profile = profile or (
            "english" if strategies is None else "custom"
        )
        if strategies is None:
            if self.profile not in PROFILES:
                raise ValueError(f"unknown profile: {self.profile}")
            strategies = list(PROFILES[self.profile])
        if not strategies:
            raise ValueError("strategies must contain at least one strategy")
        members: List[Strategy] = [
            Strategy(s) if isinstance(s, str) else s for s in strategies
        ]
        self.voting = (
            voting
            if voting is not None
            else ("majority" if self.profile == "custom" else "any")
        )
        if self.voting not in (
            "majority",
            "any",
            "all",
            "average",
            "weighted",
        ):
            raise ValueError(
                "voting must be majority, any, all, average, or weighted"
            )
        if self.voting == "weighted" and weights is None:
            raise ValueError("weights required when voting='weighted'")
        if self.voting != "weighted" and weights is not None:
            warnings.warn(
                "weights are ignored unless voting='weighted'; this will "
                "become an error in a future release",
                FutureWarning,
                stacklevel=2,
            )
        self.strategies = list(members)
        if weights is not None and len(weights) != len(members):
            raise ValueError("weights must have same length as strategies")
        self.weights = [
            finite_number("weights", weight)
            for weight in (
                weights if weights is not None else [1.0] * len(members)
            )
        ]
        if any(weight < 0 for weight in self.weights):
            raise ValueError("weights must be non-negative")
        if not any(self.weights):
            raise ValueError("weights must not all be zero")
        options: Dict[Strategy, Dict[str, Any]] = {
            (Strategy(key) if isinstance(key, str) else key): dict(value)
            for key, value in (strategy_kwargs or {}).items()
        }
        if any(strategy not in members for strategy in options):
            raise ValueError(
                "strategy_kwargs contains a strategy not selected"
            )
        class_names = {
            strategy: STRATEGY_MAP[strategy].__name__ for strategy in members
        }
        accepted_by_any = set()
        for name in class_names.values():
            accepted = accepted_options(name)
            accepted_by_any |= (
                set(kwargs) if accepted is None else set(accepted)
            )
        warn_unknown_options(
            "EnsembleDetector (no selected strategy accepts them)",
            sorted(set(kwargs) - accepted_by_any),
        )
        for strategy, member_options in options.items():
            warn_unknown_options(
                class_names[strategy],
                unknown_options(class_names[strategy], member_options),
            )
        words = (
            list(allowlist)
            if allowlist is not None and not isinstance(allowlist, str)
            else allowlist
        )
        self._detectors = []
        for strategy in members:
            accepted = accepted_options(class_names[strategy])
            shared = {
                key: value
                for key, value in kwargs.items()
                if accepted is None or key in accepted
            }
            member = dict(shared, **options.get(strategy, {}))
            member = {
                key: value
                for key, value in member.items()
                if accepted is None or key in accepted
            }
            self._detectors.append(
                GarbleDetector(
                    strategy,
                    threshold,
                    threads,
                    allowlist=words,
                    max_input_length=max_input_length,
                    timeout_per_text=timeout_per_text,
                    strategy_kwargs=member,
                )
            )
        first = self._detectors[0]
        self.threshold = first.threshold
        self.threads = first.threads
        self.max_input_length = first.max_input_length
        self.timeout_per_text = first.timeout_per_text
        self.allowlist = first.allowlist
        self.kwargs = dict(kwargs)
        self.strategy_kwargs = options

    def _aggregate(
        self, pairs: List[Tuple[Signal, float]]
    ) -> Tuple[bool, float]:
        if not pairs:
            return False, 0.0
        scores = [signal.score for signal, _ in pairs]
        if self.voting == "weighted":
            # Scaling by the largest weight keeps huge finite weights from
            # overflowing the sum; pairs only hold positive weights here.
            scale = max(weight for _, weight in pairs)
            total = sum(weight / scale for _, weight in pairs)
            score = (
                sum(signal.score * weight / scale for signal, weight in pairs)
                / total
            )
        elif self.voting == "any":
            score = max(scores)
        elif self.voting == "all":
            score = min(scores)
        else:
            score = sum(scores) / len(scores)
        if self.voting == "majority":
            return (
                sum(value >= self.threshold for value in scores)
                > len(scores) / 2,
                score,
            )
        return score >= self.threshold, score

    def _analyze_single(self, text: str) -> Analysis:
        features = TextFeatures(text, self.allowlist)
        signals = tuple(
            detector._signal(features) for detector in self._detectors
        )
        pairs = [
            (signal, weight)
            for signal, weight in zip(signals, self.weights)
            if signal.applicable and (self.voting != "weighted" or weight > 0)
        ]
        decision, score = self._aggregate(pairs)
        return Analysis(
            decision,
            score,
            (
                "garbled"
                if decision
                else ("clean" if pairs else "insufficient_evidence")
            ),
            signals,
            self.profile,
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
        if self.voting not in ("any", "all"):
            return self._analyze_single(text).garbled
        features = TextFeatures(text, self.allowlist)
        applicable = False
        for detector in self._detectors:
            evidence = detector._strategy_instance.evaluate(features)
            if not evidence.applicable:
                continue
            applicable = True
            vote = evidence.score >= self.threshold
            if self.voting == "any" and vote:
                return True
            if self.voting == "all" and not vote:
                return False
        return applicable and self.voting == "all"

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
        """Heuristic aggregate.

        Under voting='majority' the decision counts member votes, so
        Analysis.garbled can be True while Analysis.score is below threshold.
        """
        return process_input(
            X,
            lambda text: self._analyze_single(text).score,
            self.threads,
            self.timeout_per_text,
            self.max_input_length,
        )

    score = predict_proba
