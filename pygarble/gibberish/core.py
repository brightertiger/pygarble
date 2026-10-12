"""Compatibility facade for the public detection API."""

from typing import TYPE_CHECKING, Any, List

from .._compat import forward_removed
from .detector import GarbleDetector
from .ensemble import EnsembleDetector
from .registry import Strategy

if TYPE_CHECKING:
    from .registry import STRATEGY_MAP as STRATEGY_MAP

__all__ = ["GarbleDetector", "EnsembleDetector", "Strategy", "STRATEGY_MAP"]

# Strategy classes that pygarble.core exported in 0.8.0.
_MOVED = dict.fromkeys(
    (
        "BaseStrategy",
        "AffixDetectionStrategy",
        "BigramProbabilityStrategy",
        "ConsonantSequenceStrategy",
        "EntropyBasedStrategy",
        "FunctionWordDensityStrategy",
        "HexStringStrategy",
        "KeyboardAdjacencyStrategy",
        "KeyboardPatternStrategy",
        "LetterFrequencyStrategy",
        "LetterPositionStrategy",
        "LogLikelihoodRatioStrategy",
        "MarkovChainStrategy",
        "MojibakeStrategy",
        "NGramFrequencyStrategy",
        "PatternMatchingStrategy",
        "PronouncabilityStrategy",
        "RareTrigramStrategy",
        "RepetitionStrategy",
        "SymbolRatioStrategy",
        "UnicodeScriptStrategy",
        "VowelPatternStrategy",
        "VowelRatioStrategy",
        "WordAnomalyStrategy",
        "WordCollocationStrategy",
        "WordLookupStrategy",
        "ZipfConformityStrategy",
    ),
    "pygarble.gibberish.strategies",
)
# Names 0.8.0 leaked through its own imports, kept for compatibility.
_MOVED.update(
    dict.fromkeys(("Callable", "Dict", "Optional", "Type", "Union"), "typing")
)
_MOVED["concurrent"] = "concurrent.futures"


def __getattr__(name: str) -> Any:
    if name == "STRATEGY_MAP":
        # Building the map imports every strategy, so wait until asked.
        from . import registry

        return registry.STRATEGY_MAP
    return forward_removed("pygarble.core", name, _MOVED)


def __dir__() -> List[str]:
    return sorted(set(globals()) | {"STRATEGY_MAP"})
