"""Gibberish detection and calibration, with lazy strategy loading."""

from importlib import import_module
from typing import TYPE_CHECKING, Any, List

from .analysis import Analysis, Signal, Span

if TYPE_CHECKING:
    from .calibration import CalibrationReport as CalibrationReport
    from .calibration import ThresholdPoint as ThresholdPoint
    from .calibration import calibrate as calibrate
    from .core import STRATEGY_MAP as STRATEGY_MAP
    from .core import EnsembleDetector as EnsembleDetector
    from .core import GarbleDetector as GarbleDetector
    from .core import Strategy as Strategy

_LAZY = {
    "GarbleDetector": "core",
    "EnsembleDetector": "core",
    "Strategy": "core",
    "STRATEGY_MAP": "core",
    "CalibrationReport": "calibration",
    "ThresholdPoint": "calibration",
    "calibrate": "calibration",
}

__all__ = [
    "Analysis",
    "Signal",
    "Span",
    "GarbleDetector",
    "EnsembleDetector",
    "Strategy",
    "STRATEGY_MAP",
    "CalibrationReport",
    "ThresholdPoint",
    "calibrate",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(name)
    value = getattr(import_module("." + _LAZY[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
