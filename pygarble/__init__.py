__version__ = "0.11.0"
__author__ = "Ujjwal Singh Rao"
__email__ = "ujjwalsrao@gmail.com"

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any
from typing import List as _List

from .analysis import Analysis, Signal, Span
from .findings import Finding, Redaction, ScanReport

if _TYPE_CHECKING:
    from .calibration import CalibrationReport as CalibrationReport
    from .calibration import ThresholdPoint as ThresholdPoint
    from .calibration import calibrate as calibrate
    from .core import EnsembleDetector as EnsembleDetector
    from .core import GarbleDetector as GarbleDetector
    from .core import Strategy as Strategy
    from .pii import PIIDetector as PIIDetector
    from .profanity import ProfanityDetector as ProfanityDetector
    from .scanner import Scanner as Scanner
    from .scanner import redact as redact
    from .scanner import scan as scan
    from .secrets import SecretsDetector as SecretsDetector

_LAZY = {
    "GarbleDetector": ("core", "GarbleDetector"),
    "EnsembleDetector": ("core", "EnsembleDetector"),
    "Strategy": ("core", "Strategy"),
    "CalibrationReport": ("calibration", "CalibrationReport"),
    "ThresholdPoint": ("calibration", "ThresholdPoint"),
    "calibrate": ("calibration", "calibrate"),
    "Scanner": ("scanner", "Scanner"),
    "scan": ("scanner", "scan"),
    "redact": ("scanner", "redact"),
    "SecretsDetector": ("secrets", "SecretsDetector"),
    "PIIDetector": ("pii", "PIIDetector"),
    "ProfanityDetector": ("profanity", "ProfanityDetector"),
}

__all__ = [
    "GarbleDetector",
    "Strategy",
    "EnsembleDetector",
    "__version__",
    "Analysis",
    "Signal",
    "Span",
    "CalibrationReport",
    "ThresholdPoint",
    "calibrate",
    "Finding",
    "ScanReport",
    "Redaction",
    "Scanner",
    "scan",
    "redact",
    "SecretsDetector",
    "PIIDetector",
    "ProfanityDetector",
]


def __getattr__(name: str) -> _Any:
    if name not in _LAZY:
        raise AttributeError(name)
    from importlib import import_module

    module, attr = _LAZY[name]
    try:
        imported = import_module("." + module, __name__)
    except ImportError as exc:
        raise AttributeError(name) from exc
    value = getattr(imported, attr)
    globals()[name] = value
    return value


def __dir__() -> _List[str]:
    # Lazy names are listed before first use, so they tab-complete.
    return sorted(set(globals()) | set(__all__))
