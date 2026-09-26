__version__ = "0.11.0"
__author__ = "Ujjwal Singh Rao"
__email__ = "ujjwalsrao@gmail.com"

from typing import TYPE_CHECKING, Any

from .analysis import Analysis, Signal, Span
from .calibration import CalibrationReport, ThresholdPoint, calibrate
from .core import EnsembleDetector, GarbleDetector, Strategy
from .findings import Finding, Redaction, ScanReport

if TYPE_CHECKING:
    from .pii import PIIDetector as PIIDetector
    from .profanity import ProfanityDetector as ProfanityDetector
    from .scanner import Scanner as Scanner
    from .scanner import redact as redact
    from .scanner import scan as scan
    from .secrets import SecretsDetector as SecretsDetector

_LAZY = {
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


def __getattr__(name: str) -> Any:
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
