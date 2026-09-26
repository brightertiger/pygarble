__version__ = "0.11.0"
__author__ = "Ujjwal Singh Rao"
__email__ = "ujjwalsrao@gmail.com"

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any
from typing import List as _List

from .findings import Finding, Redaction, ScanReport
from .gibberish.analysis import Analysis, Signal, Span

if _TYPE_CHECKING:
    from .gibberish.calibration import CalibrationReport as CalibrationReport
    from .gibberish.calibration import ThresholdPoint as ThresholdPoint
    from .gibberish.calibration import calibrate as calibrate
    from .gibberish.core import EnsembleDetector as EnsembleDetector
    from .gibberish.core import GarbleDetector as GarbleDetector
    from .gibberish.core import Strategy as Strategy
    from .scanner import Scanner as Scanner
    from .scanner import redact as redact
    from .scanner import scan as scan
    from .screening.pii import PIIDetector as PIIDetector
    from .screening.profanity import ProfanityDetector as ProfanityDetector
    from .screening.secrets import SecretsDetector as SecretsDetector

_LAZY = {
    "GarbleDetector": ("gibberish.core", "GarbleDetector"),
    "EnsembleDetector": ("gibberish.core", "EnsembleDetector"),
    "Strategy": ("gibberish.core", "Strategy"),
    "CalibrationReport": ("gibberish.calibration", "CalibrationReport"),
    "ThresholdPoint": ("gibberish.calibration", "ThresholdPoint"),
    "calibrate": ("gibberish.calibration", "calibrate"),
    "Scanner": ("scanner", "Scanner"),
    "scan": ("scanner", "scan"),
    "redact": ("scanner", "redact"),
    "SecretsDetector": ("screening.secrets", "SecretsDetector"),
    "PIIDetector": ("screening.pii", "PIIDetector"),
    "ProfanityDetector": ("screening.profanity", "ProfanityDetector"),
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
