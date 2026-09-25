__version__ = "0.9.0"
__author__ = "Ujjwal Singh Rao"
__email__ = "ujjwalsrao@gmail.com"

from .analysis import Analysis, Signal, Span
from .calibration import CalibrationReport, ThresholdPoint, calibrate
from .core import EnsembleDetector, GarbleDetector, Strategy

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
]
