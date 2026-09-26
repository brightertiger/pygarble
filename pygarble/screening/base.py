"""Contracts for local screening detectors and optional backends."""

from importlib import import_module
from types import ModuleType
from typing import FrozenSet, Protocol, Tuple

from ..findings import Finding


class Detector(Protocol):
    """A reusable detector that returns original-text character offsets."""

    category: str

    def detect(self, text: str) -> Tuple[Finding, ...]: ...


class ScreeningDetector(Detector, Protocol):
    """Custom detectors also declare the kinds they can emit."""

    kinds: FrozenSet[str]


class BackendError(RuntimeError):
    """A requested backend could not complete; the input is not clean."""


def optional_module(name: str, extra: str) -> ModuleType:
    try:
        return import_module(name)
    except ImportError:
        raise ImportError(
            f"this backend requires pip install 'pygarble[{extra}]'"
        ) from None


def check_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
