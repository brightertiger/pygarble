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
    except ImportError as error:
        missing = getattr(error, "name", None) or ""
        root = name.split(".")[0]
        hint = f"this backend requires pip install 'pygarble[{extra}]'"
        if missing == root or missing.startswith(root + "."):
            # The package itself is absent; the traceback adds nothing.
            raise ImportError(hint) from None
        if (
            isinstance(error, ModuleNotFoundError)
            and error.name == "pkg_resources"
        ):
            # python-stdnum imports pkg_resources on Python 3.8.
            raise ImportError(
                f"{root} is installed but needs pkg_resources: "
                "pip install setuptools"
            ) from error
        raise ImportError(
            f"{root} is installed but failed to import ({error}); {hint}"
        ) from error


def check_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
