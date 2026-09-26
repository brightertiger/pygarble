"""Compatibility exports for :mod:`pygarble.gibberish.strategies`."""

from typing import TYPE_CHECKING, Any, List

from ..gibberish import strategies as _strategies

if TYPE_CHECKING:
    from ..gibberish.strategies import *  # noqa: F401,F403

__all__ = _strategies.__all__


def __getattr__(name: str) -> Any:
    return getattr(_strategies, name)


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(dir(_strategies)))
