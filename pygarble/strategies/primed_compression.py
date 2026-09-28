"""Compatibility pointer to the gibberish strategy implementation."""

from typing import TYPE_CHECKING

from .._compat import alias_module as _alias_module

if TYPE_CHECKING:
    from ..gibberish.strategies.primed_compression import *  # noqa

_alias_module(__name__, "pygarble.gibberish.strategies.primed_compression")
