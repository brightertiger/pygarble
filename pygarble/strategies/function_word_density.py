"""Compatibility pointer to the gibberish strategy implementation."""

from typing import TYPE_CHECKING

from .._compat import alias_module as _alias_module

if TYPE_CHECKING:
    from ..gibberish.strategies.function_word_density import *  # noqa

_alias_module(__name__, "pygarble.gibberish.strategies.function_word_density")
