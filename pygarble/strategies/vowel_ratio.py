"""Compatibility pointer to the gibberish strategy implementation."""

from typing import TYPE_CHECKING

from .._compat import alias_module as _alias_module

if TYPE_CHECKING:
    from ..gibberish.strategies.vowel_ratio import *  # noqa: F401,F403

_alias_module(__name__, "pygarble.gibberish.strategies.vowel_ratio")
