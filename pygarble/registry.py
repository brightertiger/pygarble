"""Compatibility pointer to :mod:`pygarble.gibberish.registry`."""

from typing import TYPE_CHECKING

from ._compat import alias_module as _alias_module

if TYPE_CHECKING:
    from .gibberish.registry import *  # noqa: F401,F403

_alias_module(__name__, "pygarble.gibberish.registry")
