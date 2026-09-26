"""Compatibility pointer to :mod:`pygarble.screening.secrets.patterns`."""

from typing import TYPE_CHECKING

from .._compat import alias_module as _alias_module

if TYPE_CHECKING:
    from ..screening.secrets.patterns import *  # noqa: F401,F403

_alias_module(__name__, "pygarble.screening.secrets.patterns")
