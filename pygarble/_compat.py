"""Old import paths share canonical modules, including mutable state."""

import sys
import warnings
from importlib import import_module
from typing import Any, Mapping, Tuple


def alias_module(
    name: str, target: str, children: Tuple[str, ...] = ()
) -> None:
    module = import_module(target)
    for child in children:
        sys.modules[name + "." + child] = import_module(target + "." + child)
    sys.modules[name] = module


def forward_removed(old: str, name: str, homes: Mapping[str, str]) -> Any:
    """Resolve a name ``old`` exported in 0.8.0 from its current home.

    Call it from a module ``__getattr__``; the warning fires on access,
    never on import.
    """
    home = homes.get(name)
    if home is None:
        raise AttributeError(f"module {old!r} has no attribute {name!r}")
    warnings.warn(
        f"{old}.{name} is deprecated and will be removed in a future "
        f"release; import {name} from {home} instead",
        DeprecationWarning,
        stacklevel=3,
    )
    return getattr(import_module(home), name)
