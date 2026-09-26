"""Old import paths share canonical modules, including mutable state."""

import sys
from importlib import import_module
from typing import Tuple


def alias_module(
    name: str, target: str, children: Tuple[str, ...] = ()
) -> None:
    module = import_module(target)
    for child in children:
        sys.modules[name + "." + child] = import_module(target + "." + child)
    sys.modules[name] = module
