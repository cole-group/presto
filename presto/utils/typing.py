"""Typing utilities for the presto package."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

PathLike = str | Path
TorchDevice = Literal["cpu", "cuda"]

OptimiserName = Literal["adam", "lm"]
"""Allowed optimiser names. 'adam' is Adam, 'lm' is Levenberg-Marquardt."""

FnTypeVar = TypeVar("FnTypeVar", bound=Callable[..., Any])

ValenceType = Literal[
    "Bonds",
    "LinearBonds",
    "Angles",
    "LinearAngles",
    "ProperTorsions",
    "ImproperTorsions",
]

NonLinearValenceType = Literal[
    "Bonds",
    "Angles",
    "ProperTorsions",
    "ImproperTorsions",
]

AllowedAttributeType = Literal[
    "vdW",
    "Electrostatics",
]
