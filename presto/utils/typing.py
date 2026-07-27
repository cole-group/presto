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
    # smee labels both Lennard-Jones and DoubleExponential potentials with type
    # "vdW", so this is the training-config key for either nonbonded form.
    "vdW",
]

NonLinearValenceType = Literal[
    "Bonds",
    "Angles",
    "ProperTorsions",
    "ImproperTorsions",
    # OpenFF handler names used for bespoke type generation. "DoubleExponential"
    # is the plugin nonbonded handler; "vdW" is standard Lennard-Jones.
    "vdW",
    "DoubleExponential",
]

AllowedAttributeType = Literal[
    "vdW",
    "Electrostatics",
]
