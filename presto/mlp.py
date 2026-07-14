"""Functionality for creating and managing ML potentials."""

import warnings
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal

import loguru
import openff.toolkit
import openmm
import torch
from openff.units import unit as off_unit
from openmmml import MLPotential

if TYPE_CHECKING:
    from .settings import MLPSettings

logger = loguru.logger


KnownModels = Literal[
    "aceff-2.0",
    "mace-off23-small",
    "mace-off23-medium",
    "mace-off23-large",
    "mace-omol-0-extra-large",
    "egret-1",
    "aimnet2",
    "orb-v3-conservative-omol",
]
"""Commonly used model names from OpenMM-ML with dependencies provided by the pixi install from GitHub."""

_cache: dict[tuple[str, tuple[tuple[str, str], ...]], MLPotential] = {}


def _freeze_kwargs(kwargs: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Create a stable cache key for constructor kwargs."""
    return tuple(sorted((key, repr(value)) for key, value in kwargs.items()))


def load_egret_1(**kwargs: Any) -> MLPotential:
    """Load the Egret-1 MLPotential from local package resources."""
    filename = "EGRET_1.model"
    with resources.path("presto.models", filename) as model_path:
        return MLPotential("mace", modelPath=str(model_path), **kwargs)


def get_mlp(model: str, **kwargs: Any) -> MLPotential:
    """Get an OpenMM-ML MLPotential using an arbitrary model identifier."""
    cache_key = (model, _freeze_kwargs(kwargs))
    if cache_key not in _cache:
        if model == "egret-1":
            _cache[cache_key] = load_egret_1(**kwargs)
        else:
            _cache[cache_key] = MLPotential(model, **kwargs)

    return _cache[cache_key]


def get_ml_omm_system(
    mol: openff.toolkit.Molecule,
    mlp_settings: "MLPSettings",
    device: torch.device,
) -> openmm.System:
    """Create an OpenMM System using configured OpenMM-ML settings.

    For non-ASE potentials, charge and device are auto-populated unless supplied
    in ``ml_system_kwargs``. For ASE potentials, charge is not auto-injected.
    """
    ml_system_kwargs = dict(mlp_settings.ml_system_kwargs)
    charge = mol.total_charge.m_as(off_unit.e)

    if mlp_settings.ml_potential != "ase":
        ml_system_kwargs.setdefault("charge", charge)
        ml_system_kwargs.setdefault("device", str(device))
    else:
        if abs(charge) > 1e-6:
            warnings.warn(
                "For ml_potential='ase', presto does not automatically pass molecular "
                "charge to OpenMM-ML. Supply charge in mlp_settings.ml_system_kwargs, "
                "as required for your MLP, for example under 'info'.",
                UserWarning,
                stacklevel=2,
            )

    potential = get_mlp(mlp_settings.ml_potential, **mlp_settings.ml_potential_kwargs)
    return potential.createSystem(mol.to_topology().to_openmm(), **ml_system_kwargs)
