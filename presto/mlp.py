"""Functionality for creating and managing ML potentials."""

from importlib import resources
from typing import Literal, get_args

import loguru
import openff.toolkit
from openff.units import unit as off_unit
from openmmml import MLPotential

from ._exceptions import InvalidSettingsError
from .utils import aimnet2

logger = loguru.logger


AvailableModels = Literal[
    "aceff-2.0",
    "mace-off23-small",
    "mace-off23-medium",
    "mace-off23-large",
    "egret-1",
    "aimnet2_b973c_d3_ens",
    "aimnet2_wb97m_d3_ens",
]
"""Available MLPotential models."""

_cache: dict[AvailableModels, MLPotential] = {}

# Models that support charged molecules
_CHARGE_SUPPORTING_MODELS: set[str] = {
    "aimnet2_b973c_d3_ens",
    "aimnet2_wb97m_d3_ens",
    "aceff-2.0",
}


def supports_charges(model: AvailableModels) -> bool:
    """Check if a given ML potential model supports charged molecules.

    Parameters
    ----------
    model : AvailableModels
        The model name to check.

    Returns
    -------
    bool
        True if the model supports charged molecules, False otherwise.
    """
    return model in _CHARGE_SUPPORTING_MODELS


def validate_model_charge_compatibility(
    model: AvailableModels, mol: openff.toolkit.Molecule
) -> None:
    """Validate that a charged molecule is only used with compatible ML potentials.

    Parameters
    ----------
    model : AvailableModels
        The ML potential model to use.
    mol : openff.toolkit.Molecule
        The molecule to check.

    Raises
    ------
    InvalidSettingsError
        If the molecule is charged but the model does not support charges.
    """
    charge = mol.total_charge.m_as(off_unit.e)

    if abs(charge) > 1e-6 and not supports_charges(model):
        raise InvalidSettingsError(
            f"Model '{model}' does not support charged molecules. "
            f"Molecule has charge {charge:.2f}. "
            f"Only the following models support charges: {sorted(_CHARGE_SUPPORTING_MODELS)}. "
            f"Please use a compatible model or ensure your molecule is neutral."
        )


def load_egret_1() -> MLPotential:
    """Load the Egret-1 MLPotential from local package resources."""
    filename = "EGRET_1.model"
    with resources.path("presto.models", filename) as model_path:
        return MLPotential("mace", modelPath=str(model_path))


def get_mlp(model: AvailableModels) -> MLPotential:
    """Get the MLPotential model based on the specified model name."""

    if model not in get_args(AvailableModels):
        raise ValueError(
            f"Invalid model name: {model}. Available models are: {get_args(AvailableModels)}"
        )

    if model not in _cache:
        if model in aimnet2._AVAILABLE_MODELS:
            # Ensure AIMNet2 models registered
            aimnet2._register_aimnet2_potentials()
        if model == "egret-1":
            _cache[model] = load_egret_1()
        else:
            _cache[model] = MLPotential(model)

    return _cache[model]
