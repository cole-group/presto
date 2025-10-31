"""Functionality for creating Open"""

from importlib import resources
from typing import Literal, get_args

import loguru
from openmmml import MLPotential

from .utils import aimnet2

logger = loguru.logger


AvailableModels = Literal[
    "mace-off23-small",
    "mace-off23-medium",
    "mace-off23-large",
    "egret-1",
    "aimnet2_b973c_d3_ens",
    "aimnet2_wb97m_d3_ens",
]
"""Available MLPotential models."""

_cache: dict[AvailableModels, MLPotential] = {}


def load_egret_1() -> MLPotential:
    """Load the Egret-1 MLPotential from local package resources."""
    filename = "EGRET_1.model"
    with resources.path("bespokefit_smee.models", filename) as model_path:
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
