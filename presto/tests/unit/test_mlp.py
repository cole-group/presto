"""Unit tests for mlp module."""

import math
from typing import get_args
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit as off_unit
from openmmml import MLPotential

from presto.mlp import KnownModels, _cache, get_mlp


EXPECTED_MODEL_ENERGIES = {
    # Note that AceFF is currently wrong in OpenMM-ML https://github.com/openmm/openmm-ml/issues/137, but
    # this energy is the one after fixes and should be correct (currently failing).
    "aceff-2.0": -89.407890319824,
    "mace-off23-small": -963.051142471468,
    "mace-off23-medium": -963.073736333697,
    "mace-off23-large": -963.177635381956,
    "mace-omol-0-extra-large": -19.449931963902,
    "egret-1": -963.030539966774,
    "aimnet2": -200784.718554282794,
    "orb-v3-conservative-omol": -200671.118723808293,
}


class TestGetMlp:
    """Tests for get_mlp function."""

    @pytest.mark.parametrize(
        "model_name",
        [pytest.param(model, id=model) for model in get_args(KnownModels)],
    )
    def test_all_known_models_can_create_systems(self, model_name):
        """Test that known models can be loaded and create OpenMM systems."""
        _cache.clear()
        mol = Molecule.from_smiles("O")  # Water
        topology = mol.to_topology().to_openmm()

        potential = get_mlp(model_name)
        assert potential is not None
        assert isinstance(potential, MLPotential)

        system = potential.createSystem(topology)
        assert system.getNumParticles() == topology.getNumAtoms()
        assert system.getNumForces() > 0

    @pytest.mark.parametrize(
        "model_name",
        [pytest.param(model, id=model) for model in get_args(KnownModels)],
    )
    @pytest.mark.slow
    def test_all_known_models_can_calculate_energy(self, model_name):
        """Test that known models can calculate an energy for water."""
        import openmm

        _cache.clear()

        positions = (
            np.array(
                [
                    [-0.00081616, 0.36637843, -0.0],
                    [-0.8123162, -0.18348211, -0.0],
                    [0.81313236, -0.18289632, 0.0],
                ]
            )
            * off_unit.angstroms
        )

        mol = Molecule.from_smiles("O")
        mol.add_conformer(positions)
        topology = mol.to_topology().to_openmm()

        potential = get_mlp(model_name)
        system = potential.createSystem(topology)

        integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
        context = openmm.Context(system, integrator)
        context.setPositions(mol.conformers[0].to_openmm())

        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoule_per_mole
        )
        assert math.isfinite(energy)
        assert energy == pytest.approx(
            EXPECTED_MODEL_ENERGIES[model_name], rel=1e-6, abs=1e-3
        )

    def test_get_mlp_accepts_arbitrary_model_name(self):
        """Test arbitrary model names are delegated directly to OpenMM-ML."""
        _cache.clear()
        with patch("presto.mlp.MLPotential") as mock_mlpotential:
            fake = MagicMock()
            mock_mlpotential.return_value = fake
            result = get_mlp("my-custom-model")
            assert result is fake
            mock_mlpotential.assert_called_once_with("my-custom-model")

    def test_get_mlp_forwards_constructor_kwargs(self):
        """Test model constructor kwargs are forwarded and cached separately."""
        _cache.clear()
        with patch("presto.mlp.MLPotential") as mock_mlpotential:
            fake = MagicMock()
            mock_mlpotential.return_value = fake

            result1 = get_mlp("custom", modelPath="a.model")
            result2 = get_mlp("custom", modelPath="a.model")
            result3 = get_mlp("custom", modelPath="b.model")

            assert result1 is fake
            assert result2 is fake
            assert result3 is fake
            assert mock_mlpotential.call_count == 2
