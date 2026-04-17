"""Unit tests for mlp module."""

import math

import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit as off_unit
from openmmml import MLPotential

from presto._exceptions import InvalidSettingsError
from presto.mlp import (
    AvailableModels,
    _cache,
    get_mlp,
    validate_model_charge_compatibility,
)

# Models that require NNPOps
NNPOPS_MODELS = {
    "egret-1",
    "mace-off23-small",
    "mace-off23-medium",
    "mace-off23-large",
}

EXPECTED_MODEL_ENERGIES = {
    "aceff-2.0": -89.77400970458984,
    "mace-off23-small": -963.051142471468,
    "mace-off23-medium": -963.073736333697,
    "mace-off23-large": -963.177635381956,
    "mace-omol-0-extra-large": -19.449931963902,
    "egret-1": -963.030539966774,
    "aimnet2": -200784.718554282794,
    "orb-v3-conservative-omol": -200671.118723808293,
}


class TestAvailableModels:
    """Tests for AvailableModels type."""

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param(model, id=model)
            for model in __import__("typing").get_args(AvailableModels)
        ],
    )
    def test_all_models_can_create_systems(self, model_name):
        """Test that all available models can be loaded and create OpenMM systems.

        This is an integration test that actually loads models and creates systems.
        """
        from openff.toolkit import Molecule

        _cache.clear()

        # Use a small neutral molecule for compatibility with all models
        mol = Molecule.from_smiles("O")  # Water
        topology = mol.to_topology().to_openmm()

        # Actually load the model
        potential = get_mlp(model_name)
        assert potential is not None
        assert isinstance(potential, MLPotential)

        # Create a real OpenMM system
        system = potential.createSystem(topology)

        # Basic sanity checks on the system
        assert system.getNumParticles() == topology.getNumAtoms()
        assert system.getNumForces() > 0

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param(model, id=model)
            for model in __import__("typing").get_args(AvailableModels)
        ],
    )
    @pytest.mark.slow
    def test_all_models_can_calculate_energy(self, model_name):
        """Test that all available models can calculate an energy for water.

        Also check we don't have any regressions that have changed the energy,
        though OpenMM-ML does have energy tests in CI now.
        """
        import openmm

        _cache.clear()

        POSITIONS = (
            np.array(
                [
                    [-0.00081616, 0.36637843, -0.0],
                    [-0.8123162, -0.18348211, -0.0],
                    [0.81313236, -0.18289632, 0.0],
                ]
            )
            * off_unit.angstroms
        )

        # Use a small neutral molecule for compatibility with all models.
        mol = Molecule.from_smiles("O")
        mol.add_conformer(POSITIONS)
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

    def test_invalid_model_name_raises_error(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Invalid model name"):
            get_mlp("invalid-model-name")


class TestValidateModelChargeCompatibility:
    """Tests for validate_model_charge_compatibility function."""

    @pytest.mark.parametrize(
        "model_name",
        ["egret-1", "mace-off23-small", "aceff-2.0", "aimnet2"],
    )
    def test_neutral_molecule_with_any_model(self, model_name):
        """Test that neutral molecules work with any model."""
        mol = Molecule.from_smiles("CCO")  # Neutral ethanol
        # Should not raise for any model
        validate_model_charge_compatibility(model_name, mol)

    @pytest.mark.parametrize(
        "model_name",
        ["aimnet2", "aceff-2.0"],
    )
    def test_charged_molecule_with_supporting_model(self, model_name):
        """Test that charged molecules work with charge-supporting models."""
        mol = Molecule.from_smiles("[NH4+]")  # Ammonium cation
        # Should not raise
        validate_model_charge_compatibility(model_name, mol)

    @pytest.mark.parametrize(
        "model_name",
        ["egret-1", "mace-off23-small"],
    )
    def test_charged_molecule_with_unsupported_model_raises(self, model_name):
        """Test that charged molecules with unsupported models raise an error."""
        mol = Molecule.from_smiles("[NH4+]")  # Ammonium cation

        with pytest.raises(
            InvalidSettingsError, match="does not support charged molecules"
        ):
            validate_model_charge_compatibility(model_name, mol)

    def test_error_message_contains_charge_value(self):
        """Test that the error message contains the charge value."""
        mol = Molecule.from_smiles("[NH4+]")  # Ammonium cation

        with pytest.raises(InvalidSettingsError, match=r"charge 1\.0"):
            validate_model_charge_compatibility("egret-1", mol)

    def test_error_message_lists_compatible_models(self):
        """Test that the error message lists compatible models."""
        mol = Molecule.from_smiles("[Cl-]")  # Chloride anion

        with pytest.raises(InvalidSettingsError, match=r"aceff-2.0"):
            validate_model_charge_compatibility("mace-off23-medium", mol)

        with pytest.raises(InvalidSettingsError, match="aimnet2"):
            validate_model_charge_compatibility("mace-off23-medium", mol)

    @pytest.mark.parametrize(
        "smiles,charge",
        [
            ("[NH4+]", 1.0),
            ("[Cl-]", -1.0),
            ("[Ca+2]", 2.0),
        ],
    )
    def test_various_charged_molecules(self, smiles, charge):
        """Test various charged molecules."""
        mol = Molecule.from_smiles(smiles)
        assert abs(mol.total_charge.m - charge) < 1e-6

        # Should work with charge-supporting models
        validate_model_charge_compatibility("aceff-2.0", mol)
        validate_model_charge_compatibility("aimnet2", mol)

        # Should fail with non-supporting models
        with pytest.raises(InvalidSettingsError):
            validate_model_charge_compatibility("egret-1", mol)
