"""Unit tests for mlp module."""

import pytest
from openff.toolkit import Molecule
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
        import openmm
        from openff.toolkit import Molecule

        _cache.clear()

        # Use a neutral molecule for compatibility with all models
        mol = Molecule.from_smiles("CCO")  # Ethanol
        topology = mol.to_topology().to_openmm()

        # Actually load the model
        potential = get_mlp(model_name)
        assert potential is not None
        assert isinstance(potential, MLPotential)

        # Create a real OpenMM system
        system = potential.createSystem(topology)
        assert system is not None
        assert isinstance(system, openmm.System)

        # Basic sanity checks on the system
        assert system.getNumParticles() == topology.getNumAtoms()
        assert system.getNumForces() > 0

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
