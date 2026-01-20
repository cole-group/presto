"""Unit tests for mlp module."""

import pytest
from openff.toolkit import Molecule
from openmmml import MLPotential

from bespokefit_smee._exceptions import InvalidSettingsError
from bespokefit_smee.mlp import (
    _CHARGE_SUPPORTING_MODELS,
    AvailableModels,
    _cache,
    get_mlp,
    load_egret_1,
    supports_charges,
    validate_model_charge_compatibility,
)

# Check if NNPOps is available (required for EGRET-1 and MACE models)
try:
    import NNPOps  # noqa: F401

    NNPOPS_AVAILABLE = True
except ImportError:
    NNPOPS_AVAILABLE = False

requires_nnpops = pytest.mark.skipif(
    not NNPOPS_AVAILABLE,
    reason="NNPOps not available (required for EGRET-1 and MACE models)",
)


class TestAvailableModels:
    """Tests for AvailableModels type."""

    def test_available_models_defined(self):
        """Test that available models are defined."""
        from typing import get_args

        available = get_args(AvailableModels)
        assert len(available) > 0
        assert "egret-1" in available
        assert "mace-off23-small" in available


class TestLoadEgret1:
    """Tests for load_egret_1 function."""

    @requires_nnpops
    def test_load_egret_returns_mlpotential(self):
        """Test that loading EGRET-1 returns an MLPotential."""
        potential = load_egret_1()
        assert isinstance(potential, MLPotential)

    @requires_nnpops
    def test_load_egret_is_cached(self):
        """Test that loaded potential is cached."""
        _cache.clear()
        load_egret_1()
        # Cache should now have it if get_mlp was used
        # But load_egret_1 doesn't cache directly


class TestGetMlp:
    """Tests for get_mlp function."""

    @requires_nnpops
    def test_get_mlp_egret(self):
        """Test getting EGRET-1 model."""
        _cache.clear()
        potential = get_mlp("egret-1")
        assert isinstance(potential, MLPotential)

    @requires_nnpops
    def test_get_mlp_caches_result(self):
        """Test that get_mlp caches the result."""
        _cache.clear()
        potential1 = get_mlp("egret-1")
        potential2 = get_mlp("egret-1")
        assert potential1 is potential2

    def test_get_mlp_invalid_model_raises_error(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Invalid model name"):
            get_mlp("invalid-model-name")

    @requires_nnpops
    @pytest.mark.parametrize(
        "model_name",
        ["egret-1", "mace-off23-small"],
    )
    def test_get_mlp_valid_models(self, model_name):
        """Test that valid models can be loaded."""
        _cache.clear()
        potential = get_mlp(model_name)
        assert isinstance(potential, MLPotential)

    @requires_nnpops
    def test_cache_persists_across_calls(self):
        """Test that cache persists across multiple calls."""
        _cache.clear()

        # Load multiple models
        pot1 = get_mlp("egret-1")
        pot2 = get_mlp("mace-off23-small")

        # Verify they're cached
        assert "egret-1" in _cache
        assert "mace-off23-small" in _cache

        # Verify same instances are returned
        assert get_mlp("egret-1") is pot1
        assert get_mlp("mace-off23-small") is pot2

    @requires_nnpops
    def test_get_mlp_different_models_are_different(self):
        """Test that different models return different objects."""
        _cache.clear()
        pot1 = get_mlp("egret-1")
        pot2 = get_mlp("mace-off23-small")
        assert pot1 is not pot2


class TestSupportsCharges:
    """Tests for supports_charges function."""

    def test_aimnet2_models_support_charges(self):
        """Test that AIMNet2 models support charges."""
        assert supports_charges("aimnet2_b973c_d3_ens")
        assert supports_charges("aimnet2_wb97m_d3_ens")

    def test_aceff_supports_charges(self):
        """Test that ACEFF-2.0 supports charges."""
        assert supports_charges("aceff-2.0")

    def test_other_models_do_not_support_charges(self):
        """Test that other models do not support charges."""
        assert not supports_charges("egret-1")
        assert not supports_charges("mace-off23-small")
        assert not supports_charges("mace-off23-medium")
        assert not supports_charges("mace-off23-large")

    def test_charge_supporting_models_set(self):
        """Test that the charge supporting models set is correct."""
        assert "aimnet2_b973c_d3_ens" in _CHARGE_SUPPORTING_MODELS
        assert "aimnet2_wb97m_d3_ens" in _CHARGE_SUPPORTING_MODELS
        assert "aceff-2.0" in _CHARGE_SUPPORTING_MODELS
        assert len(_CHARGE_SUPPORTING_MODELS) == 3


class TestValidateModelChargeCompatibility:
    """Tests for validate_model_charge_compatibility function."""

    def test_neutral_molecule_with_any_model(self):
        """Test that neutral molecules work with any model."""
        mol = Molecule.from_smiles("CCO")  # Neutral ethanol
        # Should not raise for any model
        validate_model_charge_compatibility("egret-1", mol)
        validate_model_charge_compatibility("mace-off23-small", mol)
        validate_model_charge_compatibility("aceff-2.0", mol)
        validate_model_charge_compatibility("aimnet2_b973c_d3_ens", mol)

    def test_charged_molecule_with_aimnet2(self):
        """Test that charged molecules work with AIMNet2."""
        mol = Molecule.from_smiles("[NH4+]")  # Ammonium cation
        # Should not raise
        validate_model_charge_compatibility("aimnet2_b973c_d3_ens", mol)
        validate_model_charge_compatibility("aimnet2_wb97m_d3_ens", mol)

    def test_charged_molecule_with_aceff(self):
        """Test that charged molecules work with ACEFF-2.0."""
        mol = Molecule.from_smiles("[Cl-]")  # Chloride anion
        # Should not raise
        validate_model_charge_compatibility("aceff-2.0", mol)

    def test_charged_molecule_with_unsupported_model_raises(self):
        """Test that charged molecules with unsupported models raise an error."""
        mol = Molecule.from_smiles("[NH4+]")  # Ammonium cation

        with pytest.raises(
            InvalidSettingsError, match="does not support charged molecules"
        ):
            validate_model_charge_compatibility("egret-1", mol)

        with pytest.raises(
            InvalidSettingsError, match="does not support charged molecules"
        ):
            validate_model_charge_compatibility("mace-off23-small", mol)

    def test_error_message_contains_charge_value(self):
        """Test that the error message contains the charge value."""
        mol = Molecule.from_smiles("[NH4+]")  # Ammonium cation

        with pytest.raises(InvalidSettingsError, match=r"charge 1\.0"):
            validate_model_charge_compatibility("egret-1", mol)

    def test_error_message_lists_compatible_models(self):
        """Test that the error message lists compatible models."""
        mol = Molecule.from_smiles("[Cl-]")  # Chloride anion

        with pytest.raises(InvalidSettingsError, match="aceff-2.0"):
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
        validate_model_charge_compatibility("aimnet2_b973c_d3_ens", mol)

        # Should fail with non-supporting models
        with pytest.raises(InvalidSettingsError):
            validate_model_charge_compatibility("egret-1", mol)
