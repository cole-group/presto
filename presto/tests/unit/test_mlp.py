"""Unit tests for mlp module."""

from unittest.mock import MagicMock, patch

import pytest
from openff.toolkit import Molecule
from openmmml import MLPotential

from presto._exceptions import InvalidSettingsError
from presto.mlp import (
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

    def test_load_egret_returns_mlpotential(self):
        """Test that loading EGRET-1 returns an MLPotential."""
        with (
            patch("presto.mlp.resources.path") as mock_path,
            patch("presto.mlp.MLPotential") as mock_mlp,
        ):
            mock_path.return_value.__enter__.return_value = "fake_path"
            mock_mlp.return_value = MagicMock(spec=MLPotential)

            potential = load_egret_1()
            assert potential == mock_mlp.return_value
            mock_mlp.assert_called_once_with("mace", modelPath="fake_path")


class TestGetMlp:
    """Tests for get_mlp function."""

    def test_get_mlp_egret(self):
        """Test getting EGRET-1 model."""
        _cache.clear()
        with patch("presto.mlp.load_egret_1") as mock_load:
            mock_load.return_value = MagicMock(spec=MLPotential)
            potential = get_mlp("egret-1")
            assert potential == mock_load.return_value
            assert _cache["egret-1"] == potential

    def test_get_mlp_caches_result(self):
        """Test that get_mlp caches the result."""
        _cache.clear()
        with patch("presto.mlp.MLPotential") as mock_mlp:
            mock_mlp.return_value = MagicMock(spec=MLPotential)
            potential1 = get_mlp("mace-off23-small")
            potential2 = get_mlp("mace-off23-small")
            assert potential1 is potential2
            assert mock_mlp.call_count == 1

    def test_get_mlp_invalid_model_raises_error(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Invalid model name"):
            get_mlp("invalid-model-name")

    @pytest.mark.parametrize(
        "model_name",
        ["mace-off23-small", "mace-off23-medium"],
    )
    def test_get_mlp_valid_models(self, model_name):
        """Test that valid models can be loaded."""
        _cache.clear()
        with patch("presto.mlp.MLPotential") as mock_mlp:
            mock_mlp.return_value = MagicMock(spec=MLPotential)
            potential = get_mlp(model_name)
            assert potential == mock_mlp.return_value

    def test_cache_persists_across_calls(self):
        """Test that cache persists across multiple calls."""
        _cache.clear()
        with (
            patch("presto.mlp.load_egret_1") as mock_load,
            patch("presto.mlp.MLPotential") as mock_mlp,
        ):
            mock_load.return_value = MagicMock(spec=MLPotential)
            mock_mlp.return_value = MagicMock(spec=MLPotential)

            # Load multiple models
            pot1 = get_mlp("egret-1")
            pot2 = get_mlp("mace-off23-small")

            # Verify they're cached
            assert "egret-1" in _cache
            assert "mace-off23-small" in _cache

            # Verify same instances are returned
            assert get_mlp("egret-1") is pot1
            assert get_mlp("mace-off23-small") is pot2

    def test_get_mlp_different_models_are_different(self):
        """Test that different models return different objects."""
        _cache.clear()
        with patch("presto.mlp.MLPotential") as mock_mlp:
            mock_mlp.side_effect = [
                MagicMock(spec=MLPotential),
                MagicMock(spec=MLPotential),
            ]
            pot1 = get_mlp("mace-off23-small")
            pot2 = get_mlp("mace-off23-medium")
            assert pot1 is not pot2

    def test_get_mlp_aimnet2_registration(self):
        """Test that AIMNet2 models trigger registration."""
        _cache.clear()
        with (
            patch("presto.mlp.aimnet2._register_aimnet2_potentials") as mock_reg,
            patch("presto.mlp.MLPotential") as mock_mlp,
        ):
            mock_mlp.return_value = MagicMock(spec=MLPotential)
            get_mlp("aimnet2_b973c_d3_ens")
            mock_reg.assert_called_once()


class TestSupportsCharges:
    """Tests for supports_charges function."""

    @pytest.mark.parametrize(
        "model_name",
        ["aimnet2_b973c_d3_ens", "aimnet2_wb97m_d3_ens", "aceff-2.0"],
    )
    def test_models_that_support_charges(self, model_name):
        """Test that charge-supporting models return True."""
        assert supports_charges(model_name)
        assert model_name in _CHARGE_SUPPORTING_MODELS

    @pytest.mark.parametrize(
        "model_name",
        ["egret-1", "mace-off23-small", "mace-off23-medium", "mace-off23-large"],
    )
    def test_models_that_do_not_support_charges(self, model_name):
        """Test that non-charge-supporting models return False."""
        assert not supports_charges(model_name)

    def test_charge_supporting_models_count(self):
        """Test that the charge supporting models set has the correct count."""
        assert len(_CHARGE_SUPPORTING_MODELS) == 3


class TestValidateModelChargeCompatibility:
    """Tests for validate_model_charge_compatibility function."""

    @pytest.mark.parametrize(
        "model_name",
        ["egret-1", "mace-off23-small", "aceff-2.0", "aimnet2_b973c_d3_ens"],
    )
    def test_neutral_molecule_with_any_model(self, model_name):
        """Test that neutral molecules work with any model."""
        mol = Molecule.from_smiles("CCO")  # Neutral ethanol
        # Should not raise for any model
        validate_model_charge_compatibility(model_name, mol)

    @pytest.mark.parametrize(
        "model_name",
        ["aimnet2_b973c_d3_ens", "aimnet2_wb97m_d3_ens", "aceff-2.0"],
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
