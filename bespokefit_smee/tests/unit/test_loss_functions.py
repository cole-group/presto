"""Unit tests for loss_functions module."""

import pytest
import smee
import torch
from descent.train import ParameterConfig, Trainable
from openff.toolkit import ForceField, Molecule

from bespokefit_smee.loss import (
    compute_regularisation_loss,
)
from bespokefit_smee.settings import RegularisationSettings


class TestComputeRegularisationPenalty:
    """Tests for compute_regularisation_loss function."""

    @pytest.fixture
    def simple_trainable_and_params(self):
        """Create a simple trainable and parameters for testing."""
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        import openff.interchange

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

        # Create a minimal trainable
        parameter_configs = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
            ),
        }

        trainable = Trainable(tensor_ff, parameter_configs, {})
        trainable_parameters = trainable.to_values()

        return trainable, trainable_parameters

    def test_returns_tensor(self, simple_trainable_and_params):
        """Test that function returns a tensor."""
        trainable, params = simple_trainable_and_params
        initial_params = params.clone()
        settings = RegularisationSettings()

        result = compute_regularisation_loss(
            trainable, params, initial_params, settings
        )
        assert isinstance(result, torch.Tensor)

    def test_penalty_zero_when_params_unchanged(self, simple_trainable_and_params):
        """Test that penalty is zero when parameters are unchanged."""
        trainable, params = simple_trainable_and_params
        initial_params = params.clone()
        settings = RegularisationSettings(
            regularisation_target="initial",
            regularisation_strength=100.0,
            parameters={"Bonds": ["k", "length"]},
        )

        result = compute_regularisation_loss(
            trainable, params, initial_params, settings
        )
        # Should be very small (zero or near-zero)
        assert result.item() < 1e-6

    def test_penalty_increases_with_parameter_change(self, simple_trainable_and_params):
        """Test that penalty increases when parameters change."""
        trainable, params = simple_trainable_and_params
        initial_params = params.clone()
        settings = RegularisationSettings(
            regularisation_target="initial",
            regularisation_strength=100.0,
            parameters={"Bonds": ["k", "length"]},
        )

        # Modify parameters
        modified_params = params.clone()
        modified_params = modified_params + 0.1

        penalty_original = compute_regularisation_loss(
            trainable, params, initial_params, settings
        )
        penalty_modified = compute_regularisation_loss(
            trainable, modified_params, initial_params, settings
        )

        assert penalty_modified > penalty_original

    def test_penalty_scales_with_strength(self, simple_trainable_and_params):
        """Test that penalty scales with regularisation strength."""
        trainable, params = simple_trainable_and_params
        initial_params = params.clone()
        modified_params = params + 0.1

        settings_weak = RegularisationSettings(
            regularisation_target="initial",
            regularisation_strength=10.0,
            parameters={"Bonds": ["k", "length"]},
        )
        settings_strong = RegularisationSettings(
            regularisation_target="initial",
            regularisation_strength=1000.0,
            parameters={"Bonds": ["k", "length"]},
        )

        penalty_weak = compute_regularisation_loss(
            trainable, modified_params, initial_params, settings_weak
        )
        penalty_strong = compute_regularisation_loss(
            trainable, modified_params, initial_params, settings_strong
        )

        assert penalty_strong > penalty_weak

    def test_regularisation_to_zero(self, simple_trainable_and_params):
        """Test regularisation towards zero."""
        trainable, params = simple_trainable_and_params
        initial_params = params.clone()
        settings = RegularisationSettings(
            regularisation_target="zero",
            regularisation_strength=100.0,
            parameters={"Bonds": ["k", "length"]},
        )

        result = compute_regularisation_loss(
            trainable, params, initial_params, settings
        )
        assert isinstance(result, torch.Tensor)

    def test_invalid_regularisation_target_raises_error(
        self, simple_trainable_and_params
    ):
        """Test that invalid regularisation value raises error."""
        trainable, params = simple_trainable_and_params
        initial_params = params.clone()

        # Create settings with invalid value by bypassing validation
        settings = RegularisationSettings()
        object.__setattr__(settings, "regularisation_target", "invalid")

        with pytest.raises(NotImplementedError):
            compute_regularisation_loss(trainable, params, initial_params, settings)

    def test_regularised_parameter_idxs_cached(self, simple_trainable_and_params):
        """Test that regularised parameter indices are cached."""
        trainable, params = simple_trainable_and_params
        initial_params = params.clone()
        settings = RegularisationSettings()

        # First call should compute and cache
        compute_regularisation_loss(trainable, params, initial_params, settings)
        assert hasattr(trainable, "regularised_parameter_idxs")

        # Second call should use cached value
        cached_idxs = trainable.regularised_parameter_idxs
        compute_regularisation_loss(trainable, params, initial_params, settings)
        assert trainable.regularised_parameter_idxs is cached_idxs
