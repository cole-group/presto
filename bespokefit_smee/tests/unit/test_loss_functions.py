"""Unit tests for loss_functions module."""

import pytest
import smee
import smee.converters
import torch
from descent.train import ParameterConfig, Trainable
from openff.toolkit import ForceField, Molecule

from bespokefit_smee.loss import (
    compute_regularisation_loss,
)


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

        # Create a minimal trainable with regularization settings
        parameter_configs = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                limits={"k": (None, None), "length": (None, None)},
                regularize={"k": 100.0, "length": 100.0},
                include=None,
                exclude=None,
            ),
        }

        trainable = Trainable(tensor_ff, parameter_configs, {})
        trainable_parameters = trainable.to_values()
        n_atoms = mol.n_atoms

        return trainable, trainable_parameters, n_atoms

    def test_returns_tensor(self, simple_trainable_and_params):
        """Test that function returns a tensor."""
        trainable, params, n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        result = compute_regularisation_loss(
            trainable, params, initial_params, "initial", n_atoms
        )
        assert isinstance(result, torch.Tensor)

    def test_penalty_zero_when_params_unchanged(self, simple_trainable_and_params):
        """Test that penalty is zero when parameters are unchanged."""
        trainable, params, n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        result = compute_regularisation_loss(
            trainable, params, initial_params, "initial", n_atoms
        )
        # Should be very small (zero or near-zero)
        assert result.item() < 1e-6

    def test_penalty_increases_with_parameter_change(self, simple_trainable_and_params):
        """Test that penalty increases when parameters change."""
        trainable, params, n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        # Modify parameters
        modified_params = params.clone()
        modified_params = modified_params + 0.1

        penalty_original = compute_regularisation_loss(
            trainable, params, initial_params, "initial", n_atoms
        )
        penalty_modified = compute_regularisation_loss(
            trainable, modified_params, initial_params, "initial", n_atoms
        )

        assert penalty_modified > penalty_original

    def test_penalty_scales_with_strength(self, simple_trainable_and_params):
        """Test that penalty scales with regularisation strength."""
        trainable, params, n_atoms = simple_trainable_and_params
        initial_params = params.clone()
        modified_params = initial_params + 0.1

        # Create trainable with weak regularisation
        parameter_configs_weak = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                regularize={"k": 10.0, "length": 10.0},
            ),
        }
        trainable_weak = Trainable(trainable._force_field, parameter_configs_weak, {})

        # Create trainable with strong regularisation
        parameter_configs_strong = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                regularize={"k": 1000.0, "length": 1000.0},
            ),
        }
        trainable_strong = Trainable(
            trainable._force_field, parameter_configs_strong, {}
        )

        penalty_weak = compute_regularisation_loss(
            trainable_weak, modified_params, initial_params, "initial", n_atoms
        )
        penalty_strong = compute_regularisation_loss(
            trainable_strong,
            modified_params,
            initial_params,
            "initial",
            n_atoms,
        )

        assert penalty_strong > penalty_weak

    def test_regularisation_to_zero(self, simple_trainable_and_params):
        """Test regularisation towards zero."""
        trainable, params, n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        result = compute_regularisation_loss(
            trainable, params, initial_params, "zero", n_atoms
        )
        assert isinstance(result, torch.Tensor)

    def test_invalid_regularisation_target_raises_error(
        self, simple_trainable_and_params
    ):
        """Test that invalid regularisation value raises error."""
        trainable, params, n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        with pytest.raises(NotImplementedError):
            compute_regularisation_loss(
                trainable, params, initial_params, "invalid", n_atoms
            )

    def test_regularisation_with_zero_strength(self, simple_trainable_and_params):
        """Test that regularisation with zero strength returns zero loss."""
        trainable, params, n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        # Create trainable with zero regularisation strength
        parameter_configs_zero = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                regularization_strengths={"k": 0.0, "length": 0.0},
            ),
        }
        trainable_zero = Trainable(trainable._force_field, parameter_configs_zero, {})

        # Modify parameters significantly
        modified_params = params + 10.0

        result = compute_regularisation_loss(
            trainable_zero,
            modified_params,
            initial_params,
            "initial",
            n_atoms,
        )
        # Should be zero when regularisation strength is zero
        assert result.item() == 0.0
