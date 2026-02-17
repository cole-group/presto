"""Unit tests for train module."""

from pathlib import Path

import openff.interchange
import pytest
import smee
import smee.converters
import torch
from descent.train import ParameterConfig, Trainable
from openff.toolkit import ForceField, Molecule
from openff.units import unit

from presto.data_utils import create_dataset_with_uniform_weights
from presto.outputs import OutputType
from presto.settings import TrainingSettings
from presto.train import train_adam, train_levenberg_marquardt


# Shared fixtures for both TestTrainAdam and TestTrainLevenbergMarquardt
@pytest.fixture
def ethanol_training_setup():
    """Create a complete training setup for ethanol."""

    # Create molecule and generate conformers
    mol = Molecule.from_smiles("CCO")
    # Use rms_cutoff=0 to ensure we get exactly the requested number of conformers
    mol.generate_conformers(n_conformers=3, rms_cutoff=0.0 * unit.angstrom)

    # Create force field and interchange
    ff = ForceField("openff_unconstrained-2.3.0.offxml")
    interchange = openff.interchange.Interchange.from_smirnoff(ff, mol.to_topology())

    # Convert to tensor representation
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    n_confs = mol.n_conformers

    # Verify we got the expected number of conformers
    assert n_confs == 3, f"Expected 3 conformers but got {n_confs}"

    coords_list = [
        torch.tensor(mol.conformers[i].m_as("angstrom")).unsqueeze(0)
        for i in range(n_confs)
    ]
    coords_all = torch.cat(coords_list, dim=0).requires_grad_(True)

    # Compute reference energies and forces BEFORE creating Trainable
    # to avoid any potential modification of tensor_ff parameters
    energy_all = smee.compute_energy(tensor_top, tensor_ff, coords_all)
    forces_all = -torch.autograd.grad(
        energy_all.sum(), coords_all, create_graph=True, retain_graph=True
    )[0]

    # Create trainable with minimal parameter configuration AFTER computing reference
    parameter_configs = {
        "Bonds": ParameterConfig(
            cols=["k", "length"],
            scales={"k": 1.0, "length": 1.0},
            limits={"k": (1e-8, None), "length": (None, None)},
            regularize={"k": 1.0, "length": 1.0},
            include=None,
            exclude=None,
        ),
    }

    trainable = Trainable(tensor_ff, parameter_configs, {})
    trainable_parameters = trainable.to_values()
    initial_parameters = trainable_parameters.clone()

    # Add small perturbation to trainable parameters to ensure non-zero initial loss
    # This is necessary because trainable parameters start with the same values
    # as the force field used to compute the reference, giving loss=0 initially
    with torch.no_grad():
        trainable_parameters.add_(torch.randn_like(trainable_parameters) * 0.01)

    smiles = mol.to_smiles(mapped=True)

    # Create training dataset (first 2 conformers)
    coords_train = coords_all[:2].detach()
    energy_train = energy_all[:2].detach()
    forces_train = forces_all[:2].detach()

    dataset_train = create_dataset_with_uniform_weights(
        smiles=smiles,
        coords=coords_train,
        energy=energy_train,
        forces=forces_train,
        energy_weight=1000.0,
        forces_weight=0.1,
    )

    # Create test dataset (last conformer - keep as 3D tensor)
    coords_test = coords_all[2:3].detach()
    energy_test = energy_all[2:3].detach()
    forces_test = forces_all[2:3].detach()

    dataset_test = create_dataset_with_uniform_weights(
        smiles=smiles,
        coords=coords_test,
        energy=energy_test,
        forces=forces_test,
        energy_weight=1000.0,
        forces_weight=0.1,
    )

    return {
        "trainable": trainable,
        "trainable_parameters": trainable_parameters,
        "initial_parameters": initial_parameters,
        "topologies": [tensor_top],
        "datasets": [dataset_train],
        "datasets_test": [dataset_test],
        "device": torch.device("cpu"),
    }


class TestTrainAdam:
    """Tests for train_adam function."""

    @pytest.fixture
    def training_settings(self):
        """Create minimal training settings."""
        return TrainingSettings(
            optimiser="adam",
            n_epochs=10,  # Small number for fast tests
            learning_rate=0.001,
            learning_rate_decay=1.0,  # No decay for simplicity
            learning_rate_decay_step=10,
            regularisation_target="initial",
        )

    @pytest.fixture
    def output_paths(self, tmp_path):
        """Create temporary output paths."""
        tensorboard_dir = tmp_path / "tensorboard"
        metrics_file = tmp_path / "metrics.txt"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        return {
            OutputType.TENSORBOARD: tensorboard_dir,
            OutputType.TRAINING_METRICS: metrics_file,
        }

    def test_train_adam_returns_correct_types(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that train_adam returns parameters and trainable."""
        setup = ethanol_training_setup

        result_params, result_trainable = train_adam(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        assert isinstance(result_params, torch.Tensor)
        assert isinstance(result_trainable, Trainable)

    def test_train_adam_parameters_updated(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that parameters are updated during training."""
        setup = ethanol_training_setup

        initial_params = setup["trainable_parameters"].clone()

        result_params, _ = train_adam(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        # Parameters should be different after training
        assert not torch.allclose(result_params, initial_params)

    def test_train_adam_creates_metrics_file(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that metrics file is created."""
        setup = ethanol_training_setup

        train_adam(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        metrics_file = output_paths[OutputType.TRAINING_METRICS]
        assert metrics_file.exists()
        assert metrics_file.stat().st_size > 0

    def test_train_adam_creates_tensorboard_output(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that tensorboard directory contains output."""
        setup = ethanol_training_setup

        train_adam(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        tensorboard_dir = output_paths[OutputType.TENSORBOARD]
        assert tensorboard_dir.exists()
        # Check that tensorboard created some files
        assert any(tensorboard_dir.iterdir())

    def test_train_adam_loss_decreases(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that training loss decreases over epochs."""
        setup = ethanol_training_setup

        # Increase epochs for better convergence testing
        training_settings.n_epochs = 50
        training_settings.learning_rate = 0.01

        train_adam(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        # Read metrics file and check that loss values are written
        metrics_file = output_paths[OutputType.TRAINING_METRICS]
        with open(metrics_file, "r") as f:
            lines = f.readlines()

        # Check that we have metric outputs (at least 6 lines: 0, 10, 20, 30, 40, 50)
        assert len(lines) >= 6, "Should have metrics for at least 6 checkpoints"

        # Parse loss values from all lines
        total_losses = []
        for line in lines:
            parts = line.strip().split()
            energy_loss = float(parts[0])
            force_loss = float(parts[1])
            total_losses.append(energy_loss + force_loss)

        # Check that final loss is reasonable (not NaN, not inf)
        assert not torch.isnan(torch.tensor(total_losses[-1]))
        assert not torch.isinf(torch.tensor(total_losses[-1]))

        # Check that the loss doesn't explode (stays below 100)
        assert total_losses[-1] < 100.0, "Loss should not explode during training"

        # Check that the loss has actually decreased
        assert total_losses[-1] < total_losses[0], (
            "Loss should decrease during training"
        )

    def test_train_adam_with_zero_epochs(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that training with zero epochs returns original parameters."""
        setup = ethanol_training_setup
        training_settings.n_epochs = 0

        initial_params = setup["trainable_parameters"].clone()

        result_params, _ = train_adam(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        # Parameters should be unchanged
        assert torch.allclose(result_params, initial_params)

    def test_train_adam_with_missing_output_paths(
        self, ethanol_training_setup, training_settings
    ):
        """Test that training raises error with missing output paths."""
        setup = ethanol_training_setup

        # Missing TRAINING_METRICS
        incomplete_paths = {OutputType.TENSORBOARD: Path("/tmp/tensorboard")}

        with pytest.raises(ValueError, match="Output paths must contain exactly"):
            train_adam(
                trainable_parameters=setup["trainable_parameters"],
                initial_parameters=setup["initial_parameters"],
                trainable=setup["trainable"],
                topologies=setup["topologies"],
                datasets=setup["datasets"],
                datasets_test=setup["datasets_test"],
                settings=training_settings,
                output_paths=incomplete_paths,
                device=setup["device"],
            )

    def test_train_adam_with_extra_output_paths(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that training raises error with extra output paths."""
        setup = ethanol_training_setup

        # Add an extra unexpected output type
        extra_paths = output_paths.copy()
        extra_paths[OutputType.OFFXML] = Path("/tmp/force_field.offxml")

        with pytest.raises(ValueError, match="Output paths must contain exactly"):
            train_adam(
                trainable_parameters=setup["trainable_parameters"],
                initial_parameters=setup["initial_parameters"],
                trainable=setup["trainable"],
                topologies=setup["topologies"],
                datasets=setup["datasets"],
                datasets_test=setup["datasets_test"],
                settings=training_settings,
                output_paths=extra_paths,
                device=setup["device"],
            )

    def test_train_adam_parameters_require_grad(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that returned parameters have requires_grad enabled."""
        setup = ethanol_training_setup

        result_params, _ = train_adam(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        assert result_params.requires_grad

    def test_train_adam_with_learning_rate_decay(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test training with learning rate decay."""
        setup = ethanol_training_setup

        # Set decay parameters
        training_settings.n_epochs = 30
        training_settings.learning_rate = 0.01
        training_settings.learning_rate_decay = 0.95
        training_settings.learning_rate_decay_step = 10

        result_params, _ = train_adam(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        # Should complete without error and return updated parameters
        assert result_params is not None

    def test_train_adam_with_multiple_molecules(self, training_settings, output_paths):
        """Test training with multiple molecules (different conformers of ethanol)."""
        # Create two ethanol molecules with different conformers
        mol1 = Molecule.from_smiles("CCO")
        mol1.generate_conformers(n_conformers=2, rms_cutoff=0.0 * unit.angstrom)

        mol2 = Molecule.from_smiles("CCO")
        mol2.generate_conformers(n_conformers=2, rms_cutoff=0.0 * unit.angstrom)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        # Create interchanges for both molecules using the same force field
        interchange1 = openff.interchange.Interchange.from_smirnoff(
            ff, mol1.to_topology()
        )
        interchange2 = openff.interchange.Interchange.from_smirnoff(
            ff, mol2.to_topology()
        )

        # Convert to tensor representations
        tensor_ff, [tensor_top1] = smee.converters.convert_interchange(interchange1)
        _, [tensor_top2] = smee.converters.convert_interchange(interchange2)

        # Create trainable from the shared force field
        parameter_configs = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                regularize={"k": 1.0, "length": 1.0},
            ),
        }

        trainable = Trainable(tensor_ff, parameter_configs, {})
        trainable_parameters = trainable.to_values()
        initial_parameters = trainable_parameters.clone()

        # Create datasets for both molecules
        datasets_train = []
        datasets_test = []

        for mol, tensor_top in [(mol1, tensor_top1), (mol2, tensor_top2)]:
            n_confs = mol.n_conformers
            coords_list = [
                torch.tensor(mol.conformers[i].m_as("angstrom")).unsqueeze(0)
                for i in range(n_confs)
            ]
            coords = torch.cat(coords_list, dim=0).requires_grad_(True)

            # Use the shared force field for all molecules
            energy = smee.compute_energy(tensor_top, tensor_ff, coords)
            forces = -torch.autograd.grad(
                energy.sum(), coords, create_graph=True, retain_graph=True
            )[0]

            smiles = mol.to_smiles(mapped=True)

            # Split into train and test
            dataset_train = create_dataset_with_uniform_weights(
                smiles=smiles,
                coords=coords[:1].detach(),
                energy=energy[:1].detach(),
                forces=forces[:1].detach(),
                energy_weight=1000.0,
                forces_weight=0.1,
            )

            dataset_test = create_dataset_with_uniform_weights(
                smiles=smiles,
                coords=coords[1:2].detach(),  # Use 1:2 to keep dimensionality
                energy=energy[1:2].detach(),
                forces=forces[1:2].detach(),  # Use 1:2 to keep dimensionality
                energy_weight=1000.0,
                forces_weight=0.1,
            )

            datasets_train.append(dataset_train)
            datasets_test.append(dataset_test)

        # Train with both molecules
        result_params, result_trainable = train_adam(
            trainable_parameters=trainable_parameters,
            initial_parameters=initial_parameters,
            trainable=trainable,
            topologies=[tensor_top1, tensor_top2],
            datasets=datasets_train,
            datasets_test=datasets_test,
            settings=training_settings,
            output_paths=output_paths,
            device=torch.device("cpu"),
        )

        assert isinstance(result_params, torch.Tensor)
        assert isinstance(result_trainable, Trainable)


class TestTrainLevenbergMarquardt:
    """Tests for train_levenberg_marquardt function.

    Note: Shares the same ethanol_training_setup fixture from TestTrainAdam.
    """

    @pytest.fixture
    def training_settings(self):
        """Create minimal training settings for LM."""
        return TrainingSettings(
            optimiser="lm",
            n_epochs=10,
            learning_rate=0.001,  # Not used by LM but required by settings
            learning_rate_decay=1.0,
            learning_rate_decay_step=10,
            regularisation_target="initial",
        )

    @pytest.fixture
    def output_paths(self, tmp_path):
        """Create temporary output paths."""
        tensorboard_dir = tmp_path / "tensorboard"
        metrics_file = tmp_path / "metrics.txt"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        return {
            OutputType.TENSORBOARD: tensorboard_dir,
            OutputType.TRAINING_METRICS: metrics_file,
        }

    def test_train_lm_returns_correct_types(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that LM optimizer returns correct types."""
        setup = ethanol_training_setup

        result_params, result_trainable = train_levenberg_marquardt(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        # Check return types
        assert isinstance(result_params, torch.Tensor)
        assert isinstance(result_trainable, Trainable)
        assert result_params.requires_grad

    def test_train_lm_creates_metrics_file(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that LM optimizer creates metrics file."""
        setup = ethanol_training_setup

        train_levenberg_marquardt(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        # Check that metrics file was created and has content
        metrics_file = output_paths[OutputType.TRAINING_METRICS]
        assert metrics_file.exists()
        assert metrics_file.stat().st_size > 0

    def test_train_lm_creates_tensorboard_output(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that LM optimizer creates TensorBoard output."""
        setup = ethanol_training_setup

        train_levenberg_marquardt(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        # Check that TensorBoard directory has content
        tensorboard_dir = output_paths[OutputType.TENSORBOARD]
        assert tensorboard_dir.exists()
        assert any(tensorboard_dir.iterdir())

    def test_train_lm_metrics_format_consistent_with_adam(
        self, ethanol_training_setup, training_settings, output_paths, tmp_path
    ):
        """Test that LM metrics file has the same format as Adam metrics file.

        Both optimizers should output 6 values per line:
        - train_energy, train_forces, train_regularisation
        - test_energy, test_forces, test_regularisation
        """
        setup = ethanol_training_setup

        # Train with LM
        train_levenberg_marquardt(
            trainable_parameters=setup["trainable_parameters"].clone(),
            initial_parameters=setup["initial_parameters"].clone(),
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        # Read LM metrics
        lm_metrics_file = output_paths[OutputType.TRAINING_METRICS]
        with open(lm_metrics_file, "r") as f:
            lm_lines = f.readlines()

        # Check that LM produced metrics
        assert len(lm_lines) > 0, "LM should produce at least one metrics line"

        # Check format of each line - should have 6 values (3 train + 3 test)
        for i, line in enumerate(lm_lines):
            parts = line.strip().split()
            assert len(parts) == 6, (
                f"LM metrics line {i} should have 6 values, got {len(parts)}: {line}"
            )
            # All values should be valid floats
            for j, part in enumerate(parts):
                try:
                    float(part)
                except ValueError:
                    pytest.fail(
                        f"LM metrics line {i}, value {j} is not a valid float: {part}"
                    )

    def test_train_lm_metrics_values_are_reasonable(
        self, ethanol_training_setup, training_settings, output_paths
    ):
        """Test that LM metrics values are reasonable (not NaN, not Inf, bounded)."""
        setup = ethanol_training_setup

        train_levenberg_marquardt(
            trainable_parameters=setup["trainable_parameters"],
            initial_parameters=setup["initial_parameters"],
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets=setup["datasets"],
            datasets_test=setup["datasets_test"],
            settings=training_settings,
            output_paths=output_paths,
            device=setup["device"],
        )

        metrics_file = output_paths[OutputType.TRAINING_METRICS]
        with open(metrics_file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            for j, part in enumerate(parts):
                value = float(part)
                assert not (value != value), (
                    f"LM metrics line {i}, value {j} is NaN"
                )  # NaN check
                assert abs(value) < 1e10, (  # Reasonable bound
                    f"LM metrics line {i}, value {j} is too large: {value}"
                )


class TestTrainingRegistry:
    """Tests for the training function registry."""

    def test_adam_registered(self):
        """Test that adam is registered."""
        from presto.train import _TRAINING_FNS_REGISTRY

        assert "adam" in _TRAINING_FNS_REGISTRY

    def test_lm_registered(self):
        """Test that lm is registered."""
        from presto.train import _TRAINING_FNS_REGISTRY

        assert "lm" in _TRAINING_FNS_REGISTRY

    def test_registry_functions_are_callable(self):
        """Test that registered functions are callable."""
        from presto.train import _TRAINING_FNS_REGISTRY

        for fn_name, fn in _TRAINING_FNS_REGISTRY.items():
            assert callable(fn), f"Function {fn_name} is not callable"


class TestExcludedTorsionsNotTrained:
    """Tests to verify that excluded torsions from TrainingSettings are not trained."""

    @pytest.fixture
    def linear_tor_mol_setup(self):
        """Create a set up force field for a molecule which hits all of the
        linear torsion patterns (O=C=CC#CC)."""
        # CCC#N contains linear torsions around the C#N triple bond
        mol = Molecule.from_smiles("O=C=CC#CC")
        mol.generate_conformers(n_conformers=1)

        # Create force field and interchange
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        # Convert to tensor representation
        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

        return {
            "mol": mol,
            "ff": ff,
            "tensor_ff": tensor_ff,
            "tensor_top": tensor_top,
        }

    def test_excluded_torsions_parameters_unchanged_after_training(
        self, linear_tor_mol_setup
    ):
        """Test that excluded torsion parameters remain unchanged during parameter updates.

        This test simulates a training step and verifies that parameters matching
        excluded patterns (linear torsions) do not change even when we modify the
        trainable parameters.
        """
        tensor_ff = linear_tor_mol_setup["tensor_ff"]
        settings = TrainingSettings()

        # Create trainable with default settings
        pruned_parameter_configs = {
            p_type: p_config
            for p_type, p_config in settings.parameter_configs.items()
            if p_type in tensor_ff.potentials_by_type
        }

        trainable = Trainable(
            tensor_ff,
            pruned_parameter_configs,
            settings.attribute_configs,
        )

        # Get initial parameters
        initial_params = trainable.to_values()

        # Get initial force field
        ff_initial = trainable.to_force_field(initial_params)
        initial_proper_torsions = ff_initial.potentials_by_type.get("ProperTorsions")

        assert initial_proper_torsions is not None, "No ProperTorsions in force field"

        # Store initial k values and parameter keys
        initial_k = initial_proper_torsions.parameters[
            :, 0
        ].clone()  # k is first column
        parameter_keys = initial_proper_torsions.parameter_keys

        # Simulate parameter update (modify the trainable parameters)
        modified_params = (
            initial_params + 0.1
        )  # Add small offset to all trainable parameters

        # Get modified force field
        ff_modified = trainable.to_force_field(modified_params)
        modified_proper_torsions = ff_modified.potentials_by_type.get("ProperTorsions")

        # Get modified k values
        modified_k = modified_proper_torsions.parameters[:, 0]  # k is first column

        # Get excluded pattern IDs
        excluded_patterns = [
            item.id for item in settings.parameter_configs["ProperTorsions"].exclude
        ]

        # For each parameter, check if it's excluded
        for idx, key in enumerate(parameter_keys):
            if key.id in excluded_patterns:
                # This is an excluded torsion - its parameters should not change
                assert torch.allclose(initial_k[idx], modified_k[idx], atol=1e-10), (
                    f"Excluded torsion {key.id} changed from k={initial_k[idx]} to k={modified_k[idx]}"
                )
            else:  # Check that it has changed
                assert not torch.allclose(
                    initial_k[idx], modified_k[idx], atol=1e-10
                ), f"Non-excluded torsion {key.id} did not change as expected."
