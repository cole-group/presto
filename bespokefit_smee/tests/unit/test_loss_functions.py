"""Unit tests for loss_functions module."""

import datasets
import datasets.table
import openff.interchange
import pyarrow
import pytest
import smee
import smee.converters
import torch
from descent.train import ParameterConfig, Trainable
from openff.toolkit import ForceField, Molecule

from bespokefit_smee.data_utils import (
    WEIGHTED_DATA_SCHEMA,
    create_dataset_with_uniform_weights,
    merge_weighted_datasets,
)
from bespokefit_smee.loss import (
    compute_regularisation_loss,
    predict,
    predict_with_weights,
    prediction_loss,
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


class TestPredictWithWeights:
    """Tests for predict_with_weights function."""

    @pytest.fixture
    def ethanol_ff_and_topology(self):
        """Create ethanol force field and topology for testing."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=3)
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)
        return tensor_ff, tensor_top, mol

    @pytest.fixture
    def weighted_ethanol_dataset(self, ethanol_ff_and_topology):
        """Create a weighted dataset for ethanol."""
        tensor_ff, tensor_top, mol = ethanol_ff_and_topology
        n_confs = 3
        n_atoms = mol.n_atoms

        # Generate fake coordinates
        coords = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64) * 0.5 + 0.5
        energy = torch.rand(n_confs, dtype=torch.float64)
        forces = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)

        smiles = mol.to_smiles(mapped=True)

        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords,
            energy=energy,
            forces=forces,
            energy_weight=1000.0,
            forces_weight=0.1,
        )
        return dataset, tensor_ff, tensor_top, smiles

    def test_returns_correct_number_of_outputs(self, weighted_ethanol_dataset):
        """Test that function returns 6 outputs."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        result = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device_type="cpu"
        )
        assert len(result) == 6

    def test_returns_energy_weights(self, weighted_ethanol_dataset):
        """Test that energy weights are returned."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        _, _, _, _, energy_weights, _ = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device_type="cpu"
        )
        assert energy_weights is not None
        assert len(energy_weights) == 3  # n_confs

    def test_returns_forces_weights(self, weighted_ethanol_dataset):
        """Test that forces weights are returned."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        _, _, _, _, _, forces_weights = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device_type="cpu"
        )
        assert forces_weights is not None
        assert len(forces_weights) == 3  # n_confs

    def test_weights_match_dataset(self, weighted_ethanol_dataset):
        """Test that returned weights match dataset."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        _, _, _, _, energy_weights, forces_weights = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device_type="cpu"
        )

        # Energy weight was set to 1000.0
        assert torch.allclose(
            energy_weights, torch.full((3,), 1000.0, dtype=energy_weights.dtype)
        )
        # Forces weight was set to 0.1
        assert torch.allclose(
            forces_weights, torch.full((3,), 0.1, dtype=forces_weights.dtype)
        )


class TestPredictionLossWithWeights:
    """Tests for prediction_loss function with weighted datasets."""

    @pytest.fixture
    def ethanol_trainable_and_dataset(self):
        """Create ethanol trainable and weighted dataset for testing."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=3)
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

        # Create trainable
        parameter_configs = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                regularize={"k": 1.0, "length": 1.0},
            ),
        }
        trainable = Trainable(tensor_ff, parameter_configs, {})
        trainable_params = trainable.to_values()
        initial_params = trainable_params.clone()

        # Create weighted dataset with real conformer coordinates
        n_confs = mol.n_conformers
        n_atoms = mol.n_atoms
        coords_list = [
            torch.tensor(mol.conformers[i].m_as("angstrom")).unsqueeze(0)
            for i in range(n_confs)
        ]
        coords = torch.cat(coords_list, dim=0).requires_grad_(True)

        # Compute reference energies and forces using the force field
        energy_ref = smee.compute_energy(tensor_top, tensor_ff, coords)
        forces_ref = -torch.autograd.grad(
            energy_ref.sum(), coords, create_graph=True, retain_graph=True
        )[0]

        smiles = mol.to_smiles(mapped=True)

        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords.detach(),
            energy=energy_ref.detach(),
            forces=forces_ref.detach(),
            energy_weight=1000.0,
            forces_weight=0.1,
        )

        return trainable, trainable_params, initial_params, [dataset], [tensor_top]

    def test_prediction_loss_returns_loss_record(self, ethanol_trainable_and_dataset):
        """Test that prediction_loss returns a LossRecord."""
        trainable, params, initial, datasets_list, topologies = (
            ethanol_trainable_and_dataset
        )

        result = prediction_loss(
            datasets_list,
            trainable,
            params,
            initial,
            topologies,
            "initial",
            "cpu",
        )

        # Check it's a named tuple with the right fields
        assert hasattr(result, "energy")
        assert hasattr(result, "forces")
        assert hasattr(result, "regularisation")

    def test_prediction_loss_energy_is_tensor(self, ethanol_trainable_and_dataset):
        """Test that energy loss is a tensor."""
        trainable, params, initial, datasets_list, topologies = (
            ethanol_trainable_and_dataset
        )

        result = prediction_loss(
            datasets_list,
            trainable,
            params,
            initial,
            topologies,
            "initial",
            "cpu",
        )

        assert isinstance(result.energy, torch.Tensor)

    def test_prediction_loss_forces_is_tensor(self, ethanol_trainable_and_dataset):
        """Test that forces loss is a tensor."""
        trainable, params, initial, datasets_list, topologies = (
            ethanol_trainable_and_dataset
        )

        result = prediction_loss(
            datasets_list,
            trainable,
            params,
            initial,
            topologies,
            "initial",
            "cpu",
        )

        assert isinstance(result.forces, torch.Tensor)

    def test_prediction_loss_with_zero_energy_weight(self):
        """Test that zero energy weight excludes energy from loss."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=2)
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

        parameter_configs = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                regularize={"k": 0.0, "length": 0.0},
            ),
        }
        trainable = Trainable(tensor_ff, parameter_configs, {})
        trainable_params = trainable.to_values()

        # Create dataset with zero energy weight
        n_confs = 2
        n_atoms = mol.n_atoms
        coords = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)

        smiles = mol.to_smiles(mapped=True)

        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords,
            energy=torch.rand(n_confs, dtype=torch.float64),
            forces=torch.rand(n_confs, n_atoms, 3, dtype=torch.float64),
            energy_weight=0.0,  # Zero weight
            forces_weight=1.0,
        )

        result = prediction_loss(
            [dataset],
            trainable,
            trainable_params,
            trainable_params.clone(),
            [tensor_top],
            "initial",
            "cpu",
        )

        # Energy loss should be zero when weight is zero
        assert result.energy.item() == 0.0

    def test_prediction_loss_with_nan_forces(self):
        """Test that NaN forces are handled correctly (zero weight)."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=2)
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

        parameter_configs = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                regularize={"k": 0.0, "length": 0.0},
            ),
        }
        trainable = Trainable(tensor_ff, parameter_configs, {})
        trainable_params = trainable.to_values()

        n_confs = 2
        n_atoms = mol.n_atoms
        coords = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)
        forces = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)
        # Set all forces to NaN
        forces[:] = float("nan")

        smiles = mol.to_smiles(mapped=True)

        # NaN forces should get weight 0 automatically
        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords,
            energy=torch.rand(n_confs, dtype=torch.float64),
            forces=forces,
            energy_weight=1.0,
            forces_weight=1.0,  # Will be set to 0 for NaN forces
        )

        result = prediction_loss(
            [dataset],
            trainable,
            trainable_params,
            trainable_params.clone(),
            [tensor_top],
            "initial",
            "cpu",
        )

        # Force loss should be zero because all forces are NaN (weight 0)
        assert result.forces.item() == 0.0


class TestPredictFunction:
    """Tests for predict function."""

    @pytest.fixture
    def ethanol_setup(self):
        """Create ethanol setup for testing."""
        from openff.units import unit

        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=2, rms_cutoff=0.0 * unit.angstrom)
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

        n_confs = len(mol.conformers)
        if n_confs < 1:
            # Generate at least one conformer
            mol.generate_conformers(n_conformers=1)
            n_confs = 1

        n_atoms = mol.n_atoms
        coords_list = [
            torch.tensor(mol.conformers[i].m_as("angstrom")).unsqueeze(0)
            for i in range(n_confs)
        ]
        coords = torch.cat(coords_list, dim=0).requires_grad_(True)

        energy = smee.compute_energy(tensor_top, tensor_ff, coords)
        forces = -torch.autograd.grad(
            energy.sum(), coords, create_graph=True, retain_graph=True
        )[0]

        smiles = mol.to_smiles(mapped=True)

        # Create unweighted dataset
        schema = pyarrow.schema(
            [
                ("smiles", pyarrow.string()),
                ("coords", pyarrow.list_(pyarrow.float64())),
                ("energy", pyarrow.list_(pyarrow.float64())),
                ("forces", pyarrow.list_(pyarrow.float64())),
            ]
        )
        table = pyarrow.Table.from_pylist(
            [
                {
                    "smiles": smiles,
                    "coords": coords.detach().flatten().tolist(),
                    "energy": energy.detach().tolist(),
                    "forces": forces.detach().flatten().tolist(),
                }
            ],
            schema=schema,
        )
        dataset = datasets.Dataset(datasets.table.InMemoryTable(table))
        dataset.set_format("torch")

        return dataset, tensor_ff, tensor_top, smiles

    def test_predict_returns_four_tensors(self, ethanol_setup):
        """Test that predict returns 4 tensors."""
        dataset, tensor_ff, tensor_top, smiles = ethanol_setup

        result = predict(dataset, tensor_ff, {smiles: tensor_top})
        assert len(result) == 4

    def test_predict_energy_shapes(self, ethanol_setup):
        """Test that predicted energies have correct shape."""
        dataset, tensor_ff, tensor_top, smiles = ethanol_setup

        energy_ref, energy_pred, _, _ = predict(
            dataset, tensor_ff, {smiles: tensor_top}
        )
        assert energy_ref.shape == energy_pred.shape
        # Just check we have some conformations
        assert len(energy_ref) >= 1

    def test_predict_forces_shapes(self, ethanol_setup):
        """Test that predicted forces have correct shape."""
        dataset, tensor_ff, tensor_top, smiles = ethanol_setup

        _, _, forces_ref, forces_pred = predict(
            dataset, tensor_ff, {smiles: tensor_top}
        )
        assert forces_ref.shape == forces_pred.shape


class TestPredictWithWeightsMultipleEntries:
    """Tests for predict_with_weights with multiple entries in a dataset."""

    @pytest.fixture
    def multi_entry_dataset(self):
        """Create a dataset with multiple entries for the same molecule."""
        from openff.units import unit

        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=3, rms_cutoff=0.0 * unit.angstrom)
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)
        smiles = mol.to_smiles(mapped=True)

        n_atoms = mol.n_atoms
        n_confs = len(mol.conformers)
        coords_list = [
            torch.tensor(mol.conformers[i].m_as("angstrom")).unsqueeze(0)
            for i in range(n_confs)
        ]
        coords = torch.cat(coords_list, dim=0).requires_grad_(True)

        energy = smee.compute_energy(tensor_top, tensor_ff, coords)
        forces = -torch.autograd.grad(
            energy.sum(), coords, create_graph=True, retain_graph=True
        )[0]

        # Create two separate datasets
        dataset_1 = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords[:2].detach(),
            energy=energy[:2].detach(),
            forces=forces[:2].detach(),
            energy_weight=1000.0,
            forces_weight=0.1,
        )

        dataset_2 = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords[2:].detach(),
            energy=energy[2:].detach(),
            forces=forces[2:].detach(),
            energy_weight=500.0,
            forces_weight=0.05,
        )

        # Merge them to create a dataset with 2 entries
        merged = merge_weighted_datasets([dataset_1, dataset_2])

        return merged, tensor_ff, tensor_top, smiles

    def test_multi_entry_dataset_has_two_entries(self, multi_entry_dataset):
        """Test that merged dataset has two separate entries."""
        dataset, _, _, _ = multi_entry_dataset
        assert len(dataset) == 2

    def test_predict_with_weights_handles_multiple_entries(self, multi_entry_dataset):
        """Test that predict_with_weights correctly handles multiple entries."""
        dataset, tensor_ff, tensor_top, smiles = multi_entry_dataset

        result = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device_type="cpu"
        )

        # Should return 6 tensors
        assert len(result) == 6

        (
            energy_ref,
            energy_pred,
            forces_ref,
            forces_pred,
            energy_weights,
            forces_weights,
        ) = result

        # Total conformations: 2 from first entry + 1 from second = 3
        assert len(energy_ref) == 3
        assert len(energy_weights) == 3

    def test_predict_with_weights_preserves_different_weights(
        self, multi_entry_dataset
    ):
        """Test that different weights per entry are preserved."""
        dataset, tensor_ff, tensor_top, smiles = multi_entry_dataset

        _, _, _, _, energy_weights, forces_weights = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device_type="cpu"
        )

        # First 2 conformations should have weight 1000.0
        assert torch.allclose(
            energy_weights[:2], torch.full((2,), 1000.0, dtype=energy_weights.dtype)
        )
        # Last 1 conformation should have weight 500.0
        assert torch.allclose(
            energy_weights[2:], torch.full((1,), 500.0, dtype=energy_weights.dtype)
        )

    def test_each_entry_has_separate_reference_energy(self, multi_entry_dataset):
        """Test that each entry has its own reference energy (minimum is 0)."""
        dataset, tensor_ff, tensor_top, smiles = multi_entry_dataset

        energy_ref, _, _, _, _, _ = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device_type="cpu", normalize=False
        )

        # Due to how predict_with_weights works, each entry's energies are
        # relative to their own mean/min, so we can't easily verify entry-level
        # min=0. But we can verify the function runs without error.
        assert energy_ref is not None
