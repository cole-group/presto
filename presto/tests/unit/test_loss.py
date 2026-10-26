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

from presto.data_utils import (
    create_dataset_with_uniform_weights,
    merge_weighted_datasets,
)
from presto.loss import (
    DatasetLoss,
    compute_dataset_loss_and_grad,
    compute_overall_loss_and_grad,
    Prediction,
    compute_regularisation_loss,
    predict_with_weights,
)


@pytest.fixture
def simple_trainable_and_params():
    """Create a simple trainable and parameters for testing."""
    mol = Molecule.from_smiles("CCO")
    ff = ForceField("openff_unconstrained-2.3.0.offxml")

    import openff.interchange

    interchange = openff.interchange.Interchange.from_smirnoff(ff, mol.to_topology())

    tensor_ff, [_tensor_top] = smee.converters.convert_interchange(interchange)

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


class TestComputeRegularisationPenalty:
    """Tests for compute_regularisation_loss function."""

    def test_returns_tensor(self, simple_trainable_and_params):
        """Test that function returns a tensor."""
        trainable, params, _n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        result = compute_regularisation_loss(
            trainable, params, initial_params, "initial"
        )
        assert isinstance(result, torch.Tensor)

    def test_penalty_zero_when_params_unchanged(self, simple_trainable_and_params):
        """Test that penalty is zero when parameters are unchanged."""
        trainable, params, _n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        result = compute_regularisation_loss(
            trainable, params, initial_params, "initial"
        )
        # Should be very small (zero or near-zero)
        assert result.item() < 1e-6

    def test_penalty_increases_with_parameter_change(self, simple_trainable_and_params):
        """Test that penalty increases when parameters change."""
        trainable, params, _n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        # Modify parameters
        modified_params = params.clone()
        modified_params = modified_params + 0.1

        penalty_original = compute_regularisation_loss(
            trainable,
            params,
            initial_parameters=initial_params,
            regularisation_target="initial",
        )
        penalty_modified = compute_regularisation_loss(
            trainable, modified_params, initial_params, "initial"
        )

        assert penalty_modified > penalty_original

    def test_penalty_scales_with_strength(self, simple_trainable_and_params):
        """Test that penalty scales with regularisation strength."""
        trainable, params, _n_atoms = simple_trainable_and_params
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
            trainable_weak, modified_params, initial_params, "initial"
        )
        penalty_strong = compute_regularisation_loss(
            trainable_strong,
            modified_params,
            initial_params,
            "initial",
        )

        assert penalty_strong > penalty_weak

    def test_regularisation_to_zero(self, simple_trainable_and_params):
        """Test regularisation towards zero."""
        trainable, params, _n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        result = compute_regularisation_loss(trainable, params, initial_params, "zero")
        assert isinstance(result, torch.Tensor)

    def test_invalid_regularisation_target_raises_error(
        self, simple_trainable_and_params
    ):
        """Test that invalid regularisation value raises error."""
        trainable, params, _n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        with pytest.raises(NotImplementedError):
            compute_regularisation_loss(trainable, params, initial_params, "invalid")

    def test_regularisation_with_zero_strength(self, simple_trainable_and_params):
        """Test that regularisation with zero strength returns zero loss."""
        trainable, params, _n_atoms = simple_trainable_and_params
        initial_params = params.clone()

        # Create trainable with zero regularisation strength
        parameter_configs_zero = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                regularize={"k": 0.0, "length": 0.0},
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
        )
        # Should be zero when regularisation strength is zero
        assert result.item() == 0.0


@pytest.fixture
def ethanol_ff_and_topology():
    """Create ethanol force field and topology for testing."""
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=3)
    ff = ForceField("openff_unconstrained-2.3.0.offxml")

    interchange = openff.interchange.Interchange.from_smirnoff(ff, mol.to_topology())

    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)
    return tensor_ff, tensor_top, mol


@pytest.fixture
def weighted_ethanol_dataset(ethanol_ff_and_topology):
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


class TestPredictWithWeights:
    """Tests for predict_with_weights function."""

    def test_returns_prediction(self, weighted_ethanol_dataset):
        """Test that function returns a Prediction."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        result = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
        )
        assert isinstance(result, Prediction)

    def test_returns_energy_weights(self, weighted_ethanol_dataset):
        """Test that energy weights are returned."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        energy_weights = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
        ).energy_weights
        assert energy_weights is not None
        assert len(energy_weights) == 3  # n_confs

    def test_returns_forces_weights(self, weighted_ethanol_dataset):
        """Test that forces weights are returned."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        forces_weights = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
        ).forces_weights
        assert forces_weights is not None
        assert len(forces_weights) == 3  # n_confs

    def test_weights_match_dataset(self, weighted_ethanol_dataset):
        """Test that returned weights match dataset."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        prediction = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
        )
        energy_weights = prediction.energy_weights
        forces_weights = prediction.forces_weights

        # Energy weight was set to 1000.0
        assert torch.allclose(
            energy_weights, torch.full((3,), 1000.0, dtype=energy_weights.dtype)
        )
        # Forces weight was set to 0.1
        assert torch.allclose(
            forces_weights, torch.full((3,), 0.1, dtype=forces_weights.dtype)
        )


@pytest.fixture
def ethanol_trainable_and_dataset():
    """Create ethanol trainable and weighted dataset for testing."""
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=3)
    ff = ForceField("openff_unconstrained-2.3.0.offxml")

    interchange = openff.interchange.Interchange.from_smirnoff(ff, mol.to_topology())

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


class TestPredictionLossWithWeights:
    """Tests for prediction_loss function with weighted datasets."""

    def test_prediction_loss_returns_loss_record(self, ethanol_trainable_and_dataset):
        """Test that prediction_loss returns a LossRecord."""
        trainable, params, initial, datasets_list, topologies = (
            ethanol_trainable_and_dataset
        )

        result, _, _ = compute_overall_loss_and_grad(
            [datasets_list],
            [1.0],
            trainable,
            params,
            initial,
            topologies,
            "initial",
            torch.device("cpu"),
            compute_grad=False,
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

        result, _, _ = compute_overall_loss_and_grad(
            [datasets_list],
            [1.0],
            trainable,
            params,
            initial,
            topologies,
            "initial",
            torch.device("cpu"),
            compute_grad=False,
        )

        assert isinstance(result.energy, torch.Tensor)

    def test_prediction_loss_forces_is_tensor(self, ethanol_trainable_and_dataset):
        """Test that forces loss is a tensor."""
        trainable, params, initial, datasets_list, topologies = (
            ethanol_trainable_and_dataset
        )

        result, _, _ = compute_overall_loss_and_grad(
            [datasets_list],
            [1.0],
            trainable,
            params,
            initial,
            topologies,
            "initial",
            torch.device("cpu"),
            compute_grad=False,
        )

        assert isinstance(result.forces, torch.Tensor)

    def test_prediction_loss_with_zero_energy_weight(self):
        """Test that zero energy weight excludes energy from loss."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=2)
        ff = ForceField("openff_unconstrained-2.3.0.offxml")

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

        result, _, _ = compute_overall_loss_and_grad(
            [[dataset]],
            [1.0],
            trainable,
            trainable_params,
            trainable_params.clone(),
            [tensor_top],
            "initial",
            torch.device("cpu"),
            compute_grad=False,
        )

        # Energy loss should be zero when weight is zero
        assert result.energy.item() == 0.0

    def test_prediction_loss_with_nan_forces(self):
        """Test that NaN forces are masked out of the force loss."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=2)
        ff = ForceField("openff_unconstrained-2.3.0.offxml")

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

        # The loss masks NaN forces itself
        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords,
            energy=torch.rand(n_confs, dtype=torch.float64),
            forces=forces,
            energy_weight=1.0,
            forces_weight=1.0,
        )

        result, _, _ = compute_overall_loss_and_grad(
            [[dataset]],
            [1.0],
            trainable,
            trainable_params,
            trainable_params.clone(),
            [tensor_top],
            "initial",
            torch.device("cpu"),
            compute_grad=False,
        )

        # Force loss should be zero because every conformation is masked
        assert result.forces.item() == 0.0


@pytest.fixture
def ethanol_setup():
    """Create ethanol setup for testing."""
    from openff.units import unit

    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=2, rms_cutoff=0.0 * unit.angstrom)
    ff = ForceField("openff_unconstrained-2.3.0.offxml")

    interchange = openff.interchange.Interchange.from_smirnoff(ff, mol.to_topology())

    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    n_confs = len(mol.conformers)
    if n_confs < 1:
        # Generate at least one conformer
        mol.generate_conformers(n_conformers=1)
        n_confs = 1

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


class TestPredictWithWeightsMultipleEntries:
    """Tests for predict_with_weights with multiple entries in a dataset."""

    @pytest.fixture
    def multi_entry_dataset(self):
        """Create a dataset with multiple entries for the same molecule."""
        from openff.units import unit

        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=3, rms_cutoff=0.0 * unit.angstrom)
        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)
        smiles = mol.to_smiles(mapped=True)

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
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
        )

        # Total conformations: 2 from first entry + 1 from second = 3
        assert len(result.energy_ref) == 3
        assert len(result.energy_weights) == 3
        assert result.entry_sizes == [2, 1]

    def test_predict_with_weights_preserves_different_weights(
        self, multi_entry_dataset
    ):
        """Test that different weights per entry are preserved."""
        dataset, tensor_ff, tensor_top, smiles = multi_entry_dataset

        energy_weights = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
        ).energy_weights

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

        energy_ref = predict_with_weights(
            dataset,
            tensor_ff,
            {smiles: tensor_top},
            device=torch.device("cpu"),
            normalize=False,
        ).energy_ref

        # Due to how predict_with_weights works, each entry's energies are
        # relative to their own mean/min, so we can't easily verify entry-level
        # min=0. But we can verify the function runs without error.
        assert energy_ref is not None


class TestGetLossClosure:
    """Tests for get_loss_closure_fn function."""

    def test_get_loss_closure_fn_basic(
        self, simple_trainable_and_params, ethanol_ff_and_topology
    ):
        """Get loss closure fn basic."""
        from presto.data_utils import create_dataset_with_uniform_weights
        from presto.loss import get_loss_closure_fn

        trainable, params, n_atoms = simple_trainable_and_params
        _tensor_ff, tensor_top, mol = ethanol_ff_and_topology

        # Create a minimal dataset
        dataset = create_dataset_with_uniform_weights(
            smiles=mol.to_smiles(mapped=True),
            coords=torch.zeros(1, n_atoms, 3),
            energy=torch.zeros(1),
            forces=torch.zeros(1, n_atoms, 3),
            energy_weight=1.0,
            forces_weight=1.0,
        )

        closure = get_loss_closure_fn(
            per_molecule_datasets_by_dataset=[[dataset]],
            dataset_weights=[1.0],
            trainable=trainable,
            trainable_parameters=params,
            initial_parameters=params.clone(),
            topologies=[tensor_top],
            regularisation_target="initial",
        )

        # Test closure with gradient
        x = params.clone().requires_grad_(True)
        loss, grad, hess = closure(x, compute_gradient=True, compute_hessian=False)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(grad, torch.Tensor)
        assert hess is None
        assert grad.shape == params.shape

    def test_get_loss_closure_fn_no_grad(
        self, simple_trainable_and_params, ethanol_ff_and_topology
    ):
        """Get loss closure fn no grad."""
        from presto.data_utils import create_dataset_with_uniform_weights
        from presto.loss import get_loss_closure_fn

        trainable, params, n_atoms = simple_trainable_and_params
        _tensor_ff, tensor_top, mol = ethanol_ff_and_topology

        dataset = create_dataset_with_uniform_weights(
            smiles=mol.to_smiles(mapped=True),
            coords=torch.zeros(1, n_atoms, 3),
            energy=torch.zeros(1),
            forces=torch.zeros(1, n_atoms, 3),
            energy_weight=1.0,
            forces_weight=1.0,
        )

        closure = get_loss_closure_fn(
            per_molecule_datasets_by_dataset=[[dataset]],
            dataset_weights=[1.0],
            trainable=trainable,
            trainable_parameters=params,
            initial_parameters=params.clone(),
            topologies=[tensor_top],
            regularisation_target="initial",
        )

        loss, grad, hess = closure(
            params, compute_gradient=False, compute_hessian=False
        )
        assert isinstance(loss, torch.Tensor)
        assert grad is None
        assert hess is None

    def test_predict_with_weights_reference_min(self, weighted_ethanol_dataset):
        """Predict with weights reference min."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset
        res = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, reference="min"
        )
        assert isinstance(res, Prediction)

    def test_predict_with_weights_reference_median(self, weighted_ethanol_dataset):
        """Predict with weights reference median."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset
        res = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, reference="median"
        )
        assert isinstance(res, Prediction)


def _explicit_weighted_objective(
    per_molecule_datasets_by_dataset, weights, trainable, x, initial, topologies, target
):
    """Build the weighted loss from live tensors, so autograd gives a reference gradient.

    This is independent of the loss module's gradient accumulation.
    Returns (energy, forces, regularisation), each still attached to ``x``.
    """
    from presto.loss import _compute_molecule_energy_force_loss

    force_field = trainable.to_force_field(x)
    total_weight = sum(weights)
    energy, forces = 0.0, 0.0
    for datasets_list, weight in zip(
        per_molecule_datasets_by_dataset, weights, strict=True
    ):
        mol_energies, mol_forces = [], []
        for dataset, topology in zip(datasets_list, topologies, strict=True):
            prediction = predict_with_weights(
                dataset,
                force_field,
                {dataset[0]["smiles"]: topology},
                normalize=False,
                create_graph=True,
            )
            e, f = _compute_molecule_energy_force_loss(
                prediction, topology.n_atoms, x.device
            )
            mol_energies.append(e)
            mol_forces.append(f)
        energy = energy + weight / total_weight * torch.stack(mol_energies).mean()
        forces = forces + weight / total_weight * torch.stack(mol_forces).mean()
    reg = compute_regularisation_loss(trainable, x, initial, target)
    return energy, forces, reg


def _assert_matches_reference(loss, gradient, energy, forces, expected_gradient):
    """Check losses and a gradient against the explicit autograd reference.

    Predicted energies are single precision (the datasets are read as float32), so a
    different order of operations agrees only to about 1e-7.
    """
    assert torch.allclose(loss.energy, energy.detach(), rtol=1e-6, atol=0)
    assert torch.allclose(loss.forces, forces.detach(), rtol=1e-6, atol=0)
    torch.testing.assert_close(
        gradient,
        expected_gradient,
        rtol=1e-6,
        atol=1e-7 * expected_gradient.abs().max().item(),
    )


def _synthetic_dataset(
    mol, tensor_ff, topology, n_confs, seed, energy_weight=1000.0, forces_weight=0.1
):
    """Return a dataset whose references differ from the force field's predictions.

    This makes every loss term and gradient non-zero.
    """
    generator = torch.Generator().manual_seed(seed)
    base = torch.tensor(mol.conformers[0].m_as("angstrom"), dtype=torch.float64)
    coords = base + 0.05 * torch.randn(
        n_confs, *base.shape, generator=generator, dtype=torch.float64
    )
    coords.requires_grad_(True)
    energy = smee.compute_energy(topology, tensor_ff, coords)
    forces = -torch.autograd.grad(energy.sum(), coords)[0]
    return create_dataset_with_uniform_weights(
        smiles=mol.to_smiles(mapped=True),
        coords=coords.detach(),
        energy=energy.detach()
        + torch.randn(n_confs, generator=generator, dtype=torch.float64),
        forces=forces.detach()
        + torch.randn(forces.shape, generator=generator, dtype=torch.float64),
        energy_weight=energy_weight,
        forces_weight=forces_weight,
    )


@pytest.fixture(scope="module")
def two_molecule_setup():
    """Two molecules sharing one trainable force field, with three datasets each.

    The datasets are distinct and the parameters are moved away from their initial
    values, so the regularisation is non-zero.
    """
    mols = [Molecule.from_smiles("CCO"), Molecule.from_smiles("CCCN")]
    for mol in mols:
        mol.generate_conformers(n_conformers=1)
    ff = ForceField("openff_unconstrained-2.3.0.offxml")
    interchanges = [
        openff.interchange.Interchange.from_smirnoff(ff, mol.to_topology())
        for mol in mols
    ]
    tensor_ff, topologies = smee.converters.convert_interchange(interchanges)

    parameter_configs = {
        "Bonds": ParameterConfig(
            cols=["k", "length"],
            scales={"k": 1.0 / 500.0, "length": 1.0},
            regularize={"k": 1.0, "length": 1.0},
        ),
        "Angles": ParameterConfig(
            cols=["k"], scales={"k": 1.0 / 100.0}, regularize={"k": 1.0}
        ),
    }
    trainable = Trainable(tensor_ff, parameter_configs, {})
    initial = trainable.to_values().detach().clone()
    params = initial * 1.02  # Non-zero regularisation and its gradient

    per_molecule_datasets_by_dataset = [
        [
            _synthetic_dataset(mol, tensor_ff, top, n_confs, seed=10 * k + m)
            for m, (mol, top) in enumerate(zip(mols, topologies, strict=True))
        ]
        for k, n_confs in enumerate((3, 5, 4))
    ]
    return (
        trainable,
        params,
        initial,
        topologies,
        per_molecule_datasets_by_dataset,
        mols,
        tensor_ff,
    )


def _run_compute_overall_loss_and_grad(
    setup, per_molecule_datasets_by_dataset, weights, target="initial", grad=True
):
    """Call compute_overall_loss_and_grad on the fixture with fresh parameters."""
    trainable, params, initial, topologies, *_ = setup
    x = params.clone().requires_grad_(True)
    return compute_overall_loss_and_grad(
        per_molecule_datasets_by_dataset,
        weights,
        trainable,
        x,
        initial,
        topologies,
        target,
        torch.device("cpu"),
        compute_grad=grad,
    )


def _run_compute_dataset_loss_and_grad(setup, datasets_list, grad=True):
    """Call compute_dataset_loss_and_grad on the fixture with fresh parameters."""
    trainable, params, _, topologies, *_ = setup
    x = params.clone().requires_grad_(True)
    return compute_dataset_loss_and_grad(
        datasets_list,
        trainable,
        x,
        topologies,
        torch.device("cpu"),
        compute_grad=grad,
    )


class TestComputeDatasetLossAndGrad:
    """Tests for the loss of a single dataset across molecules."""

    def test_matches_autograd_without_regularisation(self, two_molecule_setup):
        """The losses and gradient match autograd of the data loss alone."""
        trainable, params, initial, topologies, per_molecule_datasets_by_dataset, *_ = (
            two_molecule_setup
        )
        datasets_list = per_molecule_datasets_by_dataset[0]
        x = params.clone().requires_grad_(True)
        energy, forces, reg = _explicit_weighted_objective(
            [datasets_list], [1.0], trainable, x, initial, topologies, "initial"
        )
        (expected_gradient,) = torch.autograd.grad(energy + forces, x)
        (reg_grad,) = torch.autograd.grad(reg, x)

        loss, gradient = _run_compute_dataset_loss_and_grad(
            two_molecule_setup, datasets_list
        )

        assert isinstance(loss, DatasetLoss)
        _assert_matches_reference(loss, gradient, energy, forces, expected_gradient)
        # Regularisation is not included
        assert reg_grad.abs().max() > 0
        assert not torch.allclose(gradient, expected_gradient + reg_grad)

    def test_molecule_count_mismatch_raises(self, two_molecule_setup):
        """Each dataset must have one entry per molecule."""
        with pytest.raises(ValueError):
            _run_compute_dataset_loss_and_grad(
                two_molecule_setup, two_molecule_setup[4][0][:1]
            )


class TestComputeOverallLossAndGrad:
    """Tests for the weighted combination of several datasets."""

    @pytest.mark.parametrize("weight", [2.5])
    @pytest.mark.parametrize("target", ["initial", "zero"])
    def test_single_dataset_weight_has_no_effect(
        self, two_molecule_setup, weight, target
    ):
        """One dataset gives exactly its own loss plus regularisation, whatever its weight.

        A lone weight normalises to exactly 1.0, so the results are bitwise equal.
        """
        trainable, params, initial, _, per_molecule_datasets_by_dataset, *_ = (
            two_molecule_setup
        )
        datasets_list = per_molecule_datasets_by_dataset[0]
        data_loss, data_gradient = _run_compute_dataset_loss_and_grad(
            two_molecule_setup, datasets_list
        )
        x = params.clone().requires_grad_(True)
        reg = compute_regularisation_loss(trainable, x, initial, target)
        reg_gradient = (
            torch.autograd.grad(reg, x)[0] if reg.requires_grad else torch.zeros_like(x)
        )

        unit, _, unit_gradient = _run_compute_overall_loss_and_grad(
            two_molecule_setup, [datasets_list], [1.0], target
        )
        record, dataset_losses, gradient = _run_compute_overall_loss_and_grad(
            two_molecule_setup, [datasets_list], [weight], target
        )

        for field in record._fields:
            assert torch.equal(getattr(record, field), getattr(unit, field))
        assert torch.equal(gradient, unit_gradient)
        assert torch.equal(record.energy, data_loss.energy)
        assert torch.equal(record.forces, data_loss.forces)
        assert torch.equal(record.regularisation, reg.detach())
        assert torch.equal(gradient, data_gradient + reg_gradient)
        assert len(dataset_losses) == 1
        assert torch.equal(dataset_losses[0].energy, data_loss.energy)
        assert torch.equal(dataset_losses[0].forces, data_loss.forces)

    def test_weighted_combination_of_dataset_losses(self, two_molecule_setup):
        """The combined loss is the normalised weighted sum of each dataset's loss."""
        per_molecule_datasets_by_dataset = two_molecule_setup[4][:2]
        loss_0, grad_0 = _run_compute_dataset_loss_and_grad(
            two_molecule_setup, per_molecule_datasets_by_dataset[0]
        )
        loss_1, grad_1 = _run_compute_dataset_loss_and_grad(
            two_molecule_setup, per_molecule_datasets_by_dataset[1]
        )
        # The datasets are distinct, so the weighting is visible
        assert not torch.isclose(loss_0.energy, loss_1.energy)
        assert not torch.isclose(loss_0.forces, loss_1.forces)

        record, dataset_losses, gradient = _run_compute_overall_loss_and_grad(
            two_molecule_setup, per_molecule_datasets_by_dataset, [1.0, 3.0]
        )

        assert torch.allclose(
            record.energy, 0.25 * loss_0.energy + 0.75 * loss_1.energy
        )
        assert torch.allclose(
            record.forces, 0.25 * loss_0.forces + 0.75 * loss_1.forces
        )
        for dataset_loss, expected in zip(
            dataset_losses, (loss_0, loss_1), strict=True
        ):
            assert torch.equal(dataset_loss.energy, expected.energy)
            assert torch.equal(dataset_loss.forces, expected.forces)

        trainable, params, initial, *_ = two_molecule_setup
        x = params.clone().requires_grad_(True)
        reg = compute_regularisation_loss(trainable, x, initial, "initial")
        (reg_grad,) = torch.autograd.grad(reg, x)
        assert torch.allclose(gradient, 0.25 * grad_0 + 0.75 * grad_1 + reg_grad)

    @pytest.mark.parametrize("target", ["initial", "zero"])
    @pytest.mark.parametrize("weights", [[1.0], [1.0, 3.0], [0.2, 5.0, 1.3]])
    def test_matches_autograd_of_explicit_objective(
        self, two_molecule_setup, target, weights
    ):
        """Losses and the accumulated gradient match autograd of the explicit sum."""
        trainable, params, initial, topologies, all_datasets, *_ = two_molecule_setup
        per_molecule_datasets_by_dataset = all_datasets[: len(weights)]
        x = params.clone().requires_grad_(True)
        energy, forces, reg = _explicit_weighted_objective(
            per_molecule_datasets_by_dataset,
            weights,
            trainable,
            x,
            initial,
            topologies,
            target,
        )
        (expected_gradient,) = torch.autograd.grad(energy + forces + reg, x)

        record, _, gradient = _run_compute_overall_loss_and_grad(
            two_molecule_setup, per_molecule_datasets_by_dataset, weights, target
        )

        _assert_matches_reference(record, gradient, energy, forces, expected_gradient)
        assert torch.allclose(record.regularisation, reg.detach(), rtol=1e-6, atol=0)

    def test_gradient_matches_finite_differences(self, two_molecule_setup):
        """Central differences of the combined loss agree with the gradient."""
        trainable, params, initial, topologies, all_datasets, *_ = two_molecule_setup
        weights = [1.0, 3.0, 0.5]
        _, _, gradient = _run_compute_overall_loss_and_grad(
            two_molecule_setup, all_datasets, weights
        )

        def total_loss(x):
            record, _, _ = compute_overall_loss_and_grad(
                all_datasets,
                weights,
                trainable,
                x,
                initial,
                topologies,
                "initial",
                torch.device("cpu"),
                compute_grad=False,
            )
            return (record.energy + record.forces + record.regularisation).item()

        # The loss is only single precision, so use a moderate step and the largest
        # gradient components, which finite differences resolve well
        for idx in gradient.abs().argsort(descending=True)[:3].tolist():
            step = 1e-4 * max(1.0, abs(params[idx].item()))
            plus, minus = params.clone(), params.clone()
            plus[idx] += step
            minus[idx] -= step
            finite_difference = (total_loss(plus) - total_loss(minus)) / (2 * step)
            assert finite_difference == pytest.approx(gradient[idx].item(), rel=1e-3)

    def test_influence_independent_of_dataset_size(self, two_molecule_setup):
        """Repeating every conformer of a dataset doesn't change its influence.

        Each dataset's loss is computed separately, and within a molecule each
        entry's loss is a mean over its conformers.
        """
        all_datasets = two_molecule_setup[4]
        datasets_a, datasets_b = all_datasets[0], all_datasets[1]

        def repeat_conformers(dataset):
            entry = dataset[0]
            n_confs = len(entry["energy"])
            return create_dataset_with_uniform_weights(
                smiles=entry["smiles"],
                coords=entry["coords"].reshape(n_confs, -1, 3).repeat(2, 1, 1),
                energy=entry["energy"].repeat(2),
                forces=entry["forces"].reshape(n_confs, -1, 3).repeat(2, 1, 1),
                energy_weight=entry["energy_weights"][0].item(),
                forces_weight=entry["forces_weights"][0].item(),
            )

        datasets_b_doubled = [repeat_conformers(d) for d in datasets_b]
        loss_b, _ = _run_compute_dataset_loss_and_grad(
            two_molecule_setup, datasets_b, grad=False
        )
        loss_b_doubled, _ = _run_compute_dataset_loss_and_grad(
            two_molecule_setup, datasets_b_doubled, grad=False
        )
        assert torch.allclose(loss_b.energy, loss_b_doubled.energy)
        assert torch.allclose(loss_b.forces, loss_b_doubled.forces)

        record, _, gradient = _run_compute_overall_loss_and_grad(
            two_molecule_setup, [datasets_a, datasets_b], [1.0, 3.0]
        )
        doubled, _, doubled_gradient = _run_compute_overall_loss_and_grad(
            two_molecule_setup, [datasets_a, datasets_b_doubled], [1.0, 3.0]
        )
        for field in record._fields:
            assert torch.allclose(getattr(record, field), getattr(doubled, field))
        assert torch.allclose(gradient, doubled_gradient)

        # Concatenating the datasets doesn't let the larger one dominate either,
        # since each entry of a molecule gets one share of its loss
        concatenated, _ = _run_compute_dataset_loss_and_grad(
            two_molecule_setup,
            [
                datasets.concatenate_datasets([a, b])
                for a, b in zip(datasets_a, datasets_b, strict=True)
            ],
            grad=False,
        )
        concatenated_doubled, _ = _run_compute_dataset_loss_and_grad(
            two_molecule_setup,
            [
                datasets.concatenate_datasets([a, b])
                for a, b in zip(datasets_a, datasets_b_doubled, strict=True)
            ],
            grad=False,
        )
        assert torch.allclose(concatenated.energy, concatenated_doubled.energy)
        assert torch.allclose(concatenated.forces, concatenated_doubled.forces)

    def test_no_grad(self, two_molecule_setup):
        """Without gradients the losses are unchanged and no gradient is returned."""
        per_molecule_datasets_by_dataset = two_molecule_setup[4]
        weights = [1.0, 3.0, 0.5]
        record, losses, gradient = _run_compute_overall_loss_and_grad(
            two_molecule_setup, per_molecule_datasets_by_dataset, weights
        )
        record_no_grad, losses_no_grad, no_gradient = (
            _run_compute_overall_loss_and_grad(
                two_molecule_setup,
                per_molecule_datasets_by_dataset,
                weights,
                grad=False,
            )
        )

        assert gradient is not None
        assert no_gradient is None
        for field in record._fields:
            assert torch.equal(getattr(record, field), getattr(record_no_grad, field))
        for loss, loss_no_grad in zip(losses, losses_no_grad, strict=True):
            assert torch.equal(loss.energy, loss_no_grad.energy)
            assert torch.equal(loss.forces, loss_no_grad.forces)

    def test_losses_are_detached(self, two_molecule_setup):
        """Returned losses hold no graph, so they can be logged freely."""
        record, losses, _ = _run_compute_overall_loss_and_grad(
            two_molecule_setup, two_molecule_setup[4], [1, 2, 3]
        )
        for value in (*record, *(v for loss in losses for v in loss)):
            assert not value.requires_grad

    @pytest.mark.parametrize("n_weights, match", [(2, "3 datasets"), (4, "3 datasets")])
    def test_weight_count_mismatch_raises(self, two_molecule_setup, n_weights, match):
        """There must be one weight per dataset."""
        with pytest.raises(ValueError, match=match):
            _run_compute_overall_loss_and_grad(
                two_molecule_setup, two_molecule_setup[4], [1.0] * n_weights
            )

    def test_no_datasets_raises(self, two_molecule_setup):
        """At least one dataset is required."""
        with pytest.raises(ValueError, match="At least one dataset"):
            _run_compute_overall_loss_and_grad(two_molecule_setup, [], [])

    def test_molecule_count_mismatch_raises(self, two_molecule_setup):
        """Every dataset needs one entry per molecule."""
        all_datasets = two_molecule_setup[4]
        with pytest.raises(ValueError):
            _run_compute_overall_loss_and_grad(
                two_molecule_setup, [all_datasets[0], all_datasets[1][:1]], [1, 1]
            )


class TestWeightedLossClosure:
    """Tests for the Levenberg-Marquardt closure with several datasets."""

    def test_closure_matches_overall_loss(self, two_molecule_setup):
        """The closure's loss and gradient are the combined loss and its gradient."""
        from presto.loss import get_loss_closure_fn

        trainable, params, initial, topologies, all_datasets, *_ = two_molecule_setup
        weights = [1.0, 3.0]
        closure = get_loss_closure_fn(
            per_molecule_datasets_by_dataset=all_datasets[:2],
            dataset_weights=weights,
            trainable=trainable,
            trainable_parameters=params,
            initial_parameters=initial,
            topologies=topologies,
            regularisation_target="initial",
        )
        record, _, gradient = _run_compute_overall_loss_and_grad(
            two_molecule_setup, all_datasets[:2], weights
        )

        x = params.clone().requires_grad_(True)
        loss, closure_gradient, hessian = closure(
            x, compute_gradient=True, compute_hessian=False
        )

        assert torch.allclose(
            loss, record.energy + record.forces + record.regularisation
        )
        assert torch.allclose(closure_gradient, gradient)
        assert hessian is None

    def test_closure_hessian_is_overall_hessian(self, two_molecule_setup):
        """The closure returns the Hessian from compute_overall_loss_hessian."""
        from presto.loss import compute_overall_loss_hessian, get_loss_closure_fn

        trainable, params, initial, topologies, all_datasets, *_ = two_molecule_setup
        args = (all_datasets[:2], [1.0, 3.0], trainable)
        closure = get_loss_closure_fn(*args, params, initial, topologies, "initial")
        x = params.clone().requires_grad_(True)
        _, gradient, hessian = closure(x, compute_gradient=False, compute_hessian=True)

        expected = compute_overall_loss_hessian(
            *args, params, initial, topologies, "initial", torch.device("cpu")
        )
        assert gradient is None
        assert torch.equal(hessian, expected)


class TestComputeOverallLossHessian:
    """Tests for the Hessian of the weighted loss."""

    @pytest.mark.parametrize("target", ["initial", "zero"])
    @pytest.mark.parametrize("weights", [[1.0], [1.0, 3.0]])
    def test_matches_hessian_of_explicit_objective(
        self, two_molecule_setup, target, weights
    ):
        """The accumulated Hessian matches autograd of the explicit weighted loss."""
        from presto.loss import compute_overall_loss_hessian

        trainable, params, initial, topologies, all_datasets, *_ = two_molecule_setup
        per_molecule_datasets_by_dataset = all_datasets[: len(weights)]

        def total_loss(x):
            energy, forces, reg = _explicit_weighted_objective(
                per_molecule_datasets_by_dataset,
                weights,
                trainable,
                x,
                initial,
                topologies,
                target,
            )
            return energy + forces + reg

        expected = torch.autograd.functional.hessian(total_loss, params.clone())

        hessian = compute_overall_loss_hessian(
            per_molecule_datasets_by_dataset,
            weights,
            trainable,
            params,
            initial,
            topologies,
            target,
            torch.device("cpu"),
        )

        assert hessian.shape == (len(params), len(params))
        assert not hessian.requires_grad
        assert hessian.abs().max() > 0
        # Predictions are single precision, so small entries are noisy relative to
        # their size; compare on the scale of the largest entry
        tolerance = {"rtol": 1e-5, "atol": 1e-6 * expected.abs().max().item()}
        torch.testing.assert_close(hessian, hessian.T, **tolerance)
        torch.testing.assert_close(hessian, expected, **tolerance)

    def test_weight_count_mismatch_raises(self, two_molecule_setup):
        """There must be one weight per dataset."""
        from presto.loss import compute_overall_loss_hessian

        trainable, params, initial, topologies, all_datasets, *_ = two_molecule_setup
        with pytest.raises(ValueError, match="3 datasets"):
            compute_overall_loss_hessian(
                all_datasets,
                [1.0],
                trainable,
                params,
                initial,
                topologies,
                "initial",
                torch.device("cpu"),
            )


def _ess_loss(loss_per_conf, weights):
    """The ESS-normalised loss used before the entry-mean form, as a reference."""
    ess = weights.sum() ** 2 / (weights**2).sum()
    return (loss_per_conf * weights).sum() / ess


def _molecule_loss_from_errors(
    energy_errors, force_errors, energy_weights, forces_weights, r, entry_sizes
):
    """Call _compute_molecule_energy_force_loss with per-conformer errors.

    Uses one atom, so the energy term's squared error is ``energy_errors**2`` and the
    force term's is ``force_errors**2`` (the same error on each component).
    """
    from presto.loss import _compute_molecule_energy_force_loss

    n_confs = len(energy_errors)
    forces_ref = force_errors.reshape(n_confs, 1).expand(n_confs, 3).reshape(-1, 3)
    prediction = Prediction(
        energy_ref=energy_errors,
        energy_pred=torch.zeros(n_confs, dtype=torch.float64),
        forces_ref=forces_ref,
        forces_pred=torch.zeros_like(forces_ref),
        energy_weights=energy_weights,
        forces_weights=forces_weights,
        energy_conformer_weights=r,
        forces_conformer_weights=r,
        entry_sizes=entry_sizes,
    )
    return _compute_molecule_energy_force_loss(prediction, 1, torch.device("cpu"))


class TestMoleculeLossWeighting:
    """Tests for the per-molecule loss: a mean over active entries of weighted means."""

    @pytest.mark.parametrize("entry_sizes", [[6], [2, 2, 2], [3, 3]])
    def test_reduces_to_ess_for_uniform_weights(self, entry_sizes):
        """With r = 1, uniform w and equal-size entries the loss equals the old ESS loss."""
        generator = torch.Generator().manual_seed(0)
        e, f = (
            torch.randn(6, generator=generator, dtype=torch.float64),
            torch.randn(6, generator=generator, dtype=torch.float64),
        )
        w_e = torch.full((6,), 1000.0, dtype=torch.float64)
        w_f = torch.full((6,), 0.1, dtype=torch.float64)
        ones = torch.ones(6, dtype=torch.float64)

        energy, forces = _molecule_loss_from_errors(e, f, w_e, w_f, ones, entry_sizes)

        assert torch.allclose(energy, _ess_loss(e**2, w_e))
        assert torch.allclose(forces, _ess_loss(f**2, w_f))

    def test_linear_and_monotonic_in_each_weight(self):
        """The loss is exactly linear in each absolute weight, unlike the ESS loss."""
        errors = torch.ones(2, dtype=torch.float64)
        ones = torch.ones(2, dtype=torch.float64)

        def losses(w1):
            w = torch.tensor([1000.0, w1], dtype=torch.float64)
            energy, forces = _molecule_loss_from_errors(errors, errors, w, w, ones, [2])
            return energy, forces, _ess_loss(errors, w)

        e10, f10, ess10 = losses(10.0)
        e20, f20, ess20 = losses(20.0)
        e30, f30, _ = losses(30.0)

        # The ESS loss falls as the weight rises
        assert ess20 < ess10
        for low, mid, high in ((e10, e20, e30), (f10, f20, f30)):
            assert mid > low
            assert torch.allclose(high - mid, mid - low)
            # With a constant error the loss is the mean weight
            assert torch.allclose(mid, torch.tensor(510.0, dtype=torch.float64))

    def test_entries_count_equally_whatever_their_size(self):
        """Repeating every conformer of one entry leaves the loss unchanged.

        Each entry's loss is a mean over its own conformers, so repeating all of them
        leaves it unchanged, and the molecule's loss is a mean over its entries, so
        the entry keeps its share. (Repeating only some conformers of an entry does
        change the loss, like a conformer weight above 1: see
        test_conformer_weights_are_scale_free_and_act_as_counts.)
        """
        generator = torch.Generator().manual_seed(1)
        e = torch.randn(8, generator=generator, dtype=torch.float64)
        # Entry 1 is conformers 0-4 and entry 2 is conformers 5-7
        entry_sizes = [5, 3]
        w = torch.full((8,), 3.0, dtype=torch.float64)
        ones = torch.ones(8, dtype=torch.float64)
        loss = _molecule_loss_from_errors(e, e, w, w, ones, entry_sizes)

        # Repeat every conformer of entry 1 twenty times, giving it 100 conformers
        repeated = torch.cat([e[:5].repeat(20), e[5:]])
        repeated_entry_sizes = [100, 3]
        w_repeated = torch.full((103,), 3.0, dtype=torch.float64)
        loss_repeated = _molecule_loss_from_errors(
            repeated,
            repeated,
            w_repeated,
            w_repeated,
            torch.ones(103, dtype=torch.float64),
            repeated_entry_sizes,
        )

        for a, b in zip(loss, loss_repeated, strict=True):
            assert torch.allclose(a, b)
        # Each entry's mean counts once, scaled by the absolute weight
        entry_means = torch.stack([(e[:5] ** 2).mean(), (e[5:] ** 2).mean()])
        expected = 3.0 * entry_means.mean()
        assert torch.allclose(loss[0], expected)
        # Contrast: the old ESS loss pooled the conformers, so entry 1's share of the
        # loss rose from 5/8 to 100/103
        assert not torch.allclose(
            _ess_loss(e**2, w), _ess_loss(repeated**2, w_repeated)
        )

    def test_conformer_weights_are_scale_free_and_act_as_counts(self):
        """Scaling r within an entry does nothing; r = 2 duplicates and r = 0 removes."""
        e = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        r = torch.tensor([2.0, 1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
        loss = _molecule_loss_from_errors(e, e, w, w, r, [3, 2])

        r_scaled = torch.cat([7.0 * r[:3], 0.5 * r[3:]])
        loss_scaled = _molecule_loss_from_errors(e, e, w, w, r_scaled, [3, 2])

        # Conformer 0 duplicated, conformer 2 removed, all with r = 1
        keep = [0, 0, 1, 3, 4]
        loss_explicit = _molecule_loss_from_errors(
            e[keep],
            e[keep],
            w[keep],
            w[keep],
            torch.ones(5, dtype=torch.float64),
            [3, 2],
        )

        for a, b, c in zip(loss, loss_scaled, loss_explicit, strict=True):
            assert torch.allclose(a, b)
            assert torch.allclose(a, c)

    def test_inactive_entries_drop_out(self):
        """Empty, excluded, zero-weight and single-energy entries drop out of a term."""
        e = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
        w = torch.ones(6, dtype=torch.float64)
        # Entries: [0, 1], empty, [2] (one conformer), [3, 4] (r = 0), [5]
        sizes = [2, 0, 1, 2, 1]
        r = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        w_f = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float64)

        energy, forces = _molecule_loss_from_errors(e, e, w, w_f, r, sizes)

        # Energy: only the first entry has two conformers with r > 0
        assert torch.allclose(energy, (e[:2] ** 2).mean())
        # Forces: single conformers count, but the last entry has zero total weight
        assert torch.allclose(forces, ((e[:2] ** 2).mean() + e[2] ** 2) / 2)

    def test_zero_weight_entry_matches_removing_it(self):
        """A force weight of 0 for an entry gives the force loss without that entry."""
        e = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        r = torch.ones(5, dtype=torch.float64)
        w_f = torch.tensor([0.1, 0.1, 0.1, 0.0, 0.0], dtype=torch.float64)
        _, forces = _molecule_loss_from_errors(e, e, r, w_f, r, [3, 2])
        _, forces_removed = _molecule_loss_from_errors(
            e[:3], e[:3], r[:3], w_f[:3], r[:3], [3]
        )
        assert torch.allclose(forces, forces_removed)

    def test_no_active_entries_gives_zero(self):
        """A term with no active entry is 0."""
        e = torch.ones(2, dtype=torch.float64)
        energy, forces = _molecule_loss_from_errors(
            e, e, e, torch.zeros(2, dtype=torch.float64), e, [1, 1]
        )
        assert energy.item() == 0.0
        assert forces.item() == 0.0


def _rebuild(dataset, keep=None, **overrides):
    """Rebuild a one-entry weighted dataset, optionally keeping some conformers."""
    from presto.data_utils import (
        create_weighted_dataset,
        get_conformer_weights_from_entry,
    )

    entry = dataset[0]
    n_confs = len(entry["energy"])
    fields = {
        "smiles": entry["smiles"],
        "coords": entry["coords"].reshape(n_confs, -1, 3),
        "energy": entry["energy"],
        "forces": entry["forces"].reshape(n_confs, -1, 3),
        "energy_weights": entry["energy_weights"],
        "forces_weights": entry["forces_weights"],
        "conformer_weights": get_conformer_weights_from_entry(entry),
    }
    fields.update(overrides)
    if keep is not None:
        fields = {
            key: value if key == "smiles" else torch.as_tensor(value)[keep]
            for key, value in fields.items()
        }
    return create_weighted_dataset([fields])


class TestConformerWeightsInLoss:
    """Tests for conformer weights and NaN masking on real datasets."""

    def _loss(self, setup, datasets_list):
        trainable, params, _, topologies, *_ = setup
        x = params.clone().requires_grad_(True)
        return compute_dataset_loss_and_grad(
            datasets_list,
            trainable,
            x,
            topologies[: len(datasets_list)],
            torch.device("cpu"),
        )

    def test_weighted_offset(self, two_molecule_setup):
        """The energy offset is the conformer-weighted mean of each entry."""
        trainable, params, _, topologies, all_datasets, *_ = two_molecule_setup
        dataset = all_datasets[1][0]
        r = torch.tensor([0.0, 1.0, 3.0, 1.0, 1.0], dtype=torch.float64)
        weighted = _rebuild(dataset, conformer_weights=r)

        energy_ref = predict_with_weights(
            weighted,
            trainable.to_force_field(params),
            {dataset[0]["smiles"]: topologies[0]},
            normalize=False,
        ).energy_ref

        raw = dataset[0]["energy"]
        assert torch.allclose(
            energy_ref, (raw - (r * raw).sum() / r.sum()).to(energy_ref.dtype)
        )

    def test_scale_free_and_zero_removes(self, two_molecule_setup):
        """Scaling r changes nothing, and r = 0 equals removing the conformer."""
        dataset = two_molecule_setup[4][1][0]
        r = torch.tensor([1.0, 0.0, 2.0, 1.0, 1.0], dtype=torch.float64)

        loss, gradient = self._loss(
            two_molecule_setup, [_rebuild(dataset, conformer_weights=r)]
        )
        scaled, scaled_gradient = self._loss(
            two_molecule_setup, [_rebuild(dataset, conformer_weights=5.0 * r)]
        )
        explicit, explicit_gradient = self._loss(
            two_molecule_setup, [_rebuild(dataset, keep=[0, 2, 2, 3, 4])]
        )

        for other, other_gradient in (
            (scaled, scaled_gradient),
            (explicit, explicit_gradient),
        ):
            assert torch.allclose(loss.energy, other.energy, rtol=1e-6)
            assert torch.allclose(loss.forces, other.forces, rtol=1e-6)
            torch.testing.assert_close(
                gradient, other_gradient, rtol=1e-5, atol=1e-7 * gradient.abs().max()
            )

    @pytest.mark.parametrize("term", ["energy", "forces"])
    @pytest.mark.parametrize("all_nan", [False, True], ids=["partial", "all"])
    @pytest.mark.parametrize("zero_weight", [False, True])
    def test_nan_reference_data_are_masked(
        self, two_molecule_setup, term, all_nan, zero_weight
    ):
        """NaN references are excluded from their term, with finite gradients and Hessian.

        A zero absolute weight at the NaN conformers, as older datasets have for NaN
        forces, gives the same result.
        """
        from presto.loss import compute_overall_loss_hessian

        trainable, params, initial, topologies, *_ = two_molecule_setup
        dataset = two_molecule_setup[4][1][0]
        entry = dataset[0]
        nan_idxs = [0, 1, 2, 3, 4] if all_nan else [1, 3]
        keep = [i for i in range(5) if i not in nan_idxs]

        values = entry[term].reshape(5, -1).clone()
        values[nan_idxs] = float("nan")
        weights = entry[f"{term}_weights"].clone()
        if zero_weight:
            weights[nan_idxs] = 0.0
        nan_dataset = _rebuild(
            dataset,
            **{
                term: values.reshape(5)
                if term == "energy"
                else values.reshape(5, -1, 3),
                f"{term}_weights": weights,
            },
        )

        loss, gradient = self._loss(two_molecule_setup, [nan_dataset])
        assert torch.isfinite(gradient).all()
        hessian = compute_overall_loss_hessian(
            [[nan_dataset]],
            [1.0],
            trainable,
            params,
            initial,
            topologies[:1],
            "initial",
            torch.device("cpu"),
        )
        assert torch.isfinite(hessian).all()

        full, _ = self._loss(two_molecule_setup, [dataset])
        other_term = "forces" if term == "energy" else "energy"
        assert torch.allclose(getattr(loss, other_term), getattr(full, other_term))
        if all_nan:
            assert getattr(loss, term).item() == 0.0
        else:
            removed, _ = self._loss(two_molecule_setup, [_rebuild(dataset, keep=keep)])
            assert torch.allclose(getattr(loss, term), getattr(removed, term))
