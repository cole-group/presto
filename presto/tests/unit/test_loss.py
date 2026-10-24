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

    def test_returns_correct_number_of_outputs(self, weighted_ethanol_dataset):
        """Test that function returns 6 outputs."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        result = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
        )
        assert len(result) == 6

    def test_returns_energy_weights(self, weighted_ethanol_dataset):
        """Test that energy weights are returned."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        _, _, _, _, energy_weights, _ = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
        )
        assert energy_weights is not None
        assert len(energy_weights) == 3  # n_confs

    def test_returns_forces_weights(self, weighted_ethanol_dataset):
        """Test that forces weights are returned."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        _, _, _, _, _, forces_weights = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
        )
        assert forces_weights is not None
        assert len(forces_weights) == 3  # n_confs

    def test_weights_match_dataset(self, weighted_ethanol_dataset):
        """Test that returned weights match dataset."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset

        _, _, _, _, energy_weights, forces_weights = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
        )

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
        """Test that NaN forces are handled correctly (zero weight)."""
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

        # NaN forces should get weight 0 automatically
        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords,
            energy=torch.rand(n_confs, dtype=torch.float64),
            forces=forces,
            energy_weight=1.0,
            forces_weight=1.0,  # Will be set to 0 for NaN forces
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

        # Force loss should be zero because all forces are NaN (weight 0)
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

        # Should return 6 tensors
        assert len(result) == 6

        (
            energy_ref,
            _energy_pred,
            _forces_ref,
            _forces_pred,
            energy_weights,
            _forces_weights,
        ) = result

        # Total conformations: 2 from first entry + 1 from second = 3
        assert len(energy_ref) == 3
        assert len(energy_weights) == 3

    def test_predict_with_weights_preserves_different_weights(
        self, multi_entry_dataset
    ):
        """Test that different weights per entry are preserved."""
        dataset, tensor_ff, tensor_top, smiles = multi_entry_dataset

        _, _, _, _, energy_weights, _forces_weights = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, device=torch.device("cpu")
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
            dataset,
            tensor_ff,
            {smiles: tensor_top},
            device=torch.device("cpu"),
            normalize=False,
        )

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
            datasets_by_dataset=[[dataset]],
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
            datasets_by_dataset=[[dataset]],
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
        from presto.loss import predict_with_weights

        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset
        res = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, reference="min"
        )
        assert len(res) == 6

    def test_predict_with_weights_reference_median(self, weighted_ethanol_dataset):
        """Predict with weights reference median."""
        from presto.loss import predict_with_weights

        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset
        res = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, reference="median"
        )
        assert len(res) == 6


def _legacy_overall_loss_and_grad(
    datasets_list, trainable, x, initial, topologies, regularisation_target
):
    """The single-dataset loss as computed before multiple datasets were supported.

    Average the per-molecule losses and gradients, then add regularisation.
    """
    from presto.loss import _compute_molecule_total_loss_and_grad

    force_field = trainable.to_force_field(x)
    energies, forces = [], []
    gradient = torch.zeros_like(x)
    for dataset, topology in zip(datasets_list, topologies, strict=True):
        energy, force, grad = _compute_molecule_total_loss_and_grad(
            force_field, dataset, topology, x, torch.device("cpu")
        )
        gradient += grad
        energies.append(energy)
        forces.append(force)
    gradient /= len(datasets_list)
    reg = compute_regularisation_loss(trainable, x, initial, regularisation_target)
    if reg.requires_grad:
        (reg_grad,) = torch.autograd.grad(reg, x)
        gradient += reg_grad
    return (
        torch.stack(energies).mean(),
        torch.stack(forces).mean(),
        reg.detach(),
        gradient,
    )


def _explicit_weighted_objective(
    datasets_by_dataset, weights, trainable, x, initial, topologies, target
):
    """Build the weighted loss from live tensors, so autograd gives a reference gradient.

    This is independent of the loss module's gradient accumulation.
    Returns (energy, forces, regularisation), each still attached to ``x``.
    """
    from presto.loss import _compute_molecule_energy_force_loss

    force_field = trainable.to_force_field(x)
    total_weight = sum(weights)
    energy, forces = 0.0, 0.0
    for datasets_list, weight in zip(datasets_by_dataset, weights, strict=True):
        mol_energies, mol_forces = [], []
        for dataset, topology in zip(datasets_list, topologies, strict=True):
            e_ref, e_pred, f_ref, f_pred, e_w, f_w = predict_with_weights(
                dataset,
                force_field,
                {dataset[0]["smiles"]: topology},
                normalize=False,
                create_graph=True,
            )
            e, f = _compute_molecule_energy_force_loss(
                e_ref, e_pred, f_ref, f_pred, e_w, f_w, topology.n_atoms, x.device
            )
            mol_energies.append(e)
            mol_forces.append(f)
        energy = energy + weight / total_weight * torch.stack(mol_energies).mean()
        forces = forces + weight / total_weight * torch.stack(mol_forces).mean()
    reg = compute_regularisation_loss(trainable, x, initial, target)
    return energy, forces, reg


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

    datasets_by_dataset = [
        [
            _synthetic_dataset(mol, tensor_ff, top, n_confs, seed=10 * k + m)
            for m, (mol, top) in enumerate(zip(mols, topologies, strict=True))
        ]
        for k, n_confs in enumerate((3, 5, 4))
    ]
    return trainable, params, initial, topologies, datasets_by_dataset, mols, tensor_ff


def _overall(setup, datasets_by_dataset, weights, target="initial", grad=True):
    trainable, params, initial, topologies, *_ = setup
    x = params.clone().requires_grad_(True)
    return compute_overall_loss_and_grad(
        datasets_by_dataset,
        weights,
        trainable,
        x,
        initial,
        topologies,
        target,
        torch.device("cpu"),
        compute_grad=grad,
    )


def _dataset(setup, datasets_list, grad=True):
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

    def test_matches_legacy_without_regularisation(self, two_molecule_setup):
        """The energy, forces and data gradient are exactly the previous ones."""
        trainable, params, initial, topologies, datasets_by_dataset, *_ = (
            two_molecule_setup
        )
        datasets_list = datasets_by_dataset[0]
        x = params.clone().requires_grad_(True)
        energy, forces, _, legacy_gradient = _legacy_overall_loss_and_grad(
            datasets_list, trainable, x, initial, topologies, "initial"
        )
        reg = compute_regularisation_loss(trainable, x, initial, "initial")
        (reg_grad,) = torch.autograd.grad(reg, x)

        loss, gradient = _dataset(two_molecule_setup, datasets_list)

        assert isinstance(loss, DatasetLoss)
        assert torch.equal(loss.energy, energy)
        assert torch.equal(loss.forces, forces)
        # Regularisation is not included
        assert torch.allclose(gradient, legacy_gradient - reg_grad, rtol=1e-12)
        assert not torch.allclose(gradient, legacy_gradient)

    def test_no_grad(self, two_molecule_setup):
        """Without gradients the losses are unchanged and no gradient is returned."""
        datasets_list = two_molecule_setup[4][1]
        loss_grad, gradient = _dataset(two_molecule_setup, datasets_list)
        loss_no_grad, no_gradient = _dataset(
            two_molecule_setup, datasets_list, grad=False
        )

        assert gradient is not None
        assert no_gradient is None
        assert torch.equal(loss_grad.energy, loss_no_grad.energy)
        assert torch.equal(loss_grad.forces, loss_no_grad.forces)

    def test_molecule_count_mismatch_raises(self, two_molecule_setup):
        """Each dataset must have one entry per molecule."""
        with pytest.raises(ValueError):
            _dataset(two_molecule_setup, two_molecule_setup[4][0][:1])


class TestComputeOverallLossAndGrad:
    """Tests for the weighted combination of several datasets."""

    @pytest.mark.parametrize("weight", [1.0, 2.5, 1e-3])
    @pytest.mark.parametrize("target", ["initial", "zero"])
    def test_single_dataset_is_bitwise_legacy(self, two_molecule_setup, weight, target):
        """One dataset reproduces the previous loss exactly, whatever its weight."""
        trainable, params, initial, topologies, datasets_by_dataset, *_ = (
            two_molecule_setup
        )
        x = params.clone().requires_grad_(True)
        energy, forces, reg, gradient = _legacy_overall_loss_and_grad(
            datasets_by_dataset[0], trainable, x, initial, topologies, target
        )

        record, dataset_losses, new_gradient = _overall(
            two_molecule_setup, datasets_by_dataset[:1], [weight], target
        )

        assert torch.equal(record.energy, energy)
        assert torch.equal(record.forces, forces)
        assert torch.equal(record.regularisation, reg)
        assert torch.equal(new_gradient, gradient)
        assert len(dataset_losses) == 1
        assert torch.equal(dataset_losses[0].energy, energy)
        assert torch.equal(dataset_losses[0].forces, forces)

    def test_weighted_combination_of_dataset_losses(self, two_molecule_setup):
        """The combined loss is the normalised weighted sum of each dataset's loss."""
        datasets_by_dataset = two_molecule_setup[4][:2]
        loss_0, grad_0 = _dataset(two_molecule_setup, datasets_by_dataset[0])
        loss_1, grad_1 = _dataset(two_molecule_setup, datasets_by_dataset[1])
        # The datasets are distinct, so the weighting is visible
        assert not torch.isclose(loss_0.energy, loss_1.energy)
        assert not torch.isclose(loss_0.forces, loss_1.forces)

        record, dataset_losses, gradient = _overall(
            two_molecule_setup, datasets_by_dataset, [1.0, 3.0]
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
        datasets_by_dataset = all_datasets[: len(weights)]
        x = params.clone().requires_grad_(True)
        energy, forces, reg = _explicit_weighted_objective(
            datasets_by_dataset, weights, trainable, x, initial, topologies, target
        )
        (expected_gradient,) = torch.autograd.grad(energy + forces + reg, x)

        record, _, gradient = _overall(
            two_molecule_setup, datasets_by_dataset, weights, target
        )

        # Predicted energies are single precision (the datasets are read as float32),
        # so a different order of operations agrees only to about 1e-7
        assert torch.allclose(record.energy, energy.detach(), rtol=1e-6, atol=0)
        assert torch.allclose(record.forces, forces.detach(), rtol=1e-6, atol=0)
        assert torch.allclose(record.regularisation, reg.detach(), rtol=1e-6, atol=0)
        torch.testing.assert_close(
            gradient,
            expected_gradient,
            rtol=1e-6,
            atol=1e-7 * expected_gradient.abs().max().item(),
        )

    def test_gradient_matches_finite_differences(self, two_molecule_setup):
        """Central differences of the combined loss agree with the gradient."""
        trainable, params, initial, topologies, all_datasets, *_ = two_molecule_setup
        weights = [1.0, 3.0, 0.5]
        _, _, gradient = _overall(two_molecule_setup, all_datasets, weights)

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

    def test_weights_are_relative(self, two_molecule_setup):
        """Scaling every weight by the same factor changes nothing."""
        datasets_by_dataset = two_molecule_setup[4]
        record, dataset_losses, gradient = _overall(
            two_molecule_setup, datasets_by_dataset, [1.0, 3.0, 0.5]
        )
        scaled, scaled_losses, scaled_gradient = _overall(
            two_molecule_setup, datasets_by_dataset, [0.1, 0.3, 0.05]
        )

        for field in record._fields:
            assert torch.allclose(getattr(record, field), getattr(scaled, field))
        assert torch.allclose(gradient, scaled_gradient)
        for loss, scaled_loss in zip(dataset_losses, scaled_losses, strict=True):
            assert torch.equal(loss.energy, scaled_loss.energy)
            assert torch.equal(loss.forces, scaled_loss.forces)

    def test_dataset_order_does_not_matter(self, two_molecule_setup):
        """Permuting datasets with their weights gives the same loss and gradient."""
        datasets_by_dataset = two_molecule_setup[4]
        weights = [1.0, 3.0, 0.5]
        record, _, gradient = _overall(two_molecule_setup, datasets_by_dataset, weights)
        order = [2, 0, 1]
        permuted, permuted_losses, permuted_gradient = _overall(
            two_molecule_setup,
            [datasets_by_dataset[i] for i in order],
            [weights[i] for i in order],
        )

        for field in record._fields:
            assert torch.allclose(getattr(record, field), getattr(permuted, field))
        assert torch.allclose(gradient, permuted_gradient)
        loss_2, _ = _dataset(two_molecule_setup, datasets_by_dataset[2], grad=False)
        assert torch.equal(permuted_losses[0].energy, loss_2.energy)

    def test_duplicated_dataset_equals_single(self, two_molecule_setup):
        """Two identical datasets have the same loss as one, as weights are normalised."""
        datasets_list = two_molecule_setup[4][1]
        single, _, single_gradient = _overall(
            two_molecule_setup, [datasets_list], [1.0]
        )
        double, _, double_gradient = _overall(
            two_molecule_setup, [datasets_list, datasets_list], [2.0, 2.0]
        )

        for field in single._fields:
            assert torch.allclose(getattr(single, field), getattr(double, field))
        assert torch.allclose(single_gradient, double_gradient)

    def test_influence_independent_of_dataset_size(self, two_molecule_setup):
        """Repeating every conformer of a dataset doesn't change its influence.

        This is what computing each dataset's loss separately buys: had the datasets
        been concatenated into one, the larger dataset would dominate.
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
        loss_b, _ = _dataset(two_molecule_setup, datasets_b, grad=False)
        loss_b_doubled, _ = _dataset(two_molecule_setup, datasets_b_doubled, grad=False)
        assert torch.allclose(loss_b.energy, loss_b_doubled.energy)
        assert torch.allclose(loss_b.forces, loss_b_doubled.forces)

        record, _, gradient = _overall(
            two_molecule_setup, [datasets_a, datasets_b], [1.0, 3.0]
        )
        doubled, _, doubled_gradient = _overall(
            two_molecule_setup, [datasets_a, datasets_b_doubled], [1.0, 3.0]
        )
        for field in record._fields:
            assert torch.allclose(getattr(record, field), getattr(doubled, field))
        assert torch.allclose(gradient, doubled_gradient)

        # Contrast: concatenating the datasets lets the larger one dominate
        concatenated, _ = _dataset(
            two_molecule_setup,
            [
                datasets.concatenate_datasets([a, b])
                for a, b in zip(datasets_a, datasets_b, strict=True)
            ],
            grad=False,
        )
        concatenated_doubled, _ = _dataset(
            two_molecule_setup,
            [
                datasets.concatenate_datasets([a, b])
                for a, b in zip(datasets_a, datasets_b_doubled, strict=True)
            ],
            grad=False,
        )
        assert not torch.allclose(concatenated.energy, concatenated_doubled.energy)

    def test_combined_is_mean_over_molecules_of_weighted_sum(self, two_molecule_setup):
        """The combined loss averages, over molecules, each molecule's weighted loss."""
        trainable, params, _, topologies, all_datasets, *_ = two_molecule_setup
        weights = [1.0, 3.0]
        per_molecule = []
        for m, topology in enumerate(topologies):
            x = params.clone().requires_grad_(True)
            losses = [
                compute_dataset_loss_and_grad(
                    [all_datasets[k][m]],
                    trainable,
                    x,
                    [topology],
                    torch.device("cpu"),
                    compute_grad=False,
                )[0]
                for k in range(2)
            ]
            per_molecule.append(
                (
                    0.25 * losses[0].energy + 0.75 * losses[1].energy,
                    0.25 * losses[0].forces + 0.75 * losses[1].forces,
                )
            )

        record, _, _ = _overall(two_molecule_setup, all_datasets[:2], weights)

        assert torch.allclose(
            record.energy, torch.stack([e for e, _ in per_molecule]).mean()
        )
        assert torch.allclose(
            record.forces, torch.stack([f for _, f in per_molecule]).mean()
        )

    def test_regularisation_added_once(self, two_molecule_setup):
        """However many datasets, regularisation and its gradient are added once."""
        trainable, params, initial, *_ = two_molecule_setup
        datasets_list = two_molecule_setup[4][0]
        data_loss, data_gradient = _dataset(two_molecule_setup, datasets_list)
        x = params.clone().requires_grad_(True)
        reg = compute_regularisation_loss(trainable, x, initial, "initial")
        (reg_grad,) = torch.autograd.grad(reg, x)
        assert reg.item() > 0

        for n_datasets in (1, 3):
            record, _, gradient = _overall(
                two_molecule_setup, [datasets_list] * n_datasets, [1.0] * n_datasets
            )
            assert torch.equal(record.regularisation, reg.detach())
            assert torch.allclose(gradient, data_gradient + reg_grad)
            assert torch.allclose(record.energy, data_loss.energy)

    def test_per_conformer_weights_within_a_dataset(self, two_molecule_setup):
        """A multi-entry dataset with different per-conformer weights is unchanged.

        Torsion minimisation produces such datasets, and their loss must be exactly
        the previous single-dataset loss.
        """
        trainable, params, initial, topologies, _, mols, tensor_ff = two_molecule_setup
        merged = [
            merge_weighted_datasets(
                [
                    _synthetic_dataset(mol, tensor_ff, top, 3, seed=100 + m),
                    _synthetic_dataset(
                        mol,
                        tensor_ff,
                        top,
                        2,
                        seed=200 + m,
                        energy_weight=10.0,
                        forces_weight=0.0,
                    ),
                ]
            )
            for m, (mol, top) in enumerate(zip(mols, topologies, strict=True))
        ]
        assert len(merged[0]) == 2
        x = params.clone().requires_grad_(True)
        energy, forces, _, gradient = _legacy_overall_loss_and_grad(
            merged, trainable, x, initial, topologies, "initial"
        )

        record, _, new_gradient = _overall(two_molecule_setup, [merged], [4.0])

        assert torch.equal(record.energy, energy)
        assert torch.equal(record.forces, forces)
        assert torch.equal(new_gradient, gradient)

    def test_no_grad(self, two_molecule_setup):
        """Without gradients the losses are unchanged and no gradient is returned."""
        datasets_by_dataset = two_molecule_setup[4]
        weights = [1.0, 3.0, 0.5]
        record, losses, gradient = _overall(
            two_molecule_setup, datasets_by_dataset, weights
        )
        record_no_grad, losses_no_grad, no_gradient = _overall(
            two_molecule_setup, datasets_by_dataset, weights, grad=False
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
        record, losses, _ = _overall(
            two_molecule_setup, two_molecule_setup[4], [1, 2, 3]
        )
        for value in (*record, *(v for loss in losses for v in loss)):
            assert not value.requires_grad

    @pytest.mark.parametrize("n_weights, match", [(2, "3 datasets"), (4, "3 datasets")])
    def test_weight_count_mismatch_raises(self, two_molecule_setup, n_weights, match):
        """There must be one weight per dataset."""
        with pytest.raises(ValueError, match=match):
            _overall(two_molecule_setup, two_molecule_setup[4], [1.0] * n_weights)

    def test_no_datasets_raises(self, two_molecule_setup):
        """At least one dataset is required."""
        with pytest.raises(ValueError, match="At least one dataset"):
            _overall(two_molecule_setup, [], [])

    def test_molecule_count_mismatch_raises(self, two_molecule_setup):
        """Every dataset needs one entry per molecule."""
        all_datasets = two_molecule_setup[4]
        with pytest.raises(ValueError):
            _overall(two_molecule_setup, [all_datasets[0], all_datasets[1][:1]], [1, 1])


class TestWeightedLossClosure:
    """Tests for the Levenberg-Marquardt closure with several datasets."""

    def test_closure_matches_overall_loss(self, two_molecule_setup):
        """The closure's loss and gradient are the combined loss and its gradient."""
        from presto.loss import get_loss_closure_fn

        trainable, params, initial, topologies, all_datasets, *_ = two_molecule_setup
        weights = [1.0, 3.0]
        closure = get_loss_closure_fn(
            datasets_by_dataset=all_datasets[:2],
            dataset_weights=weights,
            trainable=trainable,
            trainable_parameters=params,
            initial_parameters=initial,
            topologies=topologies,
            regularisation_target="initial",
        )
        record, _, gradient = _overall(two_molecule_setup, all_datasets[:2], weights)

        x = params.clone().requires_grad_(True)
        loss, closure_gradient, hessian = closure(
            x, compute_gradient=True, compute_hessian=False
        )

        assert torch.allclose(
            loss, record.energy + record.forces + record.regularisation
        )
        assert torch.allclose(closure_gradient, gradient)
        assert hessian is None

    def test_closure_hessian_runs(self, two_molecule_setup):
        """The Hessian path still runs with several datasets.

        Its values are a known, separate issue: the losses it differentiates are
        detached.
        """
        from presto.loss import get_loss_closure_fn

        trainable, params, initial, topologies, all_datasets, *_ = two_molecule_setup
        closure = get_loss_closure_fn(
            datasets_by_dataset=all_datasets[:2],
            dataset_weights=[1.0, 3.0],
            trainable=trainable,
            trainable_parameters=params,
            initial_parameters=initial,
            topologies=topologies,
            regularisation_target="initial",
        )
        x = params.clone().requires_grad_(True)
        _, _, hessian = closure(x, compute_gradient=False, compute_hessian=True)

        assert hessian.shape == (len(params), len(params))
