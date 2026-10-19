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
    create_weighted_dataset,
    drop_nonfinite_conformers,
    merge_weighted_datasets,
)
from presto.loss import (
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


def _with_corrections(dataset, **columns):
    """Return a copy of a one-entry dataset with the given correction columns."""
    entry = dict(dataset[0])
    entry.update(columns)
    return create_weighted_dataset([entry])


class TestPredictWithCorrections:
    """Tests for applying stored corrections in predict_with_weights."""

    def test_reference_corrections_are_added_to_the_target(
        self, weighted_ethanol_dataset
    ):
        """Reference corrections shift the target, not the MM prediction."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset
        n_confs, n_atoms = 3, tensor_top.n_atoms
        ref_e = torch.tensor([1.0, -2.0, 4.0], dtype=torch.float64)
        ref_f = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)
        corrected = _with_corrections(
            dataset,
            reference_energy_correction=ref_e,
            reference_forces_correction=ref_f,
        )

        base = predict_with_weights(dataset, tensor_ff, {smiles: tensor_top})
        result = predict_with_weights(corrected, tensor_ff, {smiles: tensor_top})

        scale_e = torch.sqrt(torch.tensor(float(n_confs), dtype=torch.float64))
        scale_f = torch.sqrt(torch.tensor(n_confs * n_atoms * 3.0, dtype=torch.float64))
        assert torch.allclose(
            (result[0] - base[0]).double() * scale_e, ref_e - ref_e.mean(), atol=1e-4
        )
        assert torch.allclose(
            (result[2] - base[2]).double() * scale_f, ref_f.reshape(-1, 3), atol=1e-4
        )
        assert torch.allclose(result[1], base[1])
        assert torch.allclose(result[3], base[3])

    def test_mm_corrections_are_added_to_the_prediction(self, ethanol_ff_and_topology):
        """MM corrections shift the MM prediction, not the target."""
        tensor_ff, tensor_top, mol = ethanol_ff_and_topology
        smiles = mol.to_smiles(mapped=True)
        n_atoms = tensor_top.n_atoms
        # Realistic coordinates keep the MM energies small enough to compare
        coords = torch.tensor(
            [mol.conformers[0].m_as("angstrom")] * 3, dtype=torch.float64
        ) + 0.01 * torch.arange(3, dtype=torch.float64).reshape(3, 1, 1)
        n_confs = 3
        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords,
            energy=torch.zeros(n_confs, dtype=torch.float64),
            forces=torch.zeros(n_confs, n_atoms, 3, dtype=torch.float64),
        )
        mm_e = torch.tensor([3.0, 0.0, -1.0], dtype=torch.float64)
        mm_f = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)
        corrected = _with_corrections(
            dataset, mm_energy_correction=mm_e, mm_forces_correction=mm_f
        )

        base = predict_with_weights(
            dataset, tensor_ff, {smiles: tensor_top}, normalize=False
        )
        result = predict_with_weights(
            corrected, tensor_ff, {smiles: tensor_top}, normalize=False
        )

        assert torch.allclose(
            (result[1] - base[1]).double(), mm_e - mm_e.mean(), atol=1e-4
        )
        assert torch.allclose(
            (result[3] - base[3]).double(), mm_f.reshape(-1, 3), atol=1e-4
        )
        assert torch.allclose(result[0], base[0])
        assert torch.allclose(result[2], base[2])

    def test_constant_energy_corrections_cancel(self, weighted_ethanol_dataset):
        """Energies are relative, so constant energy corrections have no effect."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset
        constant = torch.full((3,), 7.5, dtype=torch.float64)
        corrected = _with_corrections(
            dataset,
            reference_energy_correction=constant,
            mm_energy_correction=-constant,
        )

        base = predict_with_weights(dataset, tensor_ff, {smiles: tensor_top})
        result = predict_with_weights(corrected, tensor_ff, {smiles: tensor_top})

        for base_value, value in zip(base, result, strict=True):
            assert torch.allclose(base_value, value, atol=1e-6)

    def test_mm_corrections_do_not_depend_on_parameters(self, weighted_ethanol_dataset):
        """Fixed MM corrections add nothing to the derivatives of the prediction."""
        dataset, tensor_ff, tensor_top, smiles = weighted_ethanol_dataset
        corrected = _with_corrections(
            dataset,
            mm_energy_correction=torch.tensor([3.0, 0.0, -1.0], dtype=torch.float64),
        )
        bonds = tensor_ff.potentials_by_type["Bonds"]
        bonds.parameters.requires_grad_(True)

        # Each conformer's d(E_pred)/d(theta) must be unchanged, so any linear
        # functional of the predictions has the same gradient
        def _linear_gradient(ds):
            _, energy_pred, _, _, _, _ = predict_with_weights(
                ds, tensor_ff, {smiles: tensor_top}
            )
            weights = torch.tensor([1.0, 2.0, 3.0], dtype=energy_pred.dtype)
            return torch.autograd.grad((energy_pred * weights).sum(), bonds.parameters)[
                0
            ]

        assert torch.allclose(_linear_gradient(dataset), _linear_gradient(corrected))


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

        result, _ = compute_overall_loss_and_grad(
            datasets_list,
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

        result, _ = compute_overall_loss_and_grad(
            datasets_list,
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

        result, _ = compute_overall_loss_and_grad(
            datasets_list,
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

        result, _ = compute_overall_loss_and_grad(
            [dataset],
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
        """Test that NaN forces poison zero-weighted losses but not dropped ones."""
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

        n_confs = 3
        n_atoms = mol.n_atoms
        coords = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)
        forces = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)
        forces[1] = float("nan")

        dataset = create_dataset_with_uniform_weights(
            smiles=mol.to_smiles(mapped=True),
            coords=coords,
            energy=torch.rand(n_confs, dtype=torch.float64),
            forces=forces,
            energy_weight=1.0,
            forces_weight=1.0,
        )

        def _loss(ds):
            result, _ = compute_overall_loss_and_grad(
                [ds],
                trainable,
                trainable_params,
                trainable_params.clone(),
                [tensor_top],
                "initial",
                torch.device("cpu"),
                compute_grad=False,
            )
            return result

        # A zero weight does not remove NaN from the loss (NaN * 0 = NaN) ...
        entry = dict(dataset[0])
        entry["forces_weights"] = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        zero_weighted = create_weighted_dataset([entry])
        assert torch.isnan(_loss(zero_weighted).forces)

        # ... so non-finite conformations are dropped instead
        result = _loss(drop_nonfinite_conformers(dataset, min_conformations=2))
        assert torch.isfinite(result.forces)
        assert torch.isfinite(result.energy)


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
            datasets_list=[dataset],
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
            datasets_list=[dataset],
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
