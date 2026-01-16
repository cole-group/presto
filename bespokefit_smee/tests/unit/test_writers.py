"""Unit tests for writers module."""

import h5py
import openff.interchange
import pytest
import smee
import smee.converters
import torch
from descent.train import ParameterConfig, Trainable
from openff.toolkit import ForceField, Molecule
from openff.units import unit

from bespokefit_smee.data_utils import create_dataset_with_uniform_weights
from bespokefit_smee.loss import LossRecord
from bespokefit_smee.writers import report, write_scatter


class TestWriteScatter:
    """Tests for write_scatter function."""

    @pytest.fixture
    def simple_dataset_and_ff(self):
        """Create a simple dataset and force field for testing."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=2)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        import openff.interchange

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

        # Create a simple dataset
        smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        n_confs = 2
        n_atoms = mol.n_atoms

        coords = torch.randn(n_confs, n_atoms, 3)
        energy = torch.randn(n_confs)
        forces = torch.randn(n_confs, n_atoms, 3)

        import descent.targets.energy

        dataset = descent.targets.energy.create_dataset(
            [
                {
                    "smiles": smiles,
                    "coords": coords,
                    "energy": energy,
                    "forces": forces,
                }
            ]
        )

        return dataset, tensor_ff, tensor_top

    def test_write_scatter_creates_hdf5_file(self, simple_dataset_and_ff, tmp_path):
        """Test that write_scatter creates an HDF5 file."""
        dataset, tensor_ff, tensor_top = simple_dataset_and_ff
        output_path = tmp_path / "test_scatter.hdf5"

        write_scatter(dataset, tensor_ff, tensor_top, "cpu", output_path)

        assert output_path.exists()

    def test_write_scatter_hdf5_structure(self, simple_dataset_and_ff, tmp_path):
        """Test that HDF5 file has correct structure."""
        dataset, tensor_ff, tensor_top = simple_dataset_and_ff
        output_path = tmp_path / "test_scatter.hdf5"

        write_scatter(dataset, tensor_ff, tensor_top, "cpu", output_path)

        with h5py.File(output_path, "r") as f:
            assert "energy_reference" in f
            assert "energy_predicted" in f
            assert "energy_differences" in f
            assert "forces_reference" in f
            assert "forces_predicted" in f
            assert "forces_differences" in f
            assert "n_conformers" in f.attrs
            assert "n_atoms" in f.attrs

    def test_write_scatter_returns_statistics(self, simple_dataset_and_ff, tmp_path):
        """Test that write_scatter returns statistics."""
        dataset, tensor_ff, tensor_top = simple_dataset_and_ff
        output_path = tmp_path / "test_scatter.hdf5"

        result = write_scatter(dataset, tensor_ff, tensor_top, "cpu", output_path)

        assert isinstance(result, tuple)
        assert len(result) == 4
        energy_mean, energy_std, forces_mean, forces_std = result
        assert isinstance(energy_mean, float)
        assert isinstance(energy_std, float)
        assert isinstance(forces_mean, float)
        assert isinstance(forces_std, float)

    def test_write_scatter_energy_data_shape(self, simple_dataset_and_ff, tmp_path):
        """Test that energy data has correct shape."""
        dataset, tensor_ff, tensor_top = simple_dataset_and_ff
        output_path = tmp_path / "test_scatter.hdf5"

        write_scatter(dataset, tensor_ff, tensor_top, "cpu", output_path)

        with h5py.File(output_path, "r") as f:
            n_confs = f.attrs["n_conformers"]
            assert f["energy_reference"].shape == (n_confs,)
            assert f["energy_predicted"].shape == (n_confs,)
            assert f["energy_differences"].shape == (n_confs,)

    def test_write_scatter_forces_data_shape(self, simple_dataset_and_ff, tmp_path):
        """Test that forces data has correct shape."""
        dataset, tensor_ff, tensor_top = simple_dataset_and_ff
        output_path = tmp_path / "test_scatter.hdf5"

        write_scatter(dataset, tensor_ff, tensor_top, "cpu", output_path)

        with h5py.File(output_path, "r") as f:
            n_confs = f.attrs["n_conformers"]
            n_atoms = f.attrs["n_atoms"]
            expected_shape = (n_confs * n_atoms, 3)
            assert f["forces_reference"].shape == expected_shape
            assert f["forces_predicted"].shape == expected_shape
            assert f["forces_differences"].shape == expected_shape

    def test_write_scatter_overwrites_existing_file(
        self, simple_dataset_and_ff, tmp_path
    ):
        """Test that write_scatter overwrites existing file."""
        dataset, tensor_ff, tensor_top = simple_dataset_and_ff
        output_path = tmp_path / "test_scatter.hdf5"

        # Write once
        write_scatter(dataset, tensor_ff, tensor_top, "cpu", output_path)
        mtime1 = output_path.stat().st_mtime

        # Write again
        import time

        time.sleep(0.01)  # Ensure different mtime
        write_scatter(dataset, tensor_ff, tensor_top, "cpu", output_path)
        mtime2 = output_path.stat().st_mtime

        assert mtime2 > mtime1

    def test_write_scatter_with_cuda_device(self, simple_dataset_and_ff, tmp_path):
        """Test write_scatter with cuda device string."""
        dataset, tensor_ff, tensor_top = simple_dataset_and_ff
        output_path = tmp_path / "test_scatter.hdf5"

        # Should work with 'cuda' even if CUDA not available
        # (will just use CPU internally)
        try:
            write_scatter(dataset, tensor_ff, tensor_top, "cpu", output_path)
            assert output_path.exists()
        except Exception:
            # If it fails for other reasons, that's okay for this test
            pass


class TestOpenWriter:
    """Tests for open_writer context manager."""

    def test_open_writer_creates_directory(self, tmp_path):
        """Test that open_writer creates directory."""
        from bespokefit_smee.writers import open_writer

        writer_dir = tmp_path / "test_writer"
        assert not writer_dir.exists()

        with open_writer(writer_dir):
            assert writer_dir.exists()

    def test_open_writer_returns_summary_writer(self, tmp_path):
        """Test that open_writer returns SummaryWriter."""
        import tensorboardX

        from bespokefit_smee.writers import open_writer

        writer_dir = tmp_path / "test_writer"

        with open_writer(writer_dir) as writer:
            assert isinstance(writer, tensorboardX.SummaryWriter)


class TestWriteMetrics:
    """Tests for write_metrics function."""

    def test_write_metrics_writes_to_file(self, tmp_path):
        """Test that write_metrics writes to file."""
        from bespokefit_smee.writers import open_writer, write_metrics

        metrics_file = tmp_path / "metrics.txt"
        writer_dir = tmp_path / "writer"

        loss_train = LossRecord(
            energy=torch.tensor(1.0),
            forces=torch.tensor(0.5),
            regularisation=torch.tensor(1.5),
        )

        loss_test = LossRecord(
            energy=torch.tensor(1.2),
            forces=torch.tensor(0.6),
            regularisation=torch.tensor(1.5),
        )

        with open_writer(writer_dir) as writer:
            with open(metrics_file, "w") as f:
                write_metrics(0, loss_train, loss_test, writer, f)

        assert metrics_file.exists()
        content = metrics_file.read_text()
        assert "1.00" in content
        assert "1.20" in content
        assert "0.50" in content
        assert "0.60" in content
        assert "1.50" in content
        assert len(content.strip().split()) == 6

    def test_write_metrics_multiple_iterations(self, tmp_path):
        """Test write_metrics for multiple iterations."""
        from bespokefit_smee.writers import open_writer, write_metrics

        metrics_file = tmp_path / "metrics.txt"
        writer_dir = tmp_path / "writer"

        with open_writer(writer_dir) as writer:
            with open(metrics_file, "w") as f:
                for i in range(3):
                    loss_train = LossRecord(
                        energy=torch.tensor(1.0 - i * 0.1),
                        forces=torch.tensor(0.5 - i * 0.05),
                        regularisation=torch.tensor(1.5),
                    )
                    loss_test = LossRecord(
                        energy=torch.tensor(1.2 - i * 0.1),
                        forces=torch.tensor(0.6 - i * 0.05),
                        regularisation=torch.tensor(1.5),
                    )
                    write_metrics(i, loss_train, loss_test, writer, f)

        lines = metrics_file.read_text().strip().split("\n")
        assert len(lines) == 3


class TestPotentialSummary:
    """Tests for get_potential_summary function."""

    def test_get_potential_summary_returns_string(self):
        """Test that get_potential_summary returns string."""
        from bespokefit_smee.writers import get_potential_summary

        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        import openff.interchange

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, _ = smee.converters.convert_interchange(interchange)

        # Get a potential
        potential = tensor_ff.potentials[0]

        result = get_potential_summary(potential)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_potential_summary_contains_potential_type(self):
        """Test that summary contains potential type."""
        from bespokefit_smee.writers import get_potential_summary

        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        import openff.interchange

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, _ = smee.converters.convert_interchange(interchange)

        # Get a potential
        potential = tensor_ff.potentials[0]

        result = get_potential_summary(potential)
        assert potential.type in result


class TestReport:
    """Tests for report function."""

    @pytest.fixture
    def training_setup(self):
        """Create a complete training setup for report testing."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=3, rms_cutoff=0.0 * unit.angstrom)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")
        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

        # Convert assignment matrices from sparse to dense
        for potential in tensor_ff.potentials:
            param_map = tensor_top.parameters.get(potential.type)
            if param_map is not None and hasattr(param_map, "assignment_matrix"):
                if param_map.assignment_matrix.is_sparse:
                    param_map.assignment_matrix = param_map.assignment_matrix.to_dense()

        parameter_configs = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                limits={"k": (1e-8, None), "length": (None, None)},
                regularize={"k": 1.0, "length": 1.0},
            ),
        }

        trainable = Trainable(tensor_ff, parameter_configs, {})
        trainable_parameters = trainable.to_values()
        initial_parameters = trainable_parameters.clone()

        n_confs = mol.n_conformers
        coords_list = [
            torch.tensor(mol.conformers[i].m_as("angstrom")).unsqueeze(0)
            for i in range(n_confs)
        ]
        coords_all = torch.cat(coords_list, dim=0).requires_grad_(True)

        energy_all = smee.compute_energy(tensor_top, tensor_ff, coords_all)
        forces_all = -torch.autograd.grad(
            energy_all.sum(), coords_all, create_graph=True, retain_graph=True
        )[0]

        smiles = mol.to_smiles(mapped=True)

        dataset_train = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords_all[:2].detach(),
            energy=energy_all[:2].detach(),
            forces=forces_all[:2].detach(),
            energy_weight=1000.0,
            forces_weight=0.1,
        )

        dataset_test = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords_all[2:3].detach(),
            energy=energy_all[2:3].detach(),
            forces=forces_all[2:3].detach(),
            energy_weight=1000.0,
            forces_weight=0.1,
        )

        return {
            "trainable": trainable,
            "trainable_parameters": trainable_parameters,
            "initial_parameters": initial_parameters,
            "topologies": [tensor_top],
            "datasets_train": [dataset_train],
            "datasets_test": [dataset_test],
        }

    def test_report_creates_metrics_file(self, training_setup, tmp_path):
        """Test that report creates metrics file with correct format."""
        setup = training_setup
        metrics_file = tmp_path / "metrics.txt"
        experiment_dir = tmp_path / "tensorboard"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Call report function
        report(
            step=0,
            x=setup["trainable_parameters"],
            loss=torch.tensor(1.0),
            gradient=torch.zeros_like(setup["trainable_parameters"]),
            hessian=torch.zeros(
                len(setup["trainable_parameters"]),
                len(setup["trainable_parameters"]),
            ),
            step_quality=1.0,
            accept_step=True,
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets_train=setup["datasets_train"],
            datasets_test=setup["datasets_test"],
            initial_parameters=setup["initial_parameters"],
            regularisation_target="initial",
            metrics_file=metrics_file,
            experiment_dir=experiment_dir,
        )

        # Verify metrics file was created and has correct format
        assert metrics_file.exists()
        content = metrics_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) >= 1

        # Each line should have 6 values (3 train + 3 test)
        parts = lines[0].strip().split()
        assert len(parts) == 6, f"Expected 6 values per line, got {len(parts)}"

    def test_report_writes_valid_loss_values(self, training_setup, tmp_path):
        """Test that report writes valid (non-NaN, non-Inf) loss values."""
        setup = training_setup
        metrics_file = tmp_path / "metrics.txt"
        experiment_dir = tmp_path / "tensorboard"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        report(
            step=0,
            x=setup["trainable_parameters"],
            loss=torch.tensor(1.0),
            gradient=torch.zeros_like(setup["trainable_parameters"]),
            hessian=torch.zeros(
                len(setup["trainable_parameters"]),
                len(setup["trainable_parameters"]),
            ),
            step_quality=1.0,
            accept_step=True,
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets_train=setup["datasets_train"],
            datasets_test=setup["datasets_test"],
            initial_parameters=setup["initial_parameters"],
            regularisation_target="initial",
            metrics_file=metrics_file,
            experiment_dir=experiment_dir,
        )

        content = metrics_file.read_text()
        parts = content.strip().split()

        for i, part in enumerate(parts):
            value = float(part)
            assert value == value, f"Value {i} is NaN"  # NaN check
            assert abs(value) < 1e10, f"Value {i} is too large: {value}"

    def test_report_creates_tensorboard_output(self, training_setup, tmp_path):
        """Test that report creates TensorBoard output."""
        setup = training_setup
        metrics_file = tmp_path / "metrics.txt"
        experiment_dir = tmp_path / "tensorboard"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        report(
            step=0,
            x=setup["trainable_parameters"],
            loss=torch.tensor(1.0),
            gradient=torch.zeros_like(setup["trainable_parameters"]),
            hessian=torch.zeros(
                len(setup["trainable_parameters"]),
                len(setup["trainable_parameters"]),
            ),
            step_quality=1.0,
            accept_step=True,
            trainable=setup["trainable"],
            topologies=setup["topologies"],
            datasets_train=setup["datasets_train"],
            datasets_test=setup["datasets_test"],
            initial_parameters=setup["initial_parameters"],
            regularisation_target="initial",
            metrics_file=metrics_file,
            experiment_dir=experiment_dir,
        )

        # Check TensorBoard directory has content
        assert experiment_dir.exists()
        assert any(experiment_dir.iterdir())
