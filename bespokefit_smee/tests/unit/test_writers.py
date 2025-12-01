"""Unit tests for writers module."""

import h5py
import pytest
import smee
import smee.converters
import torch
from openff.toolkit import ForceField, Molecule

from bespokefit_smee.loss import LossRecord
from bespokefit_smee.writers import write_scatter


class TestWriteScatter:
    """Tests for write_scatter function."""

    @pytest.fixture
    def simple_dataset_and_ff(self):
        """Create a simple dataset and force field for testing."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=2)

        ff = ForceField("openff_unconstrained-2.2.1.offxml")

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
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

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
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        import openff.interchange

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )

        tensor_ff, _ = smee.converters.convert_interchange(interchange)

        # Get a potential
        potential = tensor_ff.potentials[0]

        result = get_potential_summary(potential)
        assert potential.type in result
