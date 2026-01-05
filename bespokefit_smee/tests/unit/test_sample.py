"""Unit tests for precomputed dataset loading."""

import datasets
import pytest
import torch
from openff.toolkit import ForceField, Molecule

from bespokefit_smee.data_utils import create_dataset_with_uniform_weights
from bespokefit_smee.sample import _SAMPLING_FNS_REGISTRY, load_precomputed_dataset
from bespokefit_smee.settings import PreComputedDatasetSettings


class TestPreComputedDatasetSettings:
    """Tests for PreComputedDatasetSettings."""

    def test_sampling_protocol(self, tmp_path):
        """Test that sampling protocol is set correctly."""
        settings = PreComputedDatasetSettings(dataset_path=tmp_path)
        assert settings.sampling_protocol == "pre_computed"

    def test_output_types_empty(self, tmp_path):
        """Test that output types are empty."""
        settings = PreComputedDatasetSettings(dataset_path=tmp_path)
        assert settings.output_types == set()

    def test_dataset_path_required(self):
        """Test that dataset_path is required."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PreComputedDatasetSettings()  # type: ignore[call-arg]


class TestLoadPrecomputedDataset:
    """Tests for load_precomputed_dataset function."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset saved to disk."""
        mol = Molecule.from_smiles("CCO")
        smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        n_confs = 3
        n_atoms = mol.n_atoms

        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=torch.rand(n_confs, n_atoms, 3, dtype=torch.float64),
            energy=torch.rand(n_confs, dtype=torch.float64),
            forces=torch.rand(n_confs, n_atoms, 3, dtype=torch.float64),
            energy_weight=1000.0,
            forces_weight=0.1,
        )

        dataset_path = tmp_path / "test_dataset"
        dataset.save_to_disk(str(dataset_path))
        return dataset_path

    def test_load_dataset(self, sample_dataset):
        """Test loading a precomputed dataset."""
        settings = PreComputedDatasetSettings(dataset_path=sample_dataset)
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.2.1.offxml")
        device = torch.device("cpu")

        result = load_precomputed_dataset(
            mols=[mol],
            off_ff=ff,
            device=device,
            settings=settings,
            output_paths={},
        )

        assert len(result) == 1
        assert isinstance(result[0], datasets.Dataset)

    def test_raises_file_not_found(self, tmp_path):
        """Test FileNotFoundError for non-existent path."""
        settings = PreComputedDatasetSettings(dataset_path=tmp_path / "nonexistent")
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.2.1.offxml")
        device = torch.device("cpu")

        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_precomputed_dataset(
                mols=[mol],
                off_ff=ff,
                device=device,
                settings=settings,
                output_paths={},
            )

    def test_raises_for_invalid_output_paths(self, sample_dataset):
        """Test ValueError for invalid output paths."""
        from bespokefit_smee.outputs import OutputType

        settings = PreComputedDatasetSettings(dataset_path=sample_dataset)
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.2.1.offxml")
        device = torch.device("cpu")

        with pytest.raises(ValueError, match="Output paths must contain exactly"):
            load_precomputed_dataset(
                mols=[mol],
                off_ff=ff,
                device=device,
                settings=settings,
                output_paths={OutputType.PDB_TRAJECTORY: sample_dataset},
            )

    def test_function_registered(self):
        """Test that the function is registered in the registry."""
        assert PreComputedDatasetSettings in _SAMPLING_FNS_REGISTRY
        assert (
            _SAMPLING_FNS_REGISTRY[PreComputedDatasetSettings]
            == load_precomputed_dataset
        )
