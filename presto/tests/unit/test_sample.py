"""Unit tests for sample module."""

import datasets
import openmm
import pytest
import torch
from openff.toolkit import ForceField, Molecule

from presto._exceptions import InvalidSettingsError
from presto.data_utils import create_dataset_with_uniform_weights
from presto.sample import (
    _SAMPLING_FNS_REGISTRY,
    _get_ml_omm_system,
    load_precomputed_dataset,
)
from presto.settings import PreComputedDatasetSettings

# Check if NNPOps is available (required for EGRET-1 and MACE models)
try:
    import NNPOps  # noqa: F401

    NNPOPS_AVAILABLE = True
except ImportError:
    NNPOPS_AVAILABLE = False

requires_nnpops = pytest.mark.skipif(
    not NNPOPS_AVAILABLE,
    reason="NNPOps not available (required for EGRET-1 and MACE models)",
)


class TestPreComputedDatasetSettings:
    """Tests for PreComputedDatasetSettings."""

    def test_sampling_protocol(self, tmp_path):
        """Test that sampling protocol is set correctly."""
        settings = PreComputedDatasetSettings(dataset_paths=tmp_path)
        assert settings.sampling_protocol == "pre_computed"

    def test_output_types_empty(self, tmp_path):
        """Test that output types are empty."""
        settings = PreComputedDatasetSettings(dataset_paths=tmp_path)
        assert settings.output_types == set()

    def test_dataset_path_required(self):
        """Test that dataset_paths is required."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PreComputedDatasetSettings()  # type: ignore[call-arg]

    def test_single_path_normalized_to_list(self, tmp_path):
        """Test that a single path is normalized to a list."""
        settings = PreComputedDatasetSettings(dataset_paths=tmp_path)
        assert isinstance(settings.dataset_paths, list)
        assert len(settings.dataset_paths) == 1
        assert settings.dataset_paths[0] == tmp_path

    def test_list_of_paths_accepted(self, tmp_path):
        """Test that a list of paths is accepted."""
        path1 = tmp_path / "dataset1"
        path2 = tmp_path / "dataset2"
        settings = PreComputedDatasetSettings(dataset_paths=[path1, path2])
        assert isinstance(settings.dataset_paths, list)
        assert len(settings.dataset_paths) == 2
        assert settings.dataset_paths[0] == path1
        assert settings.dataset_paths[1] == path2


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
        settings = PreComputedDatasetSettings(dataset_paths=sample_dataset)
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
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
        settings = PreComputedDatasetSettings(dataset_paths=tmp_path / "nonexistent")
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
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
        from presto.outputs import OutputType

        settings = PreComputedDatasetSettings(dataset_paths=sample_dataset)
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
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

    def test_load_multiple_datasets(self, tmp_path):
        """Test loading multiple precomputed datasets for multi-molecule fits."""
        # Create two datasets for two different molecules
        mol1 = Molecule.from_smiles("CCO")
        smiles1 = mol1.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        dataset1 = create_dataset_with_uniform_weights(
            smiles=smiles1,
            coords=torch.rand(3, mol1.n_atoms, 3, dtype=torch.float64),
            energy=torch.rand(3, dtype=torch.float64),
            forces=torch.rand(3, mol1.n_atoms, 3, dtype=torch.float64),
            energy_weight=1000.0,
            forces_weight=0.1,
        )

        mol2 = Molecule.from_smiles("CC")
        smiles2 = mol2.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        dataset2 = create_dataset_with_uniform_weights(
            smiles=smiles2,
            coords=torch.rand(2, mol2.n_atoms, 3, dtype=torch.float64),
            energy=torch.rand(2, dtype=torch.float64),
            forces=torch.rand(2, mol2.n_atoms, 3, dtype=torch.float64),
            energy_weight=1000.0,
            forces_weight=0.1,
        )

        dataset_path1 = tmp_path / "dataset_mol0"
        dataset_path2 = tmp_path / "dataset_mol1"
        dataset1.save_to_disk(str(dataset_path1))
        dataset2.save_to_disk(str(dataset_path2))

        settings = PreComputedDatasetSettings(
            dataset_paths=[dataset_path1, dataset_path2]
        )
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
        device = torch.device("cpu")

        result = load_precomputed_dataset(
            mols=[mol1, mol2],
            off_ff=ff,
            device=device,
            settings=settings,
            output_paths={},
        )

        assert len(result) == 2
        assert isinstance(result[0], datasets.Dataset)
        assert isinstance(result[1], datasets.Dataset)
        assert len(result[0]) == 1  # One entry in the dataset
        assert len(result[1]) == 1

    def test_raises_when_path_count_mismatch(self, sample_dataset):
        """Test ValueError when number of paths doesn't match number of molecules."""
        settings = PreComputedDatasetSettings(dataset_paths=sample_dataset)
        mol1 = Molecule.from_smiles("CCO")
        mol2 = Molecule.from_smiles("CC")
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
        device = torch.device("cpu")

        with pytest.raises(
            ValueError,
            match="Number of dataset paths .* must match number of molecules",
        ):
            load_precomputed_dataset(
                mols=[mol1, mol2],  # 2 molecules
                off_ff=ff,
                device=device,
                settings=settings,  # 1 path
                output_paths={},
            )

    def test_multi_molecule_missing_file(self, tmp_path):
        """Test FileNotFoundError when one of multiple datasets is missing."""
        mol = Molecule.from_smiles("CCO")
        smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=torch.rand(3, mol.n_atoms, 3, dtype=torch.float64),
            energy=torch.rand(3, dtype=torch.float64),
            forces=torch.rand(3, mol.n_atoms, 3, dtype=torch.float64),
            energy_weight=1000.0,
            forces_weight=0.1,
        )

        dataset_path1 = tmp_path / "dataset_mol0"
        dataset_path2 = tmp_path / "dataset_mol1"  # This won't exist
        dataset.save_to_disk(str(dataset_path1))

        settings = PreComputedDatasetSettings(
            dataset_paths=[dataset_path1, dataset_path2]
        )
        mol1 = Molecule.from_smiles("CCO")
        mol2 = Molecule.from_smiles("CC")
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
        device = torch.device("cpu")

        with pytest.raises(FileNotFoundError, match="Dataset not found.*molecule 1"):
            load_precomputed_dataset(
                mols=[mol1, mol2],
                off_ff=ff,
                device=device,
                settings=settings,
                output_paths={},
            )


class TestGetMlOmmSystem:
    """Tests for _get_ml_omm_system function."""

    @requires_nnpops
    def test_neutral_molecule_with_egret(self):
        """Test creating system for neutral molecule with EGRET-1."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=1)

        system = _get_ml_omm_system(mol, "egret-1")

        assert isinstance(system, openmm.System)
        assert system.getNumParticles() == mol.n_atoms

    def test_neutral_molecule_with_aceff(self):
        """Test creating system for neutral molecule with ACEFF-2.0."""
        mol = Molecule.from_smiles("C")
        mol.generate_conformers(n_conformers=1)

        system = _get_ml_omm_system(mol, "aceff-2.0")

        assert isinstance(system, openmm.System)
        assert system.getNumParticles() == mol.n_atoms

    def test_charged_molecule_with_aceff(self):
        """Test creating system for charged molecule with ACEFF-2.0."""
        mol = Molecule.from_smiles("[NH4+]")
        mol.generate_conformers(n_conformers=1)

        # Should not raise
        system = _get_ml_omm_system(mol, "aceff-2.0")

        assert isinstance(system, openmm.System)
        assert system.getNumParticles() == mol.n_atoms

    def test_charged_molecule_with_aimnet2(self):
        """Test creating system for charged molecule with AIMNet2."""
        mol = Molecule.from_smiles("[Cl-]")
        mol.generate_conformers(n_conformers=1)

        # Should not raise
        system = _get_ml_omm_system(mol, "aimnet2_b973c_d3_ens")

        assert isinstance(system, openmm.System)
        assert system.getNumParticles() == mol.n_atoms

    def test_charged_molecule_with_unsupported_model_raises(self):
        """Test that charged molecule with unsupported model raises error."""
        mol = Molecule.from_smiles("[NH4+]")
        mol.generate_conformers(n_conformers=1)

        with pytest.raises(
            InvalidSettingsError, match="does not support charged molecules"
        ):
            _get_ml_omm_system(mol, "egret-1")

    def test_charged_molecule_with_mace_raises(self):
        """Test that charged molecule with MACE raises error."""
        mol = Molecule.from_smiles("[Cl-]")
        mol.generate_conformers(n_conformers=1)

        with pytest.raises(
            InvalidSettingsError, match="does not support charged molecules"
        ):
            _get_ml_omm_system(mol, "mace-off23-small")

    @pytest.mark.parametrize(
        "smiles",
        ["C", "CCO", "c1ccccc1", "CC(C)C"],
    )
    def test_various_neutral_molecules(self, smiles):
        """Test various neutral molecules work with ACEFF-2.0."""
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)

        # Test with ACEFF-2.0 (doesn't require NNPOps)
        system = _get_ml_omm_system(mol, "aceff-2.0")
        assert isinstance(system, openmm.System)

    @pytest.mark.parametrize(
        "charged_smiles",
        ["[NH4+]", "[Cl-]", "[Na+]", "[Ca+2]"],
    )
    def test_various_charged_molecules_with_compatible_models(self, charged_smiles):
        """Test various charged molecules work with compatible models."""
        mol = Molecule.from_smiles(charged_smiles)
        mol.generate_conformers(n_conformers=1)

        # Should work with charge-supporting models
        system1 = _get_ml_omm_system(mol, "aceff-2.0")
        system2 = _get_ml_omm_system(mol, "aimnet2_b973c_d3_ens")

        assert isinstance(system1, openmm.System)
        assert isinstance(system2, openmm.System)

    @pytest.mark.parametrize(
        "charged_smiles,unsupported_model",
        [
            ("[NH4+]", "egret-1"),
            ("[Cl-]", "mace-off23-small"),
            ("[Na+]", "mace-off23-medium"),
            ("[Ca+2]", "mace-off23-large"),
        ],
    )
    def test_various_charged_molecules_with_incompatible_models_raise(
        self, charged_smiles, unsupported_model
    ):
        """Test that various charged molecules fail with incompatible models."""
        mol = Molecule.from_smiles(charged_smiles)
        mol.generate_conformers(n_conformers=1)

        with pytest.raises(
            InvalidSettingsError, match="does not support charged molecules"
        ):
            _get_ml_omm_system(mol, unsupported_model)
