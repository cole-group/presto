"""Unit tests for sample module."""

from unittest.mock import MagicMock, patch

import datasets
import numpy as np
import openmm
import pytest
import torch
from openff.toolkit import ForceField, Molecule
from openmm import unit as omm_unit

from presto._exceptions import InvalidSettingsError
from presto.data_utils import create_dataset_with_uniform_weights, has_weights
from presto.outputs import OutputType
from presto.sample import (
    _SAMPLING_FNS_REGISTRY,
    _add_torsion_restraint_forces,
    _calculate_torsion_angles,
    _find_available_force_group,
    _get_integrator,
    _get_ml_omm_system,
    _get_molecule_from_dataset,
    _remove_torsion_restraint_forces,
    _run_md,
    _update_torsion_restraints,
    generate_torsion_minimised_dataset,
    load_precomputed_dataset,
    recalculate_energies_and_forces,
    sample_mlmd,
    sample_mmmd,
    sample_mmmd_metadynamics,
    sample_mmmd_metadynamics_with_torsion_minimisation,
)
from presto.settings import (
    MLMDSamplingSettings,
    MMMDMetadynamicsSamplingSettings,
    MMMDMetadynamicsTorsionMinimisationSamplingSettings,
    MMMDSamplingSettings,
    PreComputedDatasetSettings,
)

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
        # Verify dataset entry structure
        entry = result[0][0]
        assert "smiles" in entry
        assert "coords" in entry
        assert "energy" in entry
        assert "forces" in entry
        assert "energy_weights" in entry
        assert "forces_weights" in entry

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

    @pytest.fixture(autouse=True)
    def mock_get_mlp(self):
        """Mock get_mlp to avoid loading real models and pass isinstance checks."""
        from unittest.mock import MagicMock, patch

        import openmm

        with patch("presto.sample.mlp.get_mlp") as mock:
            mock_potential = MagicMock()

            def create_mock_system(topology, **kwargs):
                system = openmm.System()
                for _ in range(topology.getNumAtoms()):
                    system.addParticle(1.0)
                return system

            mock_potential.createSystem.side_effect = create_mock_system
            mock.return_value = mock_potential
            yield mock

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


@pytest.fixture
def mock_molecule():
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=1)
    return mol


@pytest.fixture
def mock_simulation():
    sim = MagicMock(spec=openmm.app.Simulation)

    # Mock context
    context = MagicMock()
    sim.context = context

    # Mock state
    state = MagicMock()
    state.getPositions.return_value = omm_unit.Quantity(
        np.random.rand(9, 3), omm_unit.angstrom
    )
    state.getPotentialEnergy.return_value = omm_unit.Quantity(
        10.0, omm_unit.kilocalorie_per_mole
    )
    state.getForces.return_value = omm_unit.Quantity(
        np.random.rand(9, 3), omm_unit.kilocalorie_per_mole / omm_unit.angstrom
    )
    context.getState.return_value = state

    # Mock reporters list
    sim.reporters = []

    return sim


def test_get_integrator():
    temp = 300 * omm_unit.kelvin
    dt = 2.0 * omm_unit.femtosecond
    integrator = _get_integrator(temp, dt)
    assert isinstance(integrator, openmm.LangevinMiddleIntegrator)
    assert integrator.getTemperature() == temp
    assert integrator.getStepSize() == dt


class TestTorsionRestraints:
    def test_find_available_force_group(self):
        system = openmm.System()
        # Add some forces
        for i in range(5):
            f = openmm.CustomBondForce("0")
            f.setForceGroup(i)
            system.addForce(f)

        group = _find_available_force_group(MagicMock(system=system))
        assert group == 5

    def test_add_torsion_restraint_forces(self, mock_simulation):
        mock_simulation.system = openmm.System()
        torsion_indices = [(0, 1, 2, 3), (4, 5, 6, 7)]
        k = 100.0  # kJ/mol/rad^2

        indices, group = _add_torsion_restraint_forces(
            mock_simulation, torsion_indices, k
        )

        # Should add one force per torsion
        forces = [
            mock_simulation.system.getForce(i)
            for i in range(mock_simulation.system.getNumForces())
        ]
        torsion_forces = [f for f in forces if isinstance(f, openmm.CustomTorsionForce)]
        assert len(torsion_forces) == 2
        assert len(indices) == 2
        assert group == 0

    def test_remove_torsion_restraint_forces(self, mock_simulation):
        mock_simulation.system = openmm.System()
        f1 = openmm.CustomBondForce("0")
        f2 = openmm.CustomTorsionForce("0")
        mock_simulation.system.addForce(f1)
        mock_simulation.system.addForce(f2)

        _remove_torsion_restraint_forces(mock_simulation, [1])
        assert mock_simulation.system.getNumForces() == 1
        assert mock_simulation.context.reinitialize.called

    def test_update_torsion_restraints(self, mock_simulation):
        mock_simulation.system = MagicMock()
        mock_force = MagicMock()
        mock_simulation.system.getForce.return_value = mock_force
        mock_force.getTorsionParameters.return_value = [0, 1, 2, 3, [0.0, 0.0]]

        _update_torsion_restraints(mock_simulation, [0], [1.5], 100.0)  # radians, k

        mock_force.setTorsionParameters.assert_called_with(0, 0, 1, 2, 3, [100.0, 1.5])
        mock_force.updateParametersInContext.assert_called_with(mock_simulation.context)


class TestTorsions:
    def test_calculate_torsion_angles(self):
        # Construct a simple geometry:   1-2-3-4
        # 1 at (1,0,0), 2 at (0,0,0), 3 at (0,1,0), 4 at (0,1,1) -> 90 degrees
        coords = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]]
        )

        # OpenFF / RDKit indices handling
        # Torsion 0-1-2-3
        angle = _calculate_torsion_angles(coords, (0, 1, 2, 3))

        # Expected angle is 90 degrees (pi/2)
        expected = np.pi / 2
        assert torch.isclose(angle[0], torch.tensor(expected), atol=1e-5)


def test_sample_mmmd_metadynamics_no_rotatable_bonds(tmp_path):
    """Test sampling with metadynamics for molecule with no rotatable bonds."""
    mol = Molecule.from_smiles("C")  # Methane
    mol.generate_conformers(n_conformers=1)

    settings = MMMDMetadynamicsSamplingSettings(
        sampling_protocol="mm_md_metadynamics",
        timestep=2.0 * omm_unit.femtoseconds,
        temperature=300.0 * omm_unit.kelvin,
        n_conformers=1,
        bias_frequency=0.1 * omm_unit.picoseconds,
        bias_save_frequency=0.1 * omm_unit.picoseconds,
        bias_height=2.0 * omm_unit.kilojoules_per_mole,
        equilibration_sampling_time_per_conformer=0.1 * omm_unit.picoseconds,
        production_sampling_time_per_conformer=0.1 * omm_unit.picoseconds,
        snapshot_interval=0.1 * omm_unit.picoseconds,
    )

    output_paths = {
        OutputType.PDB_TRAJECTORY: tmp_path,
        OutputType.METADYNAMICS_BIAS: tmp_path / "bias",
    }

    # This test verifies that molecules with no rotatable bonds are handled gracefully
    # We mock the heavy operations but use real datasets to avoid PyArrow errors
    with (
        patch("presto.sample.openff.interchange.Interchange.from_smirnoff"),
        patch("presto.sample._run_md") as mock_run,
        patch("presto.sample._get_ml_omm_system"),
        patch("presto.sample.Simulation"),
        patch("presto.sample.recalculate_energies_and_forces") as mock_recalc,
        patch("presto.sample.cleanup_simulation"),
    ):
        # Create a real dataset to avoid PyArrow issues
        real_dataset = create_dataset_with_uniform_weights(
            smiles="C",
            coords=torch.zeros(1, mol.n_atoms, 3, dtype=torch.float64),
            energy=torch.zeros(1, dtype=torch.float64),
            forces=torch.zeros(1, mol.n_atoms, 3, dtype=torch.float64),
            energy_weight=1000.0,
            forces_weight=0.1,
        )
        mock_run.return_value = real_dataset
        mock_recalc.return_value = real_dataset

        out = sample_mmmd_metadynamics(
            [mol], MagicMock(), torch.device("cpu"), settings, output_paths
        )
        assert len(out) == 1


def test_dataset_weights_integration():
    """Test that weighted datasets are created correctly."""
    n_confs = 5
    n_atoms = 9

    coords = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)
    energy = torch.rand(n_confs, dtype=torch.float64)
    forces = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)

    dataset = create_dataset_with_uniform_weights(
        smiles="CCO",
        coords=coords,
        energy=energy,
        forces=forces,
        energy_weight=1000.0,
        forces_weight=0.1,
    )

    assert len(dataset) == 1
    assert has_weights(dataset)
    entry = dataset[0]
    assert len(entry["energy"]) == n_confs
    assert torch.all(entry["energy_weights"] == 1000.0)
    assert torch.all(entry["forces_weights"] == 0.1)


def test_nan_forces_get_zero_weight():
    """Test that NaN forces get zero weight."""
    n_confs = 3
    n_atoms = 5

    coords = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)
    energy = torch.rand(n_confs, dtype=torch.float64)
    forces = torch.rand(n_confs, n_atoms, 3, dtype=torch.float64)
    # Set middle conformation forces to NaN
    forces[1] = float("nan")

    dataset = create_dataset_with_uniform_weights(
        smiles="CCO",
        coords=coords,
        energy=energy,
        forces=forces,
        energy_weight=1000.0,
        forces_weight=0.1,
    )

    entry = dataset[0]
    # Energy weights should all be 1000.0
    assert torch.all(entry["energy_weights"] == 1000.0)
    # Forces weights: conformation 1 should be 0, others 0.1
    expected_forces_weights = torch.tensor([0.1, 0.0, 0.1])
    assert torch.allclose(entry["forces_weights"], expected_forces_weights)


class TestCopyMolAndAddConformers:
    """Tests for _copy_mol_and_add_conformers helper function."""

    def test_generates_requested_conformers(self):
        """Test that function generates the requested number of conformers."""
        from presto.sample import _copy_mol_and_add_conformers

        mol = Molecule.from_smiles("CCCC")
        mol.generate_conformers(n_conformers=1)

        result = _copy_mol_and_add_conformers(mol, n_conformers=3)

        # Should have generated conformers (may be less if RMS cutoff limits them)
        assert len(result.conformers) >= 1
        # Original molecule should be unchanged
        assert len(mol.conformers) == 1

    def test_returns_deep_copy(self):
        """Test that function returns a deep copy of the molecule."""
        from presto.sample import _copy_mol_and_add_conformers

        mol = Molecule.from_smiles("CC")
        mol.generate_conformers(n_conformers=1)

        result = _copy_mol_and_add_conformers(mol, n_conformers=2)

        # Should be a different object
        assert result is not mol

    def test_returns_fewer_conformers_than_requested_for_simple_molecule(self):
        """Test that function handles molecules that can't generate many conformers."""
        from presto.sample import _copy_mol_and_add_conformers

        # Ethene has very limited conformational flexibility
        mol = Molecule.from_smiles("C=C")
        mol.generate_conformers(n_conformers=1)

        # Request a very large number - more than possible for this molecule
        result = _copy_mol_and_add_conformers(mol, n_conformers=100)

        # Should return a molecule with conformers (may be fewer than requested)
        assert len(result.conformers) >= 1


class TestGetMoleculeFromDataset:
    """Tests for _get_molecule_from_dataset function."""

    def test_extracts_molecule_from_dataset(self):
        """Test extracting molecule from dataset."""
        mol = Molecule.from_smiles("CCO")
        smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)

        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=torch.zeros(1, mol.n_atoms, 3, dtype=torch.float64),
            energy=torch.zeros(1, dtype=torch.float64),
            forces=torch.zeros(1, mol.n_atoms, 3, dtype=torch.float64),
            energy_weight=1.0,
            forces_weight=1.0,
        )

        result = _get_molecule_from_dataset(dataset)

        assert isinstance(result, Molecule)
        assert result.n_atoms == mol.n_atoms


class TestCalculateTorsionAnglesExtended:
    """Additional tests for _calculate_torsion_angles function."""

    @pytest.mark.parametrize("n_snapshots", [1, 5, 10])
    def test_handles_multiple_snapshots(self, n_snapshots):
        """Test that function handles multiple snapshots correctly."""
        # Create coords with shape (n_snapshots, 4, 3)
        coords = torch.rand(n_snapshots, 4, 3, dtype=torch.float64)
        torsion_atoms = (0, 1, 2, 3)

        result = _calculate_torsion_angles(coords, torsion_atoms)

        assert result.shape == (n_snapshots,)
        # All angles should be in valid range
        assert torch.all(result >= -np.pi)
        assert torch.all(result <= np.pi)

    def test_known_geometry_180_degrees(self):
        """Test with trans configuration (180 degrees)."""
        # Trans configuration: atoms in a plane, 180 degrees
        coords = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],  # Atom 0
                    [1.0, 0.0, 0.0],  # Atom 1
                    [2.0, 1.0, 0.0],  # Atom 2
                    [3.0, 1.0, 0.0],
                ]  # Atom 3
            ],
            dtype=torch.float64,
        )

        angle = _calculate_torsion_angles(coords, (0, 1, 2, 3))

        # Trans should be close to pi or -pi
        assert torch.isclose(
            torch.abs(angle[0]), torch.tensor(np.pi, dtype=torch.float64), atol=0.1
        )


class TestGetTorsionBiasForces:
    """Tests for _get_torsion_bias_forces function."""

    def test_returns_bias_variables_for_rotatable_bonds(self):
        """Test that bias variables are created for rotatable bonds."""
        from presto.sample import _get_torsion_bias_forces

        mol = Molecule.from_smiles("CCCC")  # Butane has rotatable bonds
        mol.generate_conformers(n_conformers=1)

        bias_vars = _get_torsion_bias_forces(mol)

        assert len(bias_vars) > 0
        assert all(
            isinstance(v, openmm.app.metadynamics.BiasVariable) for v in bias_vars
        )

    def test_returns_empty_for_no_rotatable_bonds(self):
        """Test returns empty list for molecule with no rotatable bonds."""
        from presto.sample import _get_torsion_bias_forces

        mol = Molecule.from_smiles("C")  # Methane
        mol.generate_conformers(n_conformers=1)

        bias_vars = _get_torsion_bias_forces(mol)

        assert len(bias_vars) == 0

    def test_custom_bias_width(self):
        """Test that custom bias width is applied."""
        from presto.sample import _get_torsion_bias_forces

        mol = Molecule.from_smiles("CCCC")
        mol.generate_conformers(n_conformers=1)

        custom_width = 0.5
        bias_vars = _get_torsion_bias_forces(mol, bias_width=custom_width)

        assert len(bias_vars) > 0
        # Verify bias width was set (BiasVariable stores it internally)
        for bv in bias_vars:
            assert bv.biasWidth == custom_width


class TestFindAvailableForceGroupEdgeCases:
    """Edge case tests for _find_available_force_group."""

    def test_all_groups_exhausted_raises(self):
        """Test that exhausting all force groups raises RuntimeError."""
        system = openmm.System()
        for i in range(32):
            f = openmm.CustomBondForce("0")
            f.setForceGroup(i)
            system.addForce(f)

        sim = MagicMock()
        sim.system = system

        with pytest.raises(RuntimeError, match="All force groups"):
            _find_available_force_group(sim)

    def test_finds_gap_in_used_groups(self):
        """Test that function finds gaps in used force groups."""
        system = openmm.System()
        # Use groups 0, 1, 3 (skip 2)
        for i in [0, 1, 3]:
            f = openmm.CustomBondForce("0")
            f.setForceGroup(i)
            system.addForce(f)

        sim = MagicMock()
        sim.system = system

        result = _find_available_force_group(sim)

        assert result == 2  # Should find the gap


class TestSamplingFunctionsValidation:
    """Tests for validation in sampling functions."""

    def test_sample_mmmd_validates_output_paths(self, mock_molecule):
        """Test that sample_mmmd validates output paths."""
        settings_obj = MMMDSamplingSettings(
            sampling_protocol="mm_md",
            timestep=2.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
        )

        with pytest.raises(ValueError, match="Output paths must contain exactly"):
            sample_mmmd(
                [mock_molecule],
                MagicMock(),
                torch.device("cpu"),
                settings_obj,
                {},  # Missing required output paths
            )

    def test_sample_mlmd_validates_output_paths(self, mock_molecule):
        """Test that sample_mlmd validates output paths."""
        settings_obj = MLMDSamplingSettings(
            sampling_protocol="ml_md",
            timestep=2.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
        )

        with pytest.raises(ValueError, match="Output paths must contain exactly"):
            sample_mlmd(
                [mock_molecule],
                MagicMock(),
                torch.device("cpu"),
                settings_obj,
                {},
            )

    def test_sample_mmmd_metadynamics_validates_output_paths(self, mock_molecule):
        """Test that sample_mmmd_metadynamics validates output paths."""
        settings_obj = MMMDMetadynamicsSamplingSettings(
            sampling_protocol="mm_md_metadynamics",
            timestep=2.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            bias_frequency=0.1 * omm_unit.picoseconds,
            bias_save_frequency=0.1 * omm_unit.picoseconds,
            bias_height=2.0 * omm_unit.kilojoules_per_mole,
        )

        with pytest.raises(ValueError, match="Output paths must contain exactly"):
            sample_mmmd_metadynamics(
                [mock_molecule],
                MagicMock(),
                torch.device("cpu"),
                settings_obj,
                {},
            )

    def test_sample_mmmd_metadynamics_with_torsion_min_validates_output_paths(
        self, mock_molecule
    ):
        """Test that sample_mmmd_metadynamics_with_torsion_minimisation validates output paths."""
        settings_obj = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            sampling_protocol="mm_md_metadynamics_torsion_minimisation",
            timestep=2.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            bias_frequency=0.1 * omm_unit.picoseconds,
            bias_save_frequency=0.1 * omm_unit.picoseconds,
            bias_height=2.0 * omm_unit.kilojoules_per_mole,
        )

        with pytest.raises(ValueError, match="Output paths must contain exactly"):
            sample_mmmd_metadynamics_with_torsion_minimisation(
                [mock_molecule],
                MagicMock(),
                torch.device("cpu"),
                settings_obj,
                {},
            )


class TestGenerateTorsionMinimisedDatasetEdgeCases:
    """Edge case tests for generate_torsion_minimised_dataset."""

    def test_empty_torsions_returns_empty_dataset(self):
        """Test that molecule with no torsions returns empty dataset."""
        mol = Molecule.from_smiles("C")  # Methane - no rotatable bonds
        smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)

        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=torch.zeros(1, mol.n_atoms, 3, dtype=torch.float64),
            energy=torch.zeros(1, dtype=torch.float64),
            forces=torch.zeros(1, mol.n_atoms, 3, dtype=torch.float64),
            energy_weight=1.0,
            forces_weight=1.0,
        )

        with patch("presto.sample.get_rot_torsions_by_rot_bond", return_value={}):
            mm_result, ml_result = generate_torsion_minimised_dataset(
                mm_dataset=dataset,
                ml_simulation=MagicMock(),
                mm_simulation=MagicMock(),
            )

        # Should return datasets with 0 conformations
        assert len(mm_result) == 1
        assert mm_result[0]["energy"].shape[0] == 0


class TestGenerateTorsionMinimisedDatasetIntegration:
    """Integration tests for generate_torsion_minimised_dataset."""

    def test_full_torsion_minimisation_workflow(self):
        """Test the full torsion minimisation workflow with mocked minimisation."""
        # Use butane which has a rotatable torsion
        mol = Molecule.from_smiles("CCCC")
        mol.generate_conformers(n_conformers=1)
        smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        n_atoms = mol.n_atoms

        # Create dataset with random coords
        coords = torch.rand(2, n_atoms, 3, dtype=torch.float64)
        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords,
            energy=torch.rand(2, dtype=torch.float64),
            forces=torch.rand(2, n_atoms, 3, dtype=torch.float64),
            energy_weight=1.0,
            forces_weight=1.0,
        )

        # Create mock simulations
        def create_mock_sim():
            sim = MagicMock()
            sim.system = openmm.System()
            for _ in range(n_atoms):
                sim.system.addParticle(12.0)
            sim.context = MagicMock()
            state = MagicMock()
            state.getPositions.return_value = omm_unit.Quantity(
                np.random.rand(n_atoms, 3), omm_unit.angstrom
            )
            state.getPotentialEnergy.return_value = omm_unit.Quantity(
                10.0, omm_unit.kilocalorie_per_mole
            )
            state.getForces.return_value = omm_unit.Quantity(
                np.random.rand(n_atoms, 3),
                omm_unit.kilocalorie_per_mole / omm_unit.angstrom,
            )
            sim.context.getState.return_value = state
            return sim

        ml_sim = create_mock_sim()
        mm_sim = create_mock_sim()

        # Mock _minimize_with_frozen_torsions to avoid the slow operation
        with patch("presto.sample._minimize_with_frozen_torsions") as mock_min:
            mock_min.return_value = (
                np.random.rand(n_atoms, 3),
                10.0,
                np.random.rand(n_atoms, 3),
            )

            mm_result, ml_result = generate_torsion_minimised_dataset(
                mm_dataset=dataset,
                ml_simulation=ml_sim,
                mm_simulation=mm_sim,
                torsion_restraint_force_constant=1000.0,
                ml_minimisation_steps=1,
                mm_minimisation_steps=1,
            )

        # Should return datasets with same number of conformations as input
        assert len(mm_result) == 1
        assert len(ml_result) == 1
        assert mm_result[0]["energy"].shape[0] == 2
        assert ml_result[0]["energy"].shape[0] == 2

    def test_saves_pdb_when_path_provided(self, tmp_path):
        """Test that PDB files are saved when paths are provided."""
        # Use butane - has rotatable bonds
        mol = Molecule.from_smiles("CCCC")
        mol.generate_conformers(n_conformers=1)
        smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        n_atoms = mol.n_atoms

        coords = torch.rand(1, n_atoms, 3, dtype=torch.float64)
        dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=coords,
            energy=torch.rand(1, dtype=torch.float64),
            forces=torch.rand(1, n_atoms, 3, dtype=torch.float64),
            energy_weight=1.0,
            forces_weight=1.0,
        )

        # Create mock simulations
        def create_mock_sim():
            sim = MagicMock()
            sim.system = openmm.System()
            for _ in range(n_atoms):
                sim.system.addParticle(12.0)
            sim.context = MagicMock()
            state = MagicMock()
            state.getPositions.return_value = omm_unit.Quantity(
                np.random.rand(n_atoms, 3), omm_unit.angstrom
            )
            state.getPotentialEnergy.return_value = omm_unit.Quantity(
                10.0, omm_unit.kilocalorie_per_mole
            )
            state.getForces.return_value = omm_unit.Quantity(
                np.random.rand(n_atoms, 3),
                omm_unit.kilocalorie_per_mole / omm_unit.angstrom,
            )
            sim.context.getState.return_value = state
            # Need a real topology for PDB writing
            sim.topology = mol.to_topology().to_openmm()
            return sim

        ml_sim = create_mock_sim()
        mm_sim = create_mock_sim()

        ml_pdb = tmp_path / "ml_min.pdb"
        mm_pdb = tmp_path / "mm_min.pdb"

        # Mock _minimize_with_frozen_torsions to avoid slow operation
        with patch("presto.sample._minimize_with_frozen_torsions") as mock_min:
            mock_min.return_value = (
                np.random.rand(n_atoms, 3),
                10.0,
                np.random.rand(n_atoms, 3),
            )

            generate_torsion_minimised_dataset(
                mm_dataset=dataset,
                ml_simulation=ml_sim,
                mm_simulation=mm_sim,
                ml_pdb_path=ml_pdb,
                mm_pdb_path=mm_pdb,
            )

        # Check PDB files were created
        assert ml_pdb.exists()
        assert mm_pdb.exists()


class TestRecalculateEnergiesAndForces:
    """Tests for recalculate_energies_and_forces function."""

    def test_recalculates_with_simulation(self, mock_simulation):
        """Test that function recalculates energies and forces using simulation."""
        # Create a simple dataset
        mol = Molecule.from_smiles("C")  # Methane
        smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        n_conf = 2
        n_atoms = mol.n_atoms

        # descent.targets.energy.create_dataset expects a different format
        import descent.targets.energy

        original_dataset = descent.targets.energy.create_dataset(
            [
                {
                    "smiles": smiles,
                    "coords": torch.rand(n_conf, n_atoms, 3),
                    "energy": torch.rand(n_conf),
                    "forces": torch.rand(n_conf, n_atoms, 3),
                }
            ]
        )

        result = recalculate_energies_and_forces(original_dataset, mock_simulation)

        assert len(result) == 1
        # Verify dataset entry has expected structure
        entry = result[0]
        assert "smiles" in entry
        assert "coords" in entry
        assert "energy" in entry
        assert "forces" in entry
        # Verify shapes match input
        assert len(entry["energy"]) == n_conf
        # Verify simulation was called for each conformation
        assert mock_simulation.context.setPositions.call_count == n_conf
        assert mock_simulation.context.getState.call_count == n_conf


class TestRunMdFunction:
    """Tests for _run_md helper function."""

    def test_run_md_basic_structure(self, mock_molecule, mock_simulation):
        """Test that _run_md returns correctly structured dataset."""
        step_fn = MagicMock()

        dataset = _run_md(
            mol=mock_molecule,
            simulation=mock_simulation,
            step_fn=step_fn,
            equilibration_n_steps_per_conformer=10,
            production_n_snapshots_per_conformer=2,
            production_n_steps_per_snapshot_per_conformer=5,
            pdb_reporter_path=None,
        )

        # Verify dataset structure
        assert len(dataset) == 1
        entry = dataset[0]
        assert "smiles" in entry
        assert "coords" in entry
        assert "energy" in entry
        assert "forces" in entry

        # Verify minimizeEnergy was called
        assert mock_simulation.minimizeEnergy.called
        # Verify step function was called
        assert step_fn.call_count >= 1

    def test_run_md_with_pdb_reporter(self, mock_molecule, mock_simulation, tmp_path):
        """Test that _run_md handles PDB reporter path."""
        step_fn = MagicMock()
        pdb_path = str(tmp_path / "trajectory.pdb")

        # Mock PDBReporter to avoid file writing issues
        with patch("presto.sample.PDBReporter") as mock_reporter:
            dataset = _run_md(
                mol=mock_molecule,
                simulation=mock_simulation,
                step_fn=step_fn,
                equilibration_n_steps_per_conformer=10,
                production_n_snapshots_per_conformer=1,
                production_n_steps_per_snapshot_per_conformer=5,
                pdb_reporter_path=pdb_path,
            )

        # Verify PDBReporter was created
        assert mock_reporter.called
        assert len(dataset) == 1


class TestSamplingFunctionsRegistry:
    """Tests for sampling function registry."""

    def test_all_sampling_functions_registered(self):
        """Test that all sampling settings have registered functions."""
        from presto import settings
        from presto.sample import _SAMPLING_FNS_REGISTRY

        expected_settings = [
            settings.MMMDSamplingSettings,
            settings.MLMDSamplingSettings,
            settings.MMMDMetadynamicsSamplingSettings,
            settings.MMMDMetadynamicsTorsionMinimisationSamplingSettings,
            settings.PreComputedDatasetSettings,
        ]

        for settings_cls in expected_settings:
            assert settings_cls in _SAMPLING_FNS_REGISTRY
            assert callable(_SAMPLING_FNS_REGISTRY[settings_cls])

    def test_registry_functions_have_correct_signature(self):
        """Test that registered functions have the expected signature."""
        import inspect

        from presto.sample import _SAMPLING_FNS_REGISTRY

        expected_params = {"mols", "off_ff", "device", "settings", "output_paths"}

        for _settings_cls, fn in _SAMPLING_FNS_REGISTRY.items():
            sig = inspect.signature(fn)
            actual_params = set(sig.parameters.keys())
            assert actual_params == expected_params, f"{fn.__name__} has wrong params"


class TestAddTorsionRestraintForcesWithParticles:
    """Tests for _add_torsion_restraint_forces with real particles."""

    def test_adds_forces_with_initial_angles(self):
        """Test adding torsion restraints with initial angles."""
        system = openmm.System()
        for _ in range(8):
            system.addParticle(12.0)

        sim = MagicMock()
        sim.system = system
        sim.context = MagicMock()

        torsion_indices = [(0, 1, 2, 3), (4, 5, 6, 7)]
        initial_angles = [0.5, 1.0]
        k = 100.0

        indices, group = _add_torsion_restraint_forces(
            sim, torsion_indices, k, initial_angles
        )

        assert len(indices) == 2
        assert sim.context.reinitialize.called


class TestMinimizeWithFrozenTorsions:
    """Tests for _minimize_with_frozen_torsions function."""

    def test_minimize_basic(self, mock_simulation):
        """Test basic minimization with frozen torsions."""
        from presto.sample import _minimize_with_frozen_torsions

        # Setup mock
        mock_simulation.system = MagicMock()
        mock_force = MagicMock()
        mock_simulation.system.getForce.return_value = mock_force
        mock_force.getTorsionParameters.return_value = [0, 1, 2, 3, [0.0, 0.0]]

        torsion_atoms_list = [(0, 1, 2, 3)]
        coords = np.zeros((9, 3))
        force_indices = [0]

        with (
            patch("presto.sample._calculate_torsion_angles") as mock_calc,
            patch("presto.sample._update_torsion_restraints") as mock_update,
        ):
            mock_calc.return_value = torch.tensor([0.0])

            result_coords, result_energy, result_forces = (
                _minimize_with_frozen_torsions(
                    mock_simulation,
                    coords,
                    torsion_atoms_list,
                    force_indices,
                    100.0,  # force constant
                    0,  # restraint force group
                    5,  # max iterations
                )
            )

        assert mock_simulation.minimizeEnergy.called
        assert mock_update.called


class TestSampleMmmdIntegration:
    """Integration tests for sample_mmmd function."""

    def test_sample_mmmd_single_molecule_minimal(self, tmp_path):
        """Test sample_mmmd with minimal settings to exercise full code path."""
        mol = Molecule.from_smiles("CC")  # Ethane - simple molecule
        mol.generate_conformers(n_conformers=1)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        settings_obj = MMMDSamplingSettings(
            sampling_protocol="mm_md",
            timestep=1.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            equilibration_sampling_time_per_conformer=0.002
            * omm_unit.picoseconds,  # 2 steps
            production_sampling_time_per_conformer=0.002
            * omm_unit.picoseconds,  # 2 steps
            snapshot_interval=0.001 * omm_unit.picoseconds,  # Every step
        )

        # PDB output is required by settings
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()
        output_paths = {OutputType.PDB_TRAJECTORY: pdb_dir}

        # Mock ML potential creation to avoid needing actual ML models
        with patch("presto.sample._get_ml_omm_system") as mock_ml_sys:
            # Create a mock system that returns a real OpenMM system
            mock_system = openmm.System()
            for _ in range(mol.n_atoms):
                mock_system.addParticle(12.0)
            # Add a simple force so it doesn't blow up
            force = openmm.CustomExternalForce("0")
            mock_system.addForce(force)
            mock_ml_sys.return_value = mock_system

            result = sample_mmmd(
                [mol], ff, torch.device("cpu"), settings_obj, output_paths
            )

        assert len(result) == 1
        assert isinstance(result[0], datasets.Dataset)
        # Verify weighted dataset structure
        entry = result[0][0]
        assert "smiles" in entry
        assert "coords" in entry
        assert "energy" in entry
        assert "forces" in entry
        assert "energy_weights" in entry
        assert "forces_weights" in entry
        # Verify weights are set
        assert len(entry["energy_weights"]) == len(entry["energy"])
        assert len(entry["forces_weights"]) == len(entry["energy"])

    def test_sample_mmmd_with_pdb_output(self, tmp_path):
        """Test sample_mmmd with PDB trajectory output."""
        mol = Molecule.from_smiles("C")  # Methane
        mol.generate_conformers(n_conformers=1)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        settings_obj = MMMDSamplingSettings(
            sampling_protocol="mm_md",
            timestep=1.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            equilibration_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            production_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            snapshot_interval=0.001 * omm_unit.picoseconds,
        )

        # Use a file path with .pdb extension so get_mol_path creates pdb_mol0.pdb
        pdb_base = tmp_path / "trajectory.pdb"
        output_paths = {OutputType.PDB_TRAJECTORY: pdb_base}

        with patch("presto.sample._get_ml_omm_system") as mock_ml_sys:
            mock_system = openmm.System()
            for _ in range(mol.n_atoms):
                mock_system.addParticle(12.0)
            force = openmm.CustomExternalForce("0")
            mock_system.addForce(force)
            mock_ml_sys.return_value = mock_system

            result = sample_mmmd(
                [mol], ff, torch.device("cpu"), settings_obj, output_paths
            )

        assert len(result) == 1
        assert isinstance(result[0], datasets.Dataset)
        # Verify dataset has weighted structure
        entry = result[0][0]
        assert "energy_weights" in entry
        assert "forces_weights" in entry
        # Verify PDB file was created for mol_0 (get_mol_path creates trajectory_mol0.pdb)
        pdb_file = tmp_path / "trajectory_mol0.pdb"
        assert pdb_file.exists()


class TestSampleMlmdIntegration:
    """Integration tests for sample_mlmd function."""

    def test_sample_mlmd_single_molecule_minimal(self, tmp_path):
        """Test sample_mlmd with minimal settings."""
        mol = Molecule.from_smiles("C")  # Methane
        mol.generate_conformers(n_conformers=1)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        settings_obj = MLMDSamplingSettings(
            sampling_protocol="ml_md",
            timestep=1.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            equilibration_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            production_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            snapshot_interval=0.001 * omm_unit.picoseconds,
        )

        # PDB output is required by settings
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()
        output_paths = {OutputType.PDB_TRAJECTORY: pdb_dir}

        with patch("presto.sample._get_ml_omm_system") as mock_ml_sys:
            mock_system = openmm.System()
            for _ in range(mol.n_atoms):
                mock_system.addParticle(12.0)
            force = openmm.CustomExternalForce("0")
            mock_system.addForce(force)
            mock_ml_sys.return_value = mock_system

            result = sample_mlmd(
                [mol], ff, torch.device("cpu"), settings_obj, output_paths
            )

        assert len(result) == 1
        assert isinstance(result[0], datasets.Dataset)
        # Verify weighted dataset structure
        entry = result[0][0]
        assert "smiles" in entry
        assert "coords" in entry
        assert "energy" in entry
        assert "forces" in entry
        assert "energy_weights" in entry
        assert "forces_weights" in entry

    def test_sample_mlmd_with_pdb_output(self, tmp_path):
        """Test sample_mlmd with PDB trajectory output."""
        mol = Molecule.from_smiles("C")
        mol.generate_conformers(n_conformers=1)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        settings_obj = MLMDSamplingSettings(
            sampling_protocol="ml_md",
            timestep=1.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            equilibration_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            production_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            snapshot_interval=0.001 * omm_unit.picoseconds,
        )

        # Use a file path with .pdb extension so get_mol_path creates trajectory_mol0.pdb
        pdb_base = tmp_path / "trajectory.pdb"
        output_paths = {OutputType.PDB_TRAJECTORY: pdb_base}

        with patch("presto.sample._get_ml_omm_system") as mock_ml_sys:
            mock_system = openmm.System()
            for _ in range(mol.n_atoms):
                mock_system.addParticle(12.0)
            force = openmm.CustomExternalForce("0")
            mock_system.addForce(force)
            mock_ml_sys.return_value = mock_system

            result = sample_mlmd(
                [mol], ff, torch.device("cpu"), settings_obj, output_paths
            )

        assert len(result) == 1
        assert isinstance(result[0], datasets.Dataset)
        # Verify dataset has weighted structure
        entry = result[0][0]
        assert "energy_weights" in entry
        assert "forces_weights" in entry
        # Verify PDB file was created (get_mol_path creates trajectory_mol0.pdb)
        pdb_file = tmp_path / "trajectory_mol0.pdb"
        assert pdb_file.exists()


class TestSampleMmmdMetadynamicsIntegration:
    """Integration tests for sample_mmmd_metadynamics function."""

    def test_sample_metadynamics_with_rotatable_bonds(self, tmp_path):
        """Test metadynamics sampling for molecule with rotatable bonds."""
        mol = Molecule.from_smiles("CCCC")  # Butane - has rotatable bonds
        mol.generate_conformers(n_conformers=1)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        settings_obj = MMMDMetadynamicsSamplingSettings(
            sampling_protocol="mm_md_metadynamics",
            timestep=1.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            bias_frequency=0.001 * omm_unit.picoseconds,
            bias_save_frequency=0.001 * omm_unit.picoseconds,
            bias_height=0.5 * omm_unit.kilojoules_per_mole,
            equilibration_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            production_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            snapshot_interval=0.001 * omm_unit.picoseconds,
        )

        bias_dir = tmp_path / "bias"
        bias_dir.mkdir()
        output_paths = {
            OutputType.PDB_TRAJECTORY: tmp_path,
            OutputType.METADYNAMICS_BIAS: bias_dir,
        }

        with patch("presto.sample._get_ml_omm_system") as mock_ml_sys:
            mock_system = openmm.System()
            for _ in range(mol.n_atoms):
                mock_system.addParticle(12.0)
            force = openmm.CustomExternalForce("0")
            mock_system.addForce(force)
            mock_ml_sys.return_value = mock_system

            result = sample_mmmd_metadynamics(
                [mol], ff, torch.device("cpu"), settings_obj, output_paths
            )

        assert len(result) == 1
        assert isinstance(result[0], datasets.Dataset)
        # Verify weighted dataset structure with all required fields
        entry = result[0][0]
        assert "smiles" in entry
        assert "coords" in entry
        assert "energy" in entry
        assert "forces" in entry
        assert "energy_weights" in entry
        assert "forces_weights" in entry
        # Verify bias directory was used (files created by metadynamics)
        assert bias_dir.exists()

    def test_sample_metadynamics_fallback_no_rotatable_bonds(self, tmp_path):
        """Test metadynamics falls back to regular MD for molecule without rotatable bonds."""
        mol = Molecule.from_smiles("C")  # Methane - no rotatable bonds
        mol.generate_conformers(n_conformers=1)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        settings_obj = MMMDMetadynamicsSamplingSettings(
            sampling_protocol="mm_md_metadynamics",
            timestep=1.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            bias_frequency=0.001 * omm_unit.picoseconds,
            bias_save_frequency=0.001 * omm_unit.picoseconds,
            bias_height=0.5 * omm_unit.kilojoules_per_mole,
            equilibration_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            production_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            snapshot_interval=0.001 * omm_unit.picoseconds,
        )

        bias_dir = tmp_path / "bias"
        bias_dir.mkdir()
        output_paths = {
            OutputType.PDB_TRAJECTORY: tmp_path,
            OutputType.METADYNAMICS_BIAS: bias_dir,
        }

        with patch("presto.sample._get_ml_omm_system") as mock_ml_sys:
            mock_system = openmm.System()
            for _ in range(mol.n_atoms):
                mock_system.addParticle(12.0)
            force = openmm.CustomExternalForce("0")
            mock_system.addForce(force)
            mock_ml_sys.return_value = mock_system

            result = sample_mmmd_metadynamics(
                [mol], ff, torch.device("cpu"), settings_obj, output_paths
            )

        assert len(result) == 1
        assert isinstance(result[0], datasets.Dataset)
        # Verify weighted dataset structure (should still work even with fallback to regular MD)
        entry = result[0][0]
        assert "smiles" in entry
        assert "energy" in entry
        assert "forces" in entry
        assert "energy_weights" in entry
        assert "forces_weights" in entry


class TestSampleMmmdMetadynamicsTorsionMinIntegration:
    """Integration tests for sample_mmmd_metadynamics_with_torsion_minimisation."""

    def test_with_rotatable_bonds(self, tmp_path):
        """Test torsion minimisation workflow with rotatable bonds."""
        mol = Molecule.from_smiles("CCCC")  # Butane
        mol.generate_conformers(n_conformers=1)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        settings_obj = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            sampling_protocol="mm_md_metadynamics_torsion_minimisation",
            timestep=1.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            bias_frequency=0.001 * omm_unit.picoseconds,
            bias_save_frequency=0.001 * omm_unit.picoseconds,
            bias_height=0.5 * omm_unit.kilojoules_per_mole,
            equilibration_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            production_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            snapshot_interval=0.001 * omm_unit.picoseconds,
            ml_minimisation_steps=1,
            mm_minimisation_steps=1,
        )

        bias_dir = tmp_path / "bias"
        bias_dir.mkdir()
        output_paths = {
            OutputType.PDB_TRAJECTORY: tmp_path,
            OutputType.METADYNAMICS_BIAS: bias_dir,
            OutputType.ML_MINIMISED_PDB: tmp_path / "ml_min",
            OutputType.MM_MINIMISED_PDB: tmp_path / "mm_min",
        }
        (tmp_path / "ml_min").mkdir()
        (tmp_path / "mm_min").mkdir()

        with patch("presto.sample._get_ml_omm_system") as mock_ml_sys:

            def create_mock_system(*args, **kwargs):
                mock_system = openmm.System()
                for _ in range(mol.n_atoms):
                    mock_system.addParticle(12.0)
                force = openmm.CustomExternalForce("0")
                mock_system.addForce(force)
                return mock_system

            mock_ml_sys.side_effect = create_mock_system

            result = sample_mmmd_metadynamics_with_torsion_minimisation(
                [mol], ff, torch.device("cpu"), settings_obj, output_paths
            )

        assert len(result) == 1
        assert isinstance(result[0], datasets.Dataset)
        # Verify weighted dataset structure with all required fields
        entry = result[0][0]
        assert "smiles" in entry
        assert "coords" in entry
        assert "energy" in entry
        assert "forces" in entry
        assert "energy_weights" in entry
        assert "forces_weights" in entry

    def test_fallback_no_rotatable_bonds(self, tmp_path):
        """Test fallback to regular MD when molecule has no rotatable bonds."""
        mol = Molecule.from_smiles("C")  # Methane - no rotatable bonds
        mol.generate_conformers(n_conformers=1)

        ff = ForceField("openff_unconstrained-2.3.0.offxml")

        settings_obj = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            sampling_protocol="mm_md_metadynamics_torsion_minimisation",
            timestep=1.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            bias_frequency=0.001 * omm_unit.picoseconds,
            bias_save_frequency=0.001 * omm_unit.picoseconds,
            bias_height=0.5 * omm_unit.kilojoules_per_mole,
            equilibration_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            production_sampling_time_per_conformer=0.001 * omm_unit.picoseconds,
            snapshot_interval=0.001 * omm_unit.picoseconds,
        )

        bias_dir = tmp_path / "bias"
        bias_dir.mkdir()
        output_paths = {
            OutputType.PDB_TRAJECTORY: tmp_path,
            OutputType.METADYNAMICS_BIAS: bias_dir,
            OutputType.ML_MINIMISED_PDB: tmp_path / "ml_min",
            OutputType.MM_MINIMISED_PDB: tmp_path / "mm_min",
        }
        (tmp_path / "ml_min").mkdir()
        (tmp_path / "mm_min").mkdir()

        with patch("presto.sample._get_ml_omm_system") as mock_ml_sys:
            mock_system = openmm.System()
            for _ in range(mol.n_atoms):
                mock_system.addParticle(12.0)
            force = openmm.CustomExternalForce("0")
            mock_system.addForce(force)
            mock_ml_sys.return_value = mock_system

            result = sample_mmmd_metadynamics_with_torsion_minimisation(
                [mol], ff, torch.device("cpu"), settings_obj, output_paths
            )

        assert len(result) == 1
        assert isinstance(result[0], datasets.Dataset)
        # Verify weighted dataset structure (should still work even with fallback)
        entry = result[0][0]
        assert "smiles" in entry
        assert "energy" in entry
        assert "forces" in entry
        assert "energy_weights" in entry
        assert "forces_weights" in entry
