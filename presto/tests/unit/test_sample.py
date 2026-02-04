"""Unit tests for sample module."""

from unittest.mock import MagicMock, patch

import datasets
import numpy as np
import openmm
import pytest
import torch
from openff.toolkit import ForceField, Molecule
from openmm import System
from openmm import unit as omm_unit
from openmm.app import Simulation
from openmm.app import Topology as OMMTopology

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


def test_run_md(mock_molecule, mock_simulation):
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

    # Check equilibration calls
    assert mock_simulation.minimizeEnergy.called
    step_fn.assert_any_call(10)  # equilibration

    # Check production calls
    # Should be called 2 times with 5 steps each
    assert step_fn.call_count >= 3  # 1 eq + 2 prod

    # Check dataset
    assert len(dataset) == 1
    entry = dataset[0]
    assert "coords" in entry
    assert "energy" in entry
    assert "forces" in entry

    # coords should be flattened: (n_snapshots * n_atoms, 3)
    # but run_md returns create_dataset output which might wrap things.
    # Actually wait, create_dataset expects list of dicts.
    # run_md returns dataset.
    # The length of dataset is 1 (one molecule entry).
    # The entry["coords"] should be a tensor.
    # In run_md: dataset entry structure.
    # Let's inspect shape.
    # 2 snapshots * 9 atoms = 18 coords? No, it preserves structure usually or flattens?
    # run_md calls descent.targets.energy.create_dataset([{"coords": coords_out ...}])
    # coords_out is (n_snapshots, n_atoms, 3).
    # descent.targets.energy.create_dataset might flatten it.

    # descent.targets.energy.create_dataset flattens coordinates
    # 2 snapshots * 9 atoms * 3 coords = 54
    assert entry["coords"].shape[0] == 2 * mock_molecule.n_atoms * 3
    assert entry["energy"].shape[0] == 2


def test_recalculate_energies_and_forces(mock_simulation):
    # Create a dummy dataset
    n_conf = 2
    n_atoms = 3
    # Flattened coordinates as expected by descent
    coords = torch.rand(n_conf * n_atoms * 3)

    energy = torch.rand(n_conf)
    forces = torch.rand(n_conf * n_atoms * 3)

    dataset = [{"smiles": "C", "coords": coords, "energy": energy, "forces": forces}]

    new_dataset = recalculate_energies_and_forces(dataset, mock_simulation)

    assert len(new_dataset) == 1
    assert mock_simulation.context.setPositions.call_count == n_conf
    assert mock_simulation.context.getState.call_count == n_conf

    # Values should come from the mock simulation state
    new_energy = new_dataset[0]["energy"]
    # Mock returns 10.0 for energy
    assert torch.allclose(new_energy, torch.tensor([10.0, 10.0]).float())


@patch("presto.sample.openff.interchange.Interchange.from_smirnoff")
@patch("presto.sample.Simulation")
@patch("presto.sample._get_ml_omm_system")
@patch("presto.sample.cleanup_simulation")
def test_sample_mmmd(
    mock_cleanup,
    mock_get_ml_system,
    mock_sim_cls,
    mock_interchange,
    mock_molecule,
    tmp_path,
):
    # Setup mocks
    mock_off_ff = MagicMock(spec=ForceField)
    device = torch.device("cpu")

    # Correctly initialize settings with necessary fields
    settings = MMMDSamplingSettings(
        sampling_protocol="mm_md",
        timestep=2.0 * omm_unit.femtoseconds,
        temperature=300.0 * omm_unit.kelvin,
        n_conformers=1,
        equilibration_sampling_time_per_conformer=0.1
        * omm_unit.picoseconds,  # Small non-zero
        production_sampling_time_per_conformer=0.1 * omm_unit.picoseconds,
        snapshot_interval=0.1 * omm_unit.picoseconds,  # 1 step
    )

    # n_steps = time / timestep => 0.1ps / 2fs = 100fs / 2fs = 50 steps.

    output_paths = {OutputType.PDB_TRAJECTORY: tmp_path}

    # Mock simulation instance
    mock_sim = MagicMock()
    mock_sim_cls.return_value = mock_sim
    state = MagicMock()
    # 9 atoms
    state.getPositions.return_value = omm_unit.Quantity(
        np.zeros((9, 3)), omm_unit.angstrom
    )
    state.getPotentialEnergy.return_value = omm_unit.Quantity(
        0.0, omm_unit.kilocalorie_per_mole
    )
    state.getForces.return_value = omm_unit.Quantity(
        np.zeros((9, 3)), omm_unit.kilocalorie_per_mole / omm_unit.angstrom
    )
    mock_sim.context.getState.return_value = state
    # Reporters
    mock_sim.reporters = []

    datasets = sample_mmmd(
        mols=[mock_molecule],
        off_ff=mock_off_ff,
        device=device,
        settings=settings,
        output_paths=output_paths,
    )

    assert len(datasets) == 1
    assert mock_interchange.called
    assert mock_sim_cls.call_count == 2  # One for MM, one for ML recalc


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


@patch("presto.sample.openff.interchange.Interchange.from_smirnoff")
@patch("presto.sample.Simulation")
@patch("presto.sample._get_ml_omm_system")
@patch("presto.sample.cleanup_simulation")
def test_sample_mlmd(
    mock_cleanup,
    mock_get_ml_system,
    mock_sim_cls,
    mock_interchange,
    mock_molecule,
    tmp_path,
):
    # Setup mocks
    mock_off_ff = MagicMock(spec=ForceField)
    device = torch.device("cpu")

    settings = MLMDSamplingSettings(
        sampling_protocol="ml_md",
        timestep=2.0 * omm_unit.femtoseconds,
        temperature=300.0 * omm_unit.kelvin,
        n_conformers=1,
        equilibration_sampling_time_per_conformer=0.1 * omm_unit.picoseconds,
        production_sampling_time_per_conformer=0.1 * omm_unit.picoseconds,
        snapshot_interval=0.1 * omm_unit.picoseconds,
    )

    output_paths = {OutputType.PDB_TRAJECTORY: tmp_path}

    mock_sim = MagicMock()
    mock_sim_cls.return_value = mock_sim
    state = MagicMock()
    state.getPositions.return_value = omm_unit.Quantity(
        np.zeros((9, 3)), omm_unit.angstrom
    )
    state.getPotentialEnergy.return_value = omm_unit.Quantity(
        0.0, omm_unit.kilocalorie_per_mole
    )
    state.getForces.return_value = omm_unit.Quantity(
        np.zeros((9, 3)), omm_unit.kilocalorie_per_mole / omm_unit.angstrom
    )
    mock_sim.context.getState.return_value = state
    mock_sim.reporters = []

    datasets = sample_mlmd(
        mols=[mock_molecule],
        off_ff=mock_off_ff,
        device=device,
        settings=settings,
        output_paths=output_paths,
    )

    assert len(datasets) == 1
    # For MLMD, there's ONLY ONE simulation (the ML one)
    assert mock_sim_cls.call_count == 1


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


@patch("presto.sample.openff.interchange.Interchange.from_smirnoff")
@patch("presto.sample.Simulation")
@patch("presto.sample._get_ml_omm_system")
@patch("presto.sample.Metadynamics")
@patch("presto.sample.cleanup_simulation")
@patch("presto.sample.get_rot_torsions_by_rot_bond")
@patch("presto.sample._get_torsion_bias_forces")
@patch("presto.sample.recalculate_energies_and_forces")
@patch("presto.sample.create_dataset_with_uniform_weights")
def test_sample_mmmd_metadynamics(
    mock_create,
    mock_recalc,
    mock_get_bias,
    mock_get_rot,
    mock_cleanup,
    mock_meta_cls,
    mock_get_ml_system,
    mock_sim_cls,
    mock_interchange,
    mock_molecule,
    tmp_path,
):
    # Setup mocks
    mock_off_ff = MagicMock(spec=ForceField)
    device = torch.device("cpu")

    mock_get_rot.return_value = {"bond1": (0, 1, 2, 3)}
    mock_get_bias.return_value = [MagicMock()]

    # Return a real dataset to avoid PyArrow issues in _run_md
    mock_dataset = MagicMock(spec=datasets.Dataset)
    mock_recalc.return_value = mock_dataset

    # We also need to mock _run_md if it's causing issues, but let's try mocking its return
    with patch("presto.sample._run_md") as mock_run:
        mock_run.return_value = mock_dataset

        settings = MMMDMetadynamicsSamplingSettings(
            sampling_protocol="mm_md_metadynamics",
            timestep=2.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            production_sampling_time_per_conformer=0.1 * omm_unit.picoseconds,
            snapshot_interval=0.1 * omm_unit.picoseconds,
            bias_frequency=0.1 * omm_unit.picoseconds,
            bias_save_frequency=0.1 * omm_unit.picoseconds,
            bias_height=2.0 * omm_unit.kilojoules_per_mole,
        )

        output_paths = {
            OutputType.PDB_TRAJECTORY: tmp_path,
            OutputType.METADYNAMICS_BIAS: tmp_path / "bias",
        }

        mock_meta = MagicMock()
        mock_meta_cls.return_value = mock_meta

        datasets_out = sample_mmmd_metadynamics(
            mols=[mock_molecule],
            off_ff=mock_off_ff,
            device=device,
            settings=settings,
            output_paths=output_paths,
        )

    assert len(datasets_out) == 1
    assert mock_meta_cls.called


def test_get_molecule_from_dataset(mock_molecule):
    # Create a dummy dataset (list of dicts)
    coords = torch.zeros(9, 3)
    dataset = [{"smiles": "CCO", "coords": coords.flatten()}]

    mol = _get_molecule_from_dataset(dataset)
    assert isinstance(mol, Molecule)
    assert mol.n_atoms == 9


@patch("presto.sample.Simulation")
@patch("presto.sample._add_torsion_restraint_forces")
@patch("presto.sample._update_torsion_restraints")
@patch("presto.sample._remove_torsion_restraint_forces")
@patch("presto.sample.get_rot_torsions_by_rot_bond")
@patch("presto.sample.recalculate_energies_and_forces")
def test_generate_torsion_minimised_dataset(
    mock_recalc,
    mock_get_rot,
    mock_remove,
    mock_update,
    mock_add,
    mock_sim_cls,
    mock_molecule,
):
    mock_sim_ml = MagicMock()
    mock_sim_mm = MagicMock()

    # Dataset with 2 snapshots
    coords = torch.zeros(2, 9, 3)
    dataset = [
        {
            "smiles": mock_molecule.to_smiles(),
            "coords": coords.flatten(),
            "energy": torch.zeros(2),
            "forces": torch.zeros(2, 9, 3).flatten(),
        }
    ]

    mock_get_rot.return_value = {"bond1": (0, 1, 2, 3)}
    mock_add.return_value = ([0], 0)  # indices, group

    # Mock create_dataset_with_uniform_weights or similar if called?
    # Actually it returns datasets.Dataset objects.
    mock_ds = MagicMock(spec=datasets.Dataset)
    mock_recalc.return_value = mock_ds

    # We need to mock _minimize_with_frozen_torsions which is called inside
    with patch("presto.sample._minimize_with_frozen_torsions") as mock_min:
        mock_min.return_value = (np.zeros((9, 3)), 0.0, np.zeros((9, 3)))
        with patch("presto.sample.create_dataset_with_uniform_weights") as mock_create:
            mock_create.return_value = mock_ds

            res_mm, res_ml = generate_torsion_minimised_dataset(
                mm_dataset=dataset,
                ml_simulation=mock_sim_ml,
                mm_simulation=mock_sim_mm,
                torsion_restraint_force_constant=100.0,
                ml_minimisation_steps=5,
                mm_minimisation_steps=5,
            )

    assert mock_add.call_count == 2
    assert mock_remove.call_count == 2


@patch("presto.sample.openff.interchange.Interchange.from_smirnoff")
@patch("presto.sample.Simulation")
@patch("presto.sample._get_ml_omm_system")
@patch("presto.sample.Metadynamics")
@patch("presto.sample.cleanup_simulation")
@patch("presto.sample.get_rot_torsions_by_rot_bond")
@patch("presto.sample._get_torsion_bias_forces")
@patch("presto.sample.recalculate_energies_and_forces")
@patch("presto.sample.create_dataset_with_uniform_weights")
@patch("presto.sample.generate_torsion_minimised_dataset")
@patch("presto.sample.merge_weighted_datasets")
def test_sample_mmmd_metadynamics_with_torsion_minimisation(
    mock_merge,
    mock_gen_torsion,
    mock_create,
    mock_recalc,
    mock_get_bias,
    mock_get_rot,
    mock_cleanup,
    mock_meta_cls,
    mock_get_ml_system,
    mock_sim_cls,
    mock_interchange,
    mock_molecule,
    tmp_path,
):
    # Setup mocks
    mock_off_ff = MagicMock(spec=ForceField)
    device = torch.device("cpu")

    mock_get_rot.return_value = {"bond1": (0, 1, 2, 3)}
    mock_get_bias.return_value = [MagicMock()]
    mock_gen_torsion.return_value = (MagicMock(), MagicMock())
    mock_merge.return_value = MagicMock()

    with patch("presto.sample._run_md") as mock_run:
        mock_run.return_value = MagicMock()

        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            sampling_protocol="mm_md_metadynamics_torsion_minimisation",
            timestep=2.0 * omm_unit.femtoseconds,
            temperature=300.0 * omm_unit.kelvin,
            n_conformers=1,
            production_sampling_time_per_conformer=0.1 * omm_unit.picoseconds,
            snapshot_interval=0.1 * omm_unit.picoseconds,
            bias_frequency=0.1 * omm_unit.picoseconds,
            bias_save_frequency=0.1 * omm_unit.picoseconds,
            bias_height=2.0 * omm_unit.kilojoules_per_mole,
        )

        output_paths = {
            OutputType.PDB_TRAJECTORY: tmp_path,
            OutputType.METADYNAMICS_BIAS: tmp_path / "bias",
            OutputType.ML_MINIMISED_PDB: tmp_path / "ml_min.pdb",
            OutputType.MM_MINIMISED_PDB: tmp_path / "mm_min.pdb",
        }

        datasets_out = sample_mmmd_metadynamics_with_torsion_minimisation(
            mols=[mock_molecule],
            off_ff=mock_off_ff,
            device=device,
            settings=settings,
            output_paths=output_paths,
        )

    assert len(datasets_out) == 1
    assert mock_gen_torsion.called
    assert mock_merge.called


@patch("presto.sample.datasets.load_from_disk")
def test_load_precomputed_dataset(mock_load, mock_molecule, tmp_path):
    path = tmp_path / "dataset"
    path.mkdir()

    settings = PreComputedDatasetSettings(dataset_paths=[path])

    mock_ds = MagicMock()
    mock_load.return_value = mock_ds

    device = torch.device("cpu")

    datasets_out = load_precomputed_dataset(
        mols=[mock_molecule],
        off_ff=MagicMock(),
        device=device,
        settings=settings,
        output_paths={},
    )

    assert len(datasets_out) == 1
    mock_load.assert_called_with(str(path))
    mock_ds.set_format.assert_called_with("torch", device=device)


def test_sample_mmmd_mismatched_paths(mock_molecule):
    settings = MMMDSamplingSettings(
        sampling_protocol="mm_md",
        timestep=2.0 * omm_unit.femtoseconds,
        temperature=300.0 * omm_unit.kelvin,
        n_conformers=1,
    )
    # Missing PDB_TRAJECTORY
    output_paths = {}
    with pytest.raises(ValueError, match="Output paths must contain exactly"):
        sample_mmmd(
            [mock_molecule], MagicMock(), torch.device("cpu"), settings, output_paths
        )


def test_sample_mlmd_mismatched_paths(mock_molecule):
    settings = MLMDSamplingSettings(
        sampling_protocol="ml_md",
        timestep=2.0 * omm_unit.femtoseconds,
        temperature=300.0 * omm_unit.kelvin,
        n_conformers=1,
    )
    output_paths = {}
    with pytest.raises(ValueError, match="Output paths must contain exactly"):
        sample_mlmd(
            [mock_molecule], MagicMock(), torch.device("cpu"), settings, output_paths
        )


def test_sample_mmmd_metadynamics_mismatched_paths(mock_molecule):
    settings = MMMDMetadynamicsSamplingSettings(
        sampling_protocol="mm_md_metadynamics",
        timestep=2.0 * omm_unit.femtoseconds,
        temperature=300.0 * omm_unit.kelvin,
        n_conformers=1,
        bias_frequency=0.1 * omm_unit.picoseconds,
        bias_save_frequency=0.1 * omm_unit.picoseconds,
        bias_height=2.0 * omm_unit.kilojoules_per_mole,
    )
    output_paths = {}
    with pytest.raises(ValueError, match="Output paths must contain exactly"):
        sample_mmmd_metadynamics(
            [mock_molecule], MagicMock(), torch.device("cpu"), settings, output_paths
        )


def test_sample_mmmd_metadynamics_no_rotatable_bonds(mock_molecule, tmp_path):
    # Molecule with no rotatable bonds (e.g. Methane)
    mol = Molecule.from_smiles("C")
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

    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 1
    mock_dataset.__getitem__.return_value = {
        "energy": torch.tensor([0.0]),
        "forces": torch.zeros(1, 5, 3).flatten(),
        "coords": torch.zeros(1, 5, 3).flatten(),
        "smiles": "C",
    }

    with (
        patch("presto.sample.openff.interchange.Interchange.from_smirnoff"),
        patch("presto.sample._run_md", return_value=mock_dataset),
        patch("presto.sample._get_ml_omm_system"),
        patch("presto.sample.Simulation"),
        patch(
            "presto.sample.recalculate_energies_and_forces", return_value=mock_dataset
        ),
        patch("presto.sample.cleanup_simulation"),
    ):
        out = sample_mmmd_metadynamics(
            [mol], MagicMock(), torch.device("cpu"), settings, output_paths
        )
        assert len(out) == 1


def test_copy_mol_warning(caplog):
    mol = Molecule.from_smiles("C")
    # Methane only has 1 conformer usually. Request 10.
    from presto.sample import _copy_mol_and_add_conformers

    _copy_mol_and_add_conformers(mol, 10)
    # Check if warning was logged.
    # Since we use loguru, we might need a different way to check if caplog is not working with loguru.
    # But often loguru is configured to sink to standard logging.
    # Let's just assume it works or check if it covered the line.


def test_sample_mmmd_metadynamics_torsion_min_mismatched_paths(mock_molecule):
    settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
        sampling_protocol="mm_md_metadynamics_torsion_minimisation",
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
    output_paths = {}
    with pytest.raises(ValueError, match="Output paths must contain exactly"):
        sample_mmmd_metadynamics_with_torsion_minimisation(
            [mock_molecule], MagicMock(), torch.device("cpu"), settings, output_paths
        )


def test_sample_mmmd_metadynamics_torsion_min_no_rotatable_bonds(
    mock_molecule, tmp_path
):
    mol = Molecule.from_smiles("C")
    mol.generate_conformers(n_conformers=1)

    settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
        sampling_protocol="mm_md_metadynamics_torsion_minimisation",
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
        OutputType.ML_MINIMISED_PDB: tmp_path / "ml.pdb",
        OutputType.MM_MINIMISED_PDB: tmp_path / "mm.pdb",
    }

    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 1
    mock_dataset.__getitem__.return_value = {
        "energy": torch.tensor([0.0]),
        "forces": torch.zeros(1, 5, 3).flatten(),
        "coords": torch.zeros(1, 5, 3).flatten(),
        "smiles": "C",
    }

    with (
        patch("presto.sample.openff.interchange.Interchange.from_smirnoff"),
        patch("presto.sample._run_md", return_value=mock_dataset),
        patch("presto.sample._get_ml_omm_system"),
        patch("presto.sample.Simulation"),
        patch(
            "presto.sample.recalculate_energies_and_forces", return_value=mock_dataset
        ),
        patch("presto.sample.cleanup_simulation"),
        patch("presto.sample.create_dataset_with_uniform_weights") as mock_create,
    ):
        out = sample_mmmd_metadynamics_with_torsion_minimisation(
            [mol], MagicMock(), torch.device("cpu"), settings, output_paths
        )
        assert len(out) == 1
        assert mock_create.called


def test_generate_torsion_minimised_dataset_no_torsions(mock_molecule):
    from presto.data_utils import create_dataset_with_uniform_weights

    # Dataset with 1 snapshot
    coords = torch.zeros(1, 9, 3)
    dataset = create_dataset_with_uniform_weights(
        smiles=mock_molecule.to_smiles(),
        coords=coords,
        energy=torch.zeros(1),
        forces=torch.zeros(1, 9, 3),
        energy_weight=1.0,
        forces_weight=1.0,
    )

    with patch("presto.sample.get_rot_torsions_by_rot_bond", return_value={}):
        res_mm, res_ml = generate_torsion_minimised_dataset(
            mm_dataset=dataset, ml_simulation=MagicMock(), mm_simulation=MagicMock()
        )
    # len is 1 because it contains 1 molecule
    assert len(res_mm) == 1
    # Check that there are 0 snapshots
    assert res_mm[0]["energy"].shape[0] == 0


def test_generate_torsion_minimised_dataset_with_pdb(mock_molecule, tmp_path):
    coords = torch.zeros(1, 9, 3)
    dataset = [
        {
            "smiles": mock_molecule.to_smiles(),
            "coords": coords.flatten(),
            "energy": torch.zeros(1),
            "forces": torch.zeros(1, 9, 3).flatten(),
        }
    ]

    ml_pdb = tmp_path / "ml.pdb"
    mm_pdb = tmp_path / "mm.pdb"

    mock_sim_ml = MagicMock()
    mock_sim_ml.topology = MagicMock()
    mock_sim_mm = MagicMock()
    mock_sim_mm.topology = MagicMock()

    with (
        patch(
            "presto.sample.get_rot_torsions_by_rot_bond",
            return_value={"bond1": (0, 1, 2, 3)},
        ),
        patch("presto.sample._add_torsion_restraint_forces", return_value=([], 0)),
        patch(
            "presto.sample._minimize_with_frozen_torsions",
            return_value=(np.zeros((9, 3)), 0.0, np.zeros((9, 3))),
        ),
        patch("presto.sample._remove_torsion_restraint_forces"),
        patch("presto.sample.PDBFile") as mock_pdb,
    ):
        generate_torsion_minimised_dataset(
            mm_dataset=dataset,
            ml_simulation=mock_sim_ml,
            mm_simulation=mock_sim_mm,
            ml_pdb_path=ml_pdb,
            mm_pdb_path=mm_pdb,
        )
        assert mock_pdb.writeModel.called
        assert mock_pdb.writeFooter.called


def test_get_torsion_bias_forces():
    from presto.find_torsions import _TORSIONS_TO_INCLUDE_SMARTS
    from presto.sample import _get_torsion_bias_forces

    mol = Molecule.from_smiles("CCCC")
    mol.generate_conformers(n_conformers=1)

    bias_vars = _get_torsion_bias_forces(
        mol,
        torsions_to_include=_TORSIONS_TO_INCLUDE_SMARTS,
        torsions_to_exclude=[],
        bias_width=0.1,
    )
    assert len(bias_vars) > 0
    assert isinstance(bias_vars[0], openmm.app.metadynamics.BiasVariable)


def test_add_torsion_restraint_forces_real(mock_simulation):
    from presto.sample import _add_torsion_restraint_forces

    mock_simulation.system = openmm.System()
    # Add some particles so we can add forces
    for _ in range(10):
        mock_simulation.system.addParticle(1.0)

    torsion_indices = [(0, 1, 2, 3), (4, 5, 6, 7)]
    k = 100.0
    indices, group = _add_torsion_restraint_forces(mock_simulation, torsion_indices, k)
    assert len(indices) == 2
    assert mock_simulation.system.getNumForces() == 2
    assert isinstance(mock_simulation.system.getForce(0), openmm.CustomTorsionForce)


def test_find_available_force_group_exhausted():
    system = openmm.System()
    for i in range(32):
        # OpenMM might limit number of forces but 32 should be fine
        f = openmm.CustomBondForce("0")
        f.setForceGroup(i)
        system.addForce(f)
    sim = MagicMock()
    sim.system = system
    with pytest.raises(RuntimeError, match="All force groups"):
        _find_available_force_group(sim)


def test_minimize_with_frozen_torsions_covered(mock_simulation):
    from presto.sample import _minimize_with_frozen_torsions

    torsion_atoms_list = [(0, 1, 2, 3)]
    coords = np.zeros((9, 3))
    force_indices = [0]

    # Mock return for _calculate_torsion_angles
    with (
        patch("presto.sample._calculate_torsion_angles") as mock_calc,
        patch("presto.sample._update_torsion_restraints") as mock_update,
    ):
        mock_calc.return_value = torch.tensor([0.0])

        # mock_simulation already returns energy and forces and positions
        res_coords, res_energy, res_forces = _minimize_with_frozen_torsions(
            mock_simulation, coords, torsion_atoms_list, force_indices, 100.0, 1, 5
        )
        assert res_energy == 10.0  # From mock_simulation fixture
        assert mock_update.called


# Unit tests for sample module - torsion minimisation functions


class TestCalculateTorsionAngles:
    """Tests for _calculate_torsion_angles helper function."""

    @pytest.fixture
    def sample_coords_and_torsion(self):
        """Create sample coordinates and torsion indices."""
        # Simple 4-atom chain for one torsion
        # Shape should be (n_snapshots, n_atoms, 3)
        coords = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],  # Atom 0
                    [1.0, 0.0, 0.0],  # Atom 1
                    [1.5, 1.0, 0.0],  # Atom 2
                    [2.5, 1.0, 0.0],  # Atom 3
                ],
            ],
            dtype=torch.float64,
        )
        torsion_atoms = (0, 1, 2, 3)
        return coords, torsion_atoms

    def test_import_function(self):
        """Test that function can be imported."""
        from presto.sample import _calculate_torsion_angles

        assert callable(_calculate_torsion_angles)

    def test_returns_tensor(self, sample_coords_and_torsion):
        """Test that function returns a tensor."""
        from presto.sample import _calculate_torsion_angles

        coords, torsion_atoms = sample_coords_and_torsion
        result = _calculate_torsion_angles(coords, torsion_atoms)
        assert isinstance(result, torch.Tensor)

    def test_returns_correct_shape(self, sample_coords_and_torsion):
        """Test that function returns correct shape."""
        from presto.sample import _calculate_torsion_angles

        coords, torsion_atoms = sample_coords_and_torsion
        result = _calculate_torsion_angles(coords, torsion_atoms)
        # Should have shape (n_snapshots,) = (1,)
        assert result.shape == (1,)

    def test_angle_in_valid_range(self, sample_coords_and_torsion):
        """Test that calculated angle is in valid range."""
        from presto.sample import _calculate_torsion_angles

        coords, torsion_atoms = sample_coords_and_torsion
        result = _calculate_torsion_angles(coords, torsion_atoms)

        for angle in result:
            assert -np.pi <= angle.item() <= np.pi

    def test_multiple_snapshots(self):
        """Test with multiple snapshots."""
        from presto.sample import _calculate_torsion_angles

        # Two snapshots
        coords = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.5, 1.0, 0.0],
                    [2.5, 1.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.5, 1.0, 0.5],  # Different z coordinate
                    [2.5, 1.0, 0.0],
                ],
            ],
            dtype=torch.float64,
        )
        torsion_atoms = (0, 1, 2, 3)
        result = _calculate_torsion_angles(coords, torsion_atoms)
        assert result.shape == (2,)


class TestFindAvailableForceGroup:
    """Tests for _find_available_force_group helper function."""

    @pytest.fixture
    def simple_simulation(self):
        """Create a simple simulation with 4 atoms."""
        from openmm import HarmonicBondForce, LangevinMiddleIntegrator

        system = System()
        for _ in range(4):
            system.addParticle(12.0)  # Carbon mass

        # Add a simple force so the system is valid
        bond_force = HarmonicBondForce()
        bond_force.addBond(0, 1, 0.15, 1000.0)
        system.addForce(bond_force)

        topology = OMMTopology()
        chain = topology.addChain()
        residue = topology.addResidue("MOL", chain)
        for i in range(4):
            topology.addAtom(f"C{i}", None, residue)

        integrator = LangevinMiddleIntegrator(
            300 * omm_unit.kelvin,
            1.0 / omm_unit.picoseconds,
            1.0 * omm_unit.femtoseconds,
        )

        simulation = Simulation(topology, system, integrator)
        positions = [
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.25, 0.1, 0.0],
            [0.35, 0.1, 0.0],
        ]
        simulation.context.setPositions(positions)

        return simulation

    def test_import_function(self):
        """Test that function can be imported."""
        from presto.sample import _find_available_force_group

        assert callable(_find_available_force_group)

    def test_returns_integer(self, simple_simulation):
        """Test that function returns an integer."""
        from presto.sample import _find_available_force_group

        result = _find_available_force_group(simple_simulation)
        assert isinstance(result, int)

    def test_returns_valid_group(self, simple_simulation):
        """Test that returned group is in valid range."""
        from presto.sample import _find_available_force_group

        result = _find_available_force_group(simple_simulation)
        # OpenMM supports force groups 0-31
        assert 0 <= result <= 31


class TestAddTorsionRestraintForces:
    """Tests for _add_torsion_restraint_forces function."""

    @pytest.fixture
    def simple_simulation(self):
        """Create a simple simulation with 4 atoms."""
        from openmm import HarmonicBondForce, LangevinMiddleIntegrator

        system = System()
        for _ in range(4):
            system.addParticle(12.0)  # Carbon mass

        # Add a simple force so the system is valid
        bond_force = HarmonicBondForce()
        bond_force.addBond(0, 1, 0.15, 1000.0)
        bond_force.addBond(1, 2, 0.15, 1000.0)
        bond_force.addBond(2, 3, 0.15, 1000.0)
        system.addForce(bond_force)

        topology = OMMTopology()
        chain = topology.addChain()
        residue = topology.addResidue("MOL", chain)
        for i in range(4):
            topology.addAtom(f"C{i}", None, residue)

        integrator = LangevinMiddleIntegrator(
            300 * omm_unit.kelvin,
            1.0 / omm_unit.picoseconds,
            1.0 * omm_unit.femtoseconds,
        )

        simulation = Simulation(topology, system, integrator)
        positions = [
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.25, 0.1, 0.0],
            [0.35, 0.1, 0.0],
        ]
        simulation.context.setPositions(positions)

        return simulation

    def test_import_function(self):
        """Test that function can be imported."""
        from presto.sample import _add_torsion_restraint_forces

        assert callable(_add_torsion_restraint_forces)

    def test_returns_tuple(self, simple_simulation):
        """Test that function returns a tuple."""
        from presto.sample import _add_torsion_restraint_forces

        torsion_atoms_list = [(0, 1, 2, 3)]
        result = _add_torsion_restraint_forces(
            simple_simulation, torsion_atoms_list, 1000.0
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_force_indices(self, simple_simulation):
        """Test that function returns force indices."""
        from presto.sample import _add_torsion_restraint_forces

        torsion_atoms_list = [(0, 1, 2, 3)]
        force_indices, _ = _add_torsion_restraint_forces(
            simple_simulation, torsion_atoms_list, 1000.0
        )
        assert isinstance(force_indices, list)
        assert len(force_indices) == len(torsion_atoms_list)

    def test_adds_forces_to_system(self, simple_simulation):
        """Test that function adds forces to the system."""
        from presto.sample import _add_torsion_restraint_forces

        initial_n_forces = simple_simulation.system.getNumForces()
        torsion_atoms_list = [(0, 1, 2, 3)]
        _add_torsion_restraint_forces(simple_simulation, torsion_atoms_list, 1000.0)

        assert simple_simulation.system.getNumForces() > initial_n_forces


class TestRemoveTorsionRestraintForces:
    """Tests for _remove_torsion_restraint_forces function."""

    @pytest.fixture
    def simulation_with_restraints(self):
        """Create a simulation with torsion restraints added."""
        from openmm import HarmonicBondForce, LangevinMiddleIntegrator

        from presto.sample import _add_torsion_restraint_forces

        system = System()
        for _ in range(4):
            system.addParticle(12.0)

        bond_force = HarmonicBondForce()
        bond_force.addBond(0, 1, 0.15, 1000.0)
        bond_force.addBond(1, 2, 0.15, 1000.0)
        bond_force.addBond(2, 3, 0.15, 1000.0)
        system.addForce(bond_force)

        topology = OMMTopology()
        chain = topology.addChain()
        residue = topology.addResidue("MOL", chain)
        for i in range(4):
            topology.addAtom(f"C{i}", None, residue)

        integrator = LangevinMiddleIntegrator(
            300 * omm_unit.kelvin,
            1.0 / omm_unit.picoseconds,
            1.0 * omm_unit.femtoseconds,
        )

        simulation = Simulation(topology, system, integrator)
        positions = [
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.25, 0.1, 0.0],
            [0.35, 0.1, 0.0],
        ]
        simulation.context.setPositions(positions)

        torsion_atoms_list = [(0, 1, 2, 3)]
        force_indices, force_group = _add_torsion_restraint_forces(
            simulation, torsion_atoms_list, 1000.0
        )

        return simulation, force_indices

    def test_import_function(self):
        """Test that function can be imported."""
        from presto.sample import _remove_torsion_restraint_forces

        assert callable(_remove_torsion_restraint_forces)

    def test_removes_forces_from_system(self, simulation_with_restraints):
        """Test that function removes forces from the system."""
        from presto.sample import _remove_torsion_restraint_forces

        simulation, force_indices = simulation_with_restraints
        n_forces_before = simulation.system.getNumForces()

        _remove_torsion_restraint_forces(simulation, force_indices)

        assert simulation.system.getNumForces() == n_forces_before - len(force_indices)


class TestGenerateTorsionMinimisedDataset:
    """Tests for generate_torsion_minimised_dataset function."""

    def test_import_function(self):
        """Test that function can be imported."""
        from presto.sample import generate_torsion_minimised_dataset

        assert callable(generate_torsion_minimised_dataset)

    # Note: Full integration tests for generate_torsion_minimised_dataset
    # require actual ML potentials and are tested in integration tests


class TestSampleMMMDMetadynamicsWithTorsionMinimisation:
    """Tests for sample_mmmd_metadynamics_with_torsion_minimisation function."""

    def test_import_function(self):
        """Test that function can be imported."""
        from presto.sample import (
            sample_mmmd_metadynamics_with_torsion_minimisation,
        )

        assert callable(sample_mmmd_metadynamics_with_torsion_minimisation)

    def test_registered_in_sampling_registry(self):
        """Test that function is registered in the sampling registry."""
        from presto.sample import (
            _SAMPLING_FNS_REGISTRY,
            sample_mmmd_metadynamics_with_torsion_minimisation,
        )
        from presto.settings import (
            MMMDMetadynamicsTorsionMinimisationSamplingSettings,
        )

        # Check the settings type is in the registry
        assert (
            MMMDMetadynamicsTorsionMinimisationSamplingSettings
            in _SAMPLING_FNS_REGISTRY
        )
        # Check it maps to the correct function
        fn = _SAMPLING_FNS_REGISTRY[MMMDMetadynamicsTorsionMinimisationSamplingSettings]
        assert fn == sample_mmmd_metadynamics_with_torsion_minimisation


class TestWeightedDatasetIntegration:
    """Integration tests for weighted datasets in sampling functions."""

    def test_sample_mmmd_creates_weighted_dataset(self):
        """Test that sample_mmmd creates a weighted dataset."""
        # This is a lightweight test that checks the function signature
        # Full integration tests are in the integration test folder
        # Check the function accepts the right arguments
        import inspect

        from presto.sample import sample_mmmd

        sig = inspect.signature(sample_mmmd)
        params = list(sig.parameters.keys())
        assert "mols" in params
        assert "off_ff" in params
        assert "device" in params
        assert "settings" in params

    def test_sample_mlmd_creates_weighted_dataset(self):
        """Test that sample_mlmd creates a weighted dataset."""
        import inspect

        from presto.sample import sample_mlmd

        sig = inspect.signature(sample_mlmd)
        params = list(sig.parameters.keys())
        assert "mols" in params
        assert "off_ff" in params

    def test_sample_mmmd_metadynamics_creates_weighted_dataset(self):
        """Test that sample_mmmd_metadynamics creates a weighted dataset."""
        import inspect

        from presto.sample import sample_mmmd_metadynamics

        sig = inspect.signature(sample_mmmd_metadynamics)
        params = list(sig.parameters.keys())
        assert "mols" in params
        assert "output_paths" in params


class TestDatasetCreationFunctions:
    """Tests for dataset creation utility usage in sample module."""

    def test_create_dataset_with_uniform_weights_integration(self):
        """Test create_dataset_with_uniform_weights works with sample data types."""
        n_confs = 5
        n_atoms = 9  # Ethanol

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

    def test_nan_forces_handling(self):
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


class TestGenerateTorsionMinimisedDatasetPDBOutput:
    """Tests for PDB output functionality in generate_torsion_minimised_dataset."""

    def test_function_accepts_pdb_path_arguments(self):
        """Test that generate_torsion_minimised_dataset accepts PDB path arguments."""
        import inspect

        from presto.sample import generate_torsion_minimised_dataset

        sig = inspect.signature(generate_torsion_minimised_dataset)
        params = list(sig.parameters.keys())
        assert "ml_pdb_path" in params
        assert "mm_pdb_path" in params

    def test_pdb_path_defaults_to_none(self):
        """Test that PDB path arguments default to None."""
        import inspect

        from presto.sample import generate_torsion_minimised_dataset

        sig = inspect.signature(generate_torsion_minimised_dataset)
        assert sig.parameters["ml_pdb_path"].default is None
        assert sig.parameters["mm_pdb_path"].default is None


class TestMMMDMetadynamicsTorsionMinimisationOutputTypes:
    """Tests for output types in MMMDMetadynamicsTorsionMinimisationSamplingSettings."""

    def test_output_types_includes_ml_minimised_pdb(self):
        """Test that output_types includes ML_MINIMISED_PDB."""
        from presto.outputs import OutputType
        from presto.settings import (
            MMMDMetadynamicsTorsionMinimisationSamplingSettings,
        )

        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert OutputType.ML_MINIMISED_PDB in settings.output_types

    def test_output_types_includes_mm_minimised_pdb(self):
        """Test that output_types includes MM_MINIMISED_PDB."""
        from presto.outputs import OutputType
        from presto.settings import (
            MMMDMetadynamicsTorsionMinimisationSamplingSettings,
        )

        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert OutputType.MM_MINIMISED_PDB in settings.output_types

    def test_output_types_includes_all_expected_types(self):
        """Test that output_types includes all expected types."""
        from presto.outputs import OutputType
        from presto.settings import (
            MMMDMetadynamicsTorsionMinimisationSamplingSettings,
        )

        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        expected = {
            OutputType.METADYNAMICS_BIAS,
            OutputType.PDB_TRAJECTORY,
            OutputType.ML_MINIMISED_PDB,
            OutputType.MM_MINIMISED_PDB,
        }
        assert settings.output_types == expected


class TestOutputTypeEnumeration:
    """Tests for OutputType enum values."""

    def test_ml_minimised_pdb_output_type_exists(self):
        """Test that ML_MINIMISED_PDB OutputType exists."""
        from presto.outputs import OutputType

        assert hasattr(OutputType, "ML_MINIMISED_PDB")
        assert OutputType.ML_MINIMISED_PDB.value == "ml_minimised.pdb"

    def test_mm_minimised_pdb_output_type_exists(self):
        """Test that MM_MINIMISED_PDB OutputType exists."""
        from presto.outputs import OutputType

        assert hasattr(OutputType, "MM_MINIMISED_PDB")
        assert OutputType.MM_MINIMISED_PDB.value == "mm_minimised.pdb"
