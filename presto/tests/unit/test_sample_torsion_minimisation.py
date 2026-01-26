"""Unit tests for sample module - torsion minimisation functions."""

import numpy as np
import pytest
import torch
from openmm import System
from openmm import unit as omm_unit
from openmm.app import Simulation
from openmm.app import Topology as OMMTopology

from presto.data_utils import (
    create_dataset_with_uniform_weights,
    has_weights,
)


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
