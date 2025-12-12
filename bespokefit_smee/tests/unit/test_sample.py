"""Unit tests for sample module."""

import numpy as np
import openmm
import pytest
import torch
from openff.toolkit import ForceField, Molecule
from openmm import unit as omm_unit
from openmm.app import Simulation

from bespokefit_smee.sample import (
    _add_torsion_restraint,
    _calculate_torsion_angles,
    _remove_torsion_restraint,
    _select_diverse_samples_from_bin,
)


class TestCalculateTorsionAngles:
    """Tests for _calculate_torsion_angles function."""

    def test_single_snapshot_known_angle(self):
        """Test calculation with a known torsion angle."""
        # Create coordinates for a simple torsion with known angle
        # Define 4 atoms in a line along x-axis, then rotate last atom
        coords = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],  # Atom 0
                    [1.0, 0.0, 0.0],  # Atom 1
                    [2.0, 0.0, 0.0],  # Atom 2
                    [3.0, 0.0, 0.0],  # Atom 3 (180 degree torsion)
                ]
            ]
        )
        torsion_atoms = (0, 1, 2, 3)

        angles = _calculate_torsion_angles(coords, torsion_atoms)

        # Should be approximately 180 degrees (pi radians) or 0
        # (depends on definition, but should be consistent)
        assert angles.shape == (1,)
        assert (
            torch.abs(angles[0]) < 0.1 or torch.abs(torch.abs(angles[0]) - np.pi) < 0.1
        )

    def test_multiple_snapshots(self):
        """Test calculation with multiple snapshots."""
        # Create 3 snapshots with different torsion angles
        n_snapshots = 3
        coords = torch.randn(n_snapshots, 10, 3)
        torsion_atoms = (0, 1, 2, 3)

        angles = _calculate_torsion_angles(coords, torsion_atoms)

        assert angles.shape == (n_snapshots,)
        # Angles should be in range [-pi, pi]
        assert torch.all(angles >= -np.pi)
        assert torch.all(angles <= np.pi)

    def test_zero_degree_torsion(self):
        """Test with atoms in a planar configuration (0 degree torsion)."""
        # All atoms in xy plane
        coords = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                ]
            ]
        )
        torsion_atoms = (0, 1, 2, 3)

        angles = _calculate_torsion_angles(coords, torsion_atoms)

        # Should be close to 0 / pi
        assert (
            torch.abs(torch.abs(angles[0]) - 0) < 0.1
            or torch.abs(torch.abs(angles[0]) - np.pi) < 0.1
        )

    def test_ninety_degree_torsion(self):
        """Test with a 90 degree torsion."""
        # Create a 90 degree torsion
        coords = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 1.0],  # Perpendicular to first 3 atoms
                ]
            ]
        )
        torsion_atoms = (0, 1, 2, 3)

        angles = _calculate_torsion_angles(coords, torsion_atoms)

        # Should be close to pi/2 or -pi/2
        assert torch.abs(torch.abs(angles[0]) - np.pi / 2) < 0.2


class TestSelectDiverseSamplesFromBin:
    """Tests for _select_diverse_samples_from_bin function."""

    def test_fewer_samples_than_max(self):
        """Test when bin has fewer samples than max."""
        sample_indices = [0, 1, 2]
        max_samples = 5

        selected = _select_diverse_samples_from_bin(sample_indices, max_samples)

        assert len(selected) == 3
        assert selected == sample_indices

    def test_more_samples_than_max(self):
        """Test when bin has more samples than max."""
        sample_indices = list(range(20))
        max_samples = 5

        selected = _select_diverse_samples_from_bin(sample_indices, max_samples)

        assert len(selected) == max_samples
        # Check that samples are evenly spaced
        assert selected[0] == 0
        assert selected[-1] == 16  # int(4 * 20/5) = 16

    def test_exact_max_samples(self):
        """Test when bin has exactly max samples."""
        sample_indices = [0, 1, 2, 3, 4]
        max_samples = 5

        selected = _select_diverse_samples_from_bin(sample_indices, max_samples)

        assert len(selected) == 5
        assert selected == sample_indices

    def test_single_sample(self):
        """Test with a single sample."""
        sample_indices = [42]
        max_samples = 5

        selected = _select_diverse_samples_from_bin(sample_indices, max_samples)

        assert len(selected) == 1
        assert selected == [42]

    def test_spacing_consistency(self):
        """Test that samples are evenly spaced."""
        sample_indices = list(range(100))
        max_samples = 10

        selected = _select_diverse_samples_from_bin(sample_indices, max_samples)

        assert len(selected) == max_samples
        # Check spacing is approximately uniform
        spacings = [selected[i + 1] - selected[i] for i in range(len(selected) - 1)]
        assert all(s == 10 for s in spacings)


class TestAddRemoveTorsionRestraint:
    """Tests for _add_torsion_restraint and _remove_torsion_restraint."""

    @pytest.fixture
    def ethanol_simulation(self):
        """Create a simple ethanol simulation for testing."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=1)

        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        from openff.interchange import Interchange

        interchange = Interchange.from_smirnoff(ff, mol.to_topology())
        system = interchange.to_openmm_system()

        integrator = openmm.LangevinMiddleIntegrator(
            300 * openmm.unit.kelvin,
            1 / openmm.unit.picosecond,
            1 * openmm.unit.femtosecond,
        )

        simulation = Simulation(interchange.topology.to_openmm(), system, integrator)
        simulation.context.setPositions(mol.conformers[0].to_openmm())

        return simulation

    def test_add_restraint_increases_force_count(self, ethanol_simulation):
        """Test that adding a restraint increases the number of forces."""
        initial_force_count = ethanol_simulation.system.getNumForces()

        torsion_atoms = (0, 1, 2, 3)
        target_angle = 0.0
        force_constant = 100.0

        force_idx = _add_torsion_restraint(
            ethanol_simulation,
            torsion_atoms,
            target_angle,
            force_constant,
        )

        new_force_count = ethanol_simulation.system.getNumForces()
        assert new_force_count == initial_force_count + 1
        assert force_idx == initial_force_count

    def test_remove_restraint_decreases_force_count(self, ethanol_simulation):
        """Test that removing a restraint decreases force count."""
        torsion_atoms = (0, 1, 2, 3)
        target_angle = 0.0
        force_constant = 100.0

        force_idx = _add_torsion_restraint(
            ethanol_simulation,
            torsion_atoms,
            target_angle,
            force_constant,
        )

        initial_force_count = ethanol_simulation.system.getNumForces()

        _remove_torsion_restraint(ethanol_simulation, force_idx)

        new_force_count = ethanol_simulation.system.getNumForces()
        assert new_force_count == initial_force_count - 1

    def test_add_remove_cycle(self, ethanol_simulation):
        """Test add and remove restraint returns to original state."""
        initial_force_count = ethanol_simulation.system.getNumForces()

        torsion_atoms = (0, 1, 2, 3)
        target_angle = 0.0
        force_constant = 100.0

        # Add restraint
        force_idx = _add_torsion_restraint(
            ethanol_simulation,
            torsion_atoms,
            target_angle,
            force_constant,
        )

        # Remove restraint
        _remove_torsion_restraint(ethanol_simulation, force_idx)

        assert ethanol_simulation.system.getNumForces() == initial_force_count

    def test_restraint_affects_energy(self, ethanol_simulation):
        """Test that the restraint actually affects the system energy."""
        # Get initial energy
        state1 = ethanol_simulation.context.getState(getEnergy=True)
        energy1 = state1.getPotentialEnergy()

        # Add a strong restraint at a different angle
        torsion_atoms = (0, 1, 2, 3)
        target_angle = np.pi  # 180 degrees
        force_constant = (
            1000.0 * openmm.unit.kilocalories_per_mole / openmm.unit.degree**2
        )

        force_idx = _add_torsion_restraint(
            ethanol_simulation,
            torsion_atoms,
            target_angle,
            force_constant,
        )

        # Get energy with restraint
        state2 = ethanol_simulation.context.getState(getEnergy=True)
        energy2 = state2.getPotentialEnergy()

        # Energies should be different (restraint adds energy)
        assert energy1 != energy2

        # Clean up
        _remove_torsion_restraint(ethanol_simulation, force_idx)

    def test_multiple_restraints(self, ethanol_simulation):
        """Test adding multiple restraints."""
        initial_force_count = ethanol_simulation.system.getNumForces()

        # Add first restraint
        force_idx1 = _add_torsion_restraint(
            ethanol_simulation,
            (0, 1, 2, 3),
            0.0,
            100.0 * openmm.unit.kilocalories_per_mole / openmm.unit.degree**2,
        )

        # Add second restraint
        force_idx2 = _add_torsion_restraint(
            ethanol_simulation,
            (1, 2, 3, 4),
            np.pi / 2,
            100.0 * openmm.unit.kilocalories_per_mole / openmm.unit.degree**2,
        )

        new_force_count = ethanol_simulation.system.getNumForces()
        assert new_force_count == initial_force_count + 2
        assert force_idx2 > force_idx1

        # Remove in reverse order to maintain indices
        _remove_torsion_restraint(ethanol_simulation, force_idx2)
        _remove_torsion_restraint(ethanol_simulation, force_idx1)

        final_force_count = ethanol_simulation.system.getNumForces()
        assert final_force_count == initial_force_count


class TestSamplingSettingsIntegration:
    """Integration tests for the new sampling settings."""

    def test_settings_class_exists(self):
        """Test that the new settings class can be imported."""
        from bespokefit_smee.settings import (
            MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings,
        )

        protocol = "mm_md_metadynamics_seeded_frozen_torsions"
        settings = MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings(
            sampling_protocol=protocol
        )

        assert settings.n_angle_bins == 12
        assert settings.n_samples_per_bin == 5
        assert settings.bias_width == np.pi / 10

    def test_settings_validation(self):
        """Test that settings validation works."""
        from bespokefit_smee.settings import (
            MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings,
        )

        # Should work with valid settings
        protocol = "mm_md_metadynamics_seeded_frozen_torsions"
        settings = MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings(
            sampling_protocol=protocol,
            timestep=1 * omm_unit.femtosecond,
            snapshot_interval=100 * omm_unit.femtosecond,
            frozen_production_sampling_time_per_seed=(1.0 * omm_unit.picosecond),
        )

        assert settings.frozen_production_n_snapshots_per_seed == 10

    def test_settings_in_union(self):
        """Test new settings class is in SamplingSettings union."""
        # Check that the union can handle the new type
        from pydantic import TypeAdapter

        from bespokefit_smee.settings import SamplingSettings

        adapter = TypeAdapter(SamplingSettings)

        protocol = "mm_md_metadynamics_seeded_frozen_torsions"
        data = {
            "sampling_protocol": protocol,
        }

        settings = adapter.validate_python(data)
        assert settings.sampling_protocol == protocol


class TestSamplingFunctionRegistry:
    """Test that the new sampling function is properly registered."""

    def test_sampling_function_registered(self):
        """Test that the new sampling function is in the registry."""
        from bespokefit_smee.sample import _SAMPLING_FNS_REGISTRY
        from bespokefit_smee.settings import (
            MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings,
        )

        assert (
            MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings
            in _SAMPLING_FNS_REGISTRY
        )

    def test_sampling_function_callable(self):
        """Test that the registered sampling function is callable."""
        from bespokefit_smee.sample import _SAMPLING_FNS_REGISTRY
        from bespokefit_smee.settings import (
            MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings,
        )

        sampling_fn = _SAMPLING_FNS_REGISTRY[
            MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings
        ]

        assert callable(sampling_fn)
        # Function name should match expected name
        expected_name = "sample_mmmd_metadynamics_seeded_frozen_torsions"
        assert hasattr(sampling_fn, "__name__")
        if hasattr(sampling_fn, "__name__"):
            assert sampling_fn.__name__ == expected_name  # type: ignore
