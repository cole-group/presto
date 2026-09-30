"""Unit tests for the implicit_solvent module."""

import math

import openff.interchange
import openmm
import pytest
from openff.toolkit import Molecule
from openmm import unit as omm_unit

from presto.implicit_solvent import (
    _GB_FORCE_CLASSES,
    _get_charges,
    _get_kappa,
    add_implicit_solvent_force,
)
from presto.settings import ImplicitSolventSettings

_TEMPERATURE = 300 * omm_unit.kelvin


@pytest.fixture
def ethanol_interchange(simple_force_field):
    """An interchange for ethanol with a single conformer."""
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=1)
    return openff.interchange.Interchange.from_smirnoff(
        simple_force_field, mol.to_topology()
    )


def _potential_energy(system: openmm.System, positions) -> float:
    """Compute the potential energy (kJ / mol) of a system at the given positions."""
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(1.0 * omm_unit.femtosecond),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(positions)
    return (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(omm_unit.kilojoules_per_mole)
    )


def _get_gb_forces(system: openmm.System) -> list[openmm.CustomGBForce]:
    return [
        force for force in system.getForces() if isinstance(force, openmm.CustomGBForce)
    ]


class TestGetKappa:
    """Tests for the salt concentration to kappa conversion."""

    def test_zero_salt_gives_zero_kappa(self):
        """Zero salt concentration means no screening."""
        assert _get_kappa(0.0 * omm_unit.molar, 78.5, _TEMPERATURE) == 0.0

    def test_matches_openmm_conversion(self):
        """The conversion matches the one used by OpenMM for Amber systems."""
        kappa = _get_kappa(0.15 * omm_unit.molar, 78.5, _TEMPERATURE)
        expected = 50.33355 * (0.15 / 78.5 / 300.0) ** 0.5 * 7.3
        assert kappa == pytest.approx(expected)

    def test_increases_with_salt_concentration(self):
        """More salt means more screening."""
        assert _get_kappa(0.5 * omm_unit.molar, 78.5, _TEMPERATURE) > _get_kappa(
            0.1 * omm_unit.molar, 78.5, _TEMPERATURE
        )


class TestGetCharges:
    """Tests for reading the partial charges back from a system."""

    def test_charges_read_from_nonbonded_force(self, ethanol_interchange):
        """The charges match those in the system's nonbonded force."""
        system = ethanol_interchange.to_openmm_system()
        charges = _get_charges(system)

        nonbonded = next(
            force
            for force in system.getForces()
            if isinstance(force, openmm.NonbondedForce)
        )
        expected = [
            nonbonded.getParticleParameters(i)[0].value_in_unit(
                omm_unit.elementary_charge
            )
            for i in range(nonbonded.getNumParticles())
        ]

        assert charges == expected
        assert sum(charges) == pytest.approx(0.0, abs=1e-6)

    def test_raises_without_nonbonded_force(self):
        """A helpful error is raised if the charges cannot be determined."""
        system = openmm.System()
        system.addParticle(1.0)

        with pytest.raises(ValueError, match="without a NonbondedForce"):
            _get_charges(system)


class TestAddImplicitSolventForce:
    """Tests for add_implicit_solvent_force."""

    @pytest.mark.parametrize("model", sorted(_GB_FORCE_CLASSES))
    def test_all_models_can_be_added(self, ethanol_interchange, model):
        """Every supported GB model produces a usable force."""
        system = ethanol_interchange.to_openmm_system()
        n_atoms = ethanol_interchange.topology.n_atoms

        force = add_implicit_solvent_force(
            system,
            ethanol_interchange.topology.to_openmm(),
            ImplicitSolventSettings(model=model),
            _TEMPERATURE,
        )

        assert isinstance(force, _GB_FORCE_CLASSES[model])
        assert force.getNumParticles() == n_atoms
        assert len(_get_gb_forces(system)) == 1

        # The force must be usable, i.e. the energy is finite.
        energy = _potential_energy(system, ethanol_interchange.positions.to_openmm())
        assert math.isfinite(energy)

    def test_charges_match_the_nonbonded_force(self, ethanol_interchange):
        """The per-particle charges are taken from the MM system itself."""
        system = ethanol_interchange.to_openmm_system()
        expected_charges = _get_charges(system)

        force = add_implicit_solvent_force(
            system,
            ethanol_interchange.topology.to_openmm(),
            ImplicitSolventSettings(),
            _TEMPERATURE,
        )

        charges = [
            force.getParticleParameters(i)[0] for i in range(force.getNumParticles())
        ]
        assert charges == pytest.approx(expected_charges)

    def test_solvation_lowers_the_energy_of_a_polar_molecule(self, ethanol_interchange):
        """Adding implicit solvent changes (and here lowers) the MM energy."""
        positions = ethanol_interchange.positions.to_openmm()

        vacuum_energy = _potential_energy(
            ethanol_interchange.to_openmm_system(), positions
        )

        solvated_system = ethanol_interchange.to_openmm_system()
        add_implicit_solvent_force(
            solvated_system,
            ethanol_interchange.topology.to_openmm(),
            ImplicitSolventSettings(),
            _TEMPERATURE,
        )
        solvated_energy = _potential_energy(solvated_system, positions)

        assert solvated_energy < vacuum_energy

    def test_dielectrics_change_the_energy(self, ethanol_interchange):
        """A vacuum-like solvent dielectric gives a smaller solvation energy."""
        positions = ethanol_interchange.positions.to_openmm()

        energies = []
        for solvent_dielectric in (2.0, 78.5):
            system = ethanol_interchange.to_openmm_system()
            add_implicit_solvent_force(
                system,
                ethanol_interchange.topology.to_openmm(),
                ImplicitSolventSettings(solvent_dielectric=solvent_dielectric),
                _TEMPERATURE,
            )
            energies.append(_potential_energy(system, positions))

        assert energies[0] > energies[1]

    def test_salt_concentration_changes_the_energy(self, ethanol_interchange):
        """A non-zero salt concentration screens the solvation term."""
        positions = ethanol_interchange.positions.to_openmm()

        energies = []
        for salt_concentration in (0.0, 0.5):
            system = ethanol_interchange.to_openmm_system()
            add_implicit_solvent_force(
                system,
                ethanol_interchange.topology.to_openmm(),
                ImplicitSolventSettings(
                    salt_concentration=salt_concentration * omm_unit.molar
                ),
                _TEMPERATURE,
            )
            energies.append(_potential_energy(system, positions))

        assert energies[0] != pytest.approx(energies[1])

    def test_no_surface_area_term_changes_the_energy(self, ethanol_interchange):
        """Omitting the non-polar term changes the energy."""
        positions = ethanol_interchange.positions.to_openmm()

        energies = []
        for surface_area_model in ("ACE", None):
            system = ethanol_interchange.to_openmm_system()
            add_implicit_solvent_force(
                system,
                ethanol_interchange.topology.to_openmm(),
                ImplicitSolventSettings(surface_area_model=surface_area_model),
                _TEMPERATURE,
            )
            energies.append(_potential_energy(system, positions))

        assert energies[0] != pytest.approx(energies[1])

    def test_raises_on_particle_count_mismatch(self, ethanol_interchange):
        """Systems with extra particles (e.g. virtual sites) are rejected."""
        system = ethanol_interchange.to_openmm_system()
        system.addParticle(0.0)

        with pytest.raises(ValueError, match="virtual sites"):
            add_implicit_solvent_force(
                system,
                ethanol_interchange.topology.to_openmm(),
                ImplicitSolventSettings(),
                _TEMPERATURE,
            )
