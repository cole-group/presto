"""Unit tests for hessian.py."""

import numpy as np
import pytest
from openff.toolkit import ForceField, Molecule
from openmm import CustomBondForce, LangevinMiddleIntegrator, System, unit
from openmm.app import Simulation, Topology

from ...hessian import calculate_hessian
from ...sample import _get_integrator


def test_calculate_hessian_simple_harmonic():
    """Test Hessian calculation with a simple 2-particle harmonic oscillator.

    For a simple harmonic bond potential U = (k/2) * (r - r0)^2, the Hessian
    at equilibrium can be calculated analytically. This test verifies that
    the numerical Hessian matches the analytical result.

    All calculations use kcal/mol units consistently.
    """
    # Define system parameters
    k = 1000 * unit.kilojoule_per_mole / unit.nanometers**2
    r0 = 0.15 * unit.nanometers  # equilibrium bond length
    mass = 12.0 * unit.amu  # particle mass

    # Create a simple 2-particle system
    system = System()
    system.addParticle(mass)
    system.addParticle(mass)

    # Add harmonic bond force: U = (k/2) * (r - r0)^2
    # CustomBondForce expects energy in kJ/mol, so convert k
    k_kj = k.value_in_unit(unit.kilojoules_per_mole / unit.nanometers**2)
    r0_nm = r0.value_in_unit(unit.nanometers)

    bond_force = CustomBondForce("0.5 * k * (r - r0)^2")
    bond_force.addGlobalParameter("k", k_kj)
    bond_force.addGlobalParameter("r0", r0_nm)
    bond_force.addBond(0, 1)
    system.addForce(bond_force)

    # Create topology
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("DUM", chain)
    topology.addAtom("C1", "C", residue)
    topology.addAtom("C2", "C", residue)

    # Create simulation at equilibrium position
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtoseconds
    )
    simulation = Simulation(topology, system, integrator)

    # Place particles at equilibrium separation along x-axis
    positions = np.array([[0.0, 0.0, 0.0], [r0_nm, 0.0, 0.0]]) * unit.nanometers
    simulation.context.setPositions(positions)

    # Calculate Hessian (returns in kcal/mol/nm^2)
    hessian = calculate_hessian(simulation, positions)

    # Extract values in kcal/mol/nm^2
    hessian_value = hessian.value_in_unit(
        unit.kilocalories_per_mole / unit.nanometers**2
    )

    # Expected force constant in kcal/mol/nm^2
    k_kcal = k.value_in_unit(unit.kilocalories_per_mole / unit.nanometers**2)

    # Analytical Hessian for harmonic bond along x-axis
    # The potential is U = (k/2) * (r - r0)^2 where r = |r2 - r1|
    # At equilibrium (r = r0), the Hessian has a specific structure:
    # For a bond along x-axis between particles at positions x1 and x2,
    # the second derivatives are:
    # d²U/dx1² = k,  d²U/dx1dx2 = -k
    # d²U/dx2² = k,  d²U/dx2dx1 = -k
    # All other elements (involving y, z) are zero at equilibrium

    expected_hessian = np.zeros((6, 6))
    # Block for particle 1
    expected_hessian[0, 0] = k_kcal  # d²U/dx1²
    expected_hessian[0, 3] = -k_kcal  # d²U/dx1dx2
    # Block for particle 2
    expected_hessian[3, 0] = -k_kcal  # d²U/dx2dx1
    expected_hessian[3, 3] = k_kcal  # d²U/dx2²

    # Check shape
    assert hessian_value.shape == (
        6,
        6,
    ), f"Expected shape (6, 6), got {hessian_value.shape}"

    # Check that non-zero elements match analytical result (1% tolerance)
    np.testing.assert_allclose(
        hessian_value[0, 0],
        expected_hessian[0, 0],
        rtol=0.01,
        err_msg=f"Hessian[0,0] mismatch: got {hessian_value[0, 0]:.2f}, expected {expected_hessian[0, 0]:.2f} kcal/mol/nm²",
    )
    np.testing.assert_allclose(
        hessian_value[0, 3],
        expected_hessian[0, 3],
        rtol=0.01,
        err_msg=f"Hessian[0,3] mismatch: got {hessian_value[0, 3]:.2f}, expected {expected_hessian[0, 3]:.2f} kcal/mol/nm²",
    )
    np.testing.assert_allclose(
        hessian_value[3, 0],
        expected_hessian[3, 0],
        rtol=0.01,
        err_msg=f"Hessian[3,0] mismatch: got {hessian_value[3, 0]:.2f}, expected {expected_hessian[3, 0]:.2f} kcal/mol/nm²",
    )
    np.testing.assert_allclose(
        hessian_value[3, 3],
        expected_hessian[3, 3],
        rtol=0.01,
        err_msg=f"Hessian[3,3] mismatch: got {hessian_value[3, 3]:.2f}, expected {expected_hessian[3, 3]:.2f} kcal/mol/nm²",
    )

    # Check that elements involving y and z coordinates are approximately zero
    # (small numerical errors are acceptable)
    for i in [1, 2, 4, 5]:  # y and z coordinates
        for j in range(6):
            assert pytest.approx(hessian_value[i, j], abs=1e-2) == 0.0, (
                f"Hessian[{i},{j}] should be ~0, got {hessian_value[i, j]:.4f} kcal/mol/nm²"
            )


def test_calculate_hessian_three_particle_angle():
    """Test Hessian calculation with a 3-particle harmonic angle potential.

    For a harmonic angle potential U = (k/2) * (theta - theta0)^2,
    we can verify the Hessian has the correct symmetry and magnitude.

    All calculations use kcal/mol units consistently.
    """
    # Define system parameters with explicit units
    k_angle = 100 * unit.kilojoules_per_mole / unit.radians**2  # angle force constant
    theta0 = (np.pi / 2.0) * unit.radians  # 90 degrees equilibrium angle
    r_bond = 0.15 * unit.nanometers  # bond length
    mass = 12.0 * unit.amu  # particle mass

    # Create a 3-particle system
    system = System()
    for _ in range(3):
        system.addParticle(mass)

    # Add harmonic angle force: U = (k/2) * (theta - theta0)^2
    from openmm import CustomAngleForce

    # CustomAngleForce expects energy in kJ/mol, so convert k
    k_angle_kj = k_angle.value_in_unit(unit.kilojoules_per_mole / unit.radians**2)
    theta0_rad = theta0.value_in_unit(unit.radians)

    angle_force = CustomAngleForce("0.5 * k * (theta - theta0)^2")
    angle_force.addGlobalParameter("k", k_angle_kj)
    angle_force.addGlobalParameter("theta0", theta0_rad)
    angle_force.addAngle(0, 1, 2)
    system.addForce(angle_force)

    # Create topology
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("DUM", chain)
    for i in range(3):
        topology.addAtom(f"C{i + 1}", "C", residue)

    # Create simulation at equilibrium geometry (90 degree angle in xy plane)
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtoseconds
    )
    simulation = Simulation(topology, system, integrator)

    # Place particles in L-shape at 90 degrees
    r_bond_nm = r_bond.value_in_unit(unit.nanometers)
    positions = (
        np.array(
            [
                [0.0, 0.0, 0.0],  # particle 0
                [r_bond_nm, 0.0, 0.0],  # particle 1 (central)
                [r_bond_nm, r_bond_nm, 0.0],  # particle 2
            ]
        )
        * unit.nanometers
    )
    simulation.context.setPositions(positions)

    # Calculate Hessian (returns in kcal/mol/nm^2)
    hessian = calculate_hessian(simulation, positions)

    # Basic checks
    assert hessian.shape == (9, 9), f"Expected shape (9, 9), got {hessian.shape}"
    assert isinstance(hessian, unit.Quantity)

    # Check symmetry - extract values in kcal/mol/nm^2
    hessian_value = hessian.value_in_unit(
        unit.kilocalories_per_mole / unit.nanometers**2
    )
    np.testing.assert_allclose(
        hessian_value, hessian_value.T, rtol=1e-6, err_msg="Hessian should be symmetric"
    )

    # The Hessian should have non-zero elements related to the angle
    # At equilibrium with a 90-degree angle, the force constant should appear
    # in the appropriate second derivative terms
    assert np.abs(hessian_value).max() > 0.1, (
        f"Hessian should have non-trivial values, max value: {np.abs(hessian_value).max():.4f} kcal/mol/nm²"
    )


def test_calculate_hessian_ethanol():
    """Test the calculate_hessian function with a realistic molecule."""

    # Create an OpenMM System and input coordinates for testing
    molecule = Molecule.from_smiles("CCO")
    molecule.generate_conformers(n_conformers=1)
    forcefield = ForceField("openff_unconstrained-2.3.0.offxml")
    system = forcefield.create_openmm_system(molecule.to_topology())
    simulation = Simulation(
        topology=molecule.to_topology().to_openmm(),
        system=system,
        integrator=_get_integrator(300 * unit.kelvin, 1.0 * unit.femtoseconds),
    )

    input_coords = molecule.conformers[0].to_openmm()

    hessian = calculate_hessian(simulation, input_coords)

    assert hessian.shape == (3 * molecule.n_atoms, 3 * molecule.n_atoms)
    assert isinstance(hessian, unit.Quantity)

    # Check that Hessian is symmetric
    hessian_value = hessian.value_in_unit(
        unit.kilocalories_per_mole / unit.nanometers**2
    )
    np.testing.assert_allclose(
        hessian_value, hessian_value.T, rtol=1e-6, err_msg="Hessian should be symmetric"
    )

    # Check that Hessian has reasonable magnitudes (not all zeros, not infinite)
    assert np.abs(hessian_value).max() > 0.1, "Hessian should have non-trivial values"
    assert np.all(np.isfinite(hessian_value)), "Hessian should have finite values"
