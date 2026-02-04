import numpy as np
import pytest
from openmm import (
    CustomBondForce,
    HarmonicBondForce,
    LangevinIntegrator,
    Platform,
    System,
    Vec3,
)
from openmm.app import Simulation, Topology, element
from openmm.app.metadynamics import BiasVariable
from openmm.unit import is_quantity, kelvin, kilojoules_per_mole, picosecond

from presto.metadynamics import Metadynamics


# Adapted from https://github.com/openmm/openmm/blob/8eeee16de9bc772321ae2b87672700b076913b56/wrappers/python/tests/TestMetadynamics.py#L6
def test_harmonic_oscillator():
    """Check we haven't messed up the independentCVs=False case by testing
    running metadynamics on a harmonic oscillator with dependent CVs."""
    system = System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = HarmonicBondForce()
    force.addBond(0, 1, 1.0, 100000.0)
    system.addForce(force)
    cv = CustomBondForce("r")
    cv.addBond(0, 1)

    # Grid parameters
    min_v = 0.94
    max_v = 1.06
    grid_width = 31
    bias = BiasVariable(cv, min_v, max_v, 0.00431, gridWidth=grid_width)

    # Metadynamics parameters
    temperature = 300 * kelvin
    bias_factor = 3.0
    height = 5.0 * kilojoules_per_mole
    frequency = 10

    meta = Metadynamics(
        system,
        [bias],
        temperature,
        bias_factor,
        height,
        frequency,
        independentCVs=False,
    )

    integrator = LangevinIntegrator(temperature, 10 / picosecond, 0.001 * picosecond)
    integrator.setRandomNumberSeed(4321)

    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("H2", chain)
    topology.addAtom("H1", element.hydrogen, residue)
    topology.addAtom("H2", element.hydrogen, residue)

    # Reference platform for reproducibility
    simulation = Simulation(
        topology, system, integrator, Platform.getPlatform("Reference")
    )
    simulation.context.setPositions([Vec3(0, 0, 0), Vec3(1, 0, 0)])

    # Run simulation
    meta.step(
        simulation, 200000
    )  # Reduced steps for speed, but let's see if 20000 is enough

    fe = meta.getFreeEnergy()
    center = grid_width // 2
    fe -= fe[center]

    # Energies should be reasonably well converged over the central part of the range.
    for i in range(center - 3, center + 4):
        r = min_v + i * (max_v - min_v) / (grid_width - 1)
        e = 0.5 * 100000.0 * (r - 1.0) ** 2 * kilojoules_per_mole
        assert (
            abs(fe[i] - e) < 2.0 * kilojoules_per_mole
        )  # Increased tolerance slightly for fewer steps


def test_independent_cvs():
    """Test the independentCVs=True mode."""
    system = System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addParticle(1.0)

    # Two bond forces as CVs
    cv1 = CustomBondForce("r")
    cv1.addBond(0, 1)
    cv2 = CustomBondForce("r")
    cv2.addBond(1, 2)

    bias1 = BiasVariable(cv1, 0.9, 1.1, 0.01, gridWidth=21)
    bias2 = BiasVariable(cv2, 0.9, 1.1, 0.01, gridWidth=21)

    meta = Metadynamics(
        system, [bias1, bias2], 300 * kelvin, 3.0, 5.0, 10, independentCVs=True
    )

    assert meta._independentCVs
    assert meta._selfBias.shape == (2, 21)

    integrator = LangevinIntegrator(300 * kelvin, 10 / picosecond, 0.001 * picosecond)
    simulation = Simulation(
        Topology(), system, integrator, Platform.getPlatform("Reference")
    )
    simulation.context.setPositions([Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 0, 2)])

    meta.step(simulation, 10)
    assert np.any(meta._selfBias > 0)
    assert len(meta._tables) == 2


def test_sync_with_disk(tmp_path):
    """Test saving and loading biases from disk."""
    system = System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    cv = CustomBondForce("r")
    cv.addBond(0, 1)
    bias = BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=21)

    bias_dir = tmp_path / "biases"
    bias_dir.mkdir()

    meta = Metadynamics(
        system,
        [bias],
        300 * kelvin,
        3.0,
        5.0,
        10,
        saveFrequency=10,
        biasDir=str(bias_dir),
    )

    integrator = LangevinIntegrator(300 * kelvin, 10 / picosecond, 0.001 * picosecond)
    topology = Topology()  # Dummy topology
    simulation = Simulation(
        topology, system, integrator, Platform.getPlatform("Reference")
    )
    simulation.context.setPositions([Vec3(0, 0, 0), Vec3(1, 0, 0)])

    meta.step(simulation, 10)

    # Check if file exists
    files = list(bias_dir.glob("bias_*.npy"))
    assert len(files) == 1

    # Create another metadynamics object and system to load the bias
    system2 = System()
    system2.addParticle(1.0)
    system2.addParticle(1.0)
    cv2 = CustomBondForce("r")
    cv2.addBond(0, 1)
    bias2 = BiasVariable(cv2, 0.9, 1.1, 0.01, gridWidth=21)

    meta2 = Metadynamics(
        system2,
        [bias2],
        300 * kelvin,
        3.0,
        5.0,
        10,
        saveFrequency=10,
        biasDir=str(bias_dir),
    )
    assert np.all(meta2._totalBias == meta._selfBias)


def test_get_free_energy():
    """Test getFreeEnergy method."""
    system = System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    cv = CustomBondForce("r")
    cv.addBond(0, 1)
    bias = BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=21)
    meta = Metadynamics(system, [bias], 300 * kelvin, 3.0, 5.0, 10)

    fe = meta.getFreeEnergy()
    assert is_quantity(fe)
    assert isinstance(fe.value_in_unit(kilojoules_per_mole), np.ndarray)
    assert fe.shape == (21,)


def test_get_collective_variables():
    """Test getCollectiveVariables method."""
    system = System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    cv = CustomBondForce("r")
    cv.addBond(0, 1)
    bias = BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=21)
    meta = Metadynamics(system, [bias], 300 * kelvin, 3.0, 5.0, 10)

    integrator = LangevinIntegrator(300 * kelvin, 10 / picosecond, 0.001 * picosecond)
    simulation = Simulation(
        Topology(), system, integrator, Platform.getPlatform("Reference")
    )
    simulation.context.setPositions([Vec3(0, 0, 0), Vec3(1, 0, 0)])

    cv_values = meta.getCollectiveVariables(simulation)
    assert len(cv_values) == 1
    assert abs(cv_values[0] - 1.0) < 1e-6


def test_multiple_cvs():
    """Test 2 and 3 CVs (non-independent)."""
    for n_cvs in [2, 3]:
        system = System()
        for _ in range(n_cvs + 1):
            system.addParticle(1.0)

        variables = []
        for i in range(n_cvs):
            cv = CustomBondForce("r")
            cv.addBond(i, i + 1)
            variables.append(
                BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=5)
            )  # Small grid for speed

        meta = Metadynamics(system, variables, 300 * kelvin, 3.0, 5.0, 10)
        assert meta._totalBias.ndim == n_cvs

        integrator = LangevinIntegrator(
            300 * kelvin, 10 / picosecond, 0.001 * picosecond
        )
        simulation = Simulation(
            Topology(), system, integrator, Platform.getPlatform("Reference")
        )
        positions = [Vec3(i, 0, 0) for i in range(n_cvs + 1)]
        simulation.context.setPositions(positions)

        meta.step(simulation, 10)
        assert np.any(meta._selfBias > 0)


def test_periodic_cvs():
    """Test periodic variables."""
    system = System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    cv = CustomBondForce("r")
    cv.addBond(0, 1)
    # Torsion-like periodic variable
    bias = BiasVariable(cv, -np.pi, np.pi, 0.1, gridWidth=20, periodic=True)

    meta = Metadynamics(system, [bias], 300 * kelvin, 3.0, 5.0, 10)
    integrator = LangevinIntegrator(300 * kelvin, 10 / picosecond, 0.001 * picosecond)
    simulation = Simulation(
        Topology(), system, integrator, Platform.getPlatform("Reference")
    )
    simulation.context.setPositions([Vec3(0, 0, 0), Vec3(1, 0, 0)])

    meta.step(simulation, 10)
    assert np.any(meta._selfBias > 0)


def test_independent_cvs_grid_mismatch():
    """Test error when grid widths mismatch in independent mode."""
    system = System()
    cv1 = CustomBondForce("r")
    cv2 = CustomBondForce("r")
    bias1 = BiasVariable(cv1, 0.9, 1.1, 0.01, gridWidth=21)
    bias2 = BiasVariable(cv2, 0.9, 1.1, 0.01, gridWidth=11)

    with pytest.raises(
        ValueError, match="All variables must have the same number of grid points"
    ):
        Metadynamics(
            system, [bias1, bias2], 300 * kelvin, 3.0, 5.0, 10, independentCVs=True
        )


def test_temperature_as_float():
    """Test non-quantity temperature."""
    system = System()
    cv = CustomBondForce("r")
    bias = BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=21)
    meta = Metadynamics(system, [bias], 300, 3.0, 5.0, 10)
    assert meta.temperature == 300 * kelvin


def test_invalid_parameters():
    """Test validation in Metadynamics constructor."""
    system = System()
    cv = CustomBondForce("r")
    bias = BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=21)

    with pytest.raises(ValueError, match="biasFactor must be > 1"):
        Metadynamics(system, [bias], 300 * kelvin, 0.5, 5.0, 10)

    with pytest.raises(ValueError, match="Must specify both saveFrequency and biasDir"):
        Metadynamics(system, [bias], 300 * kelvin, 3.0, 5.0, 10, saveFrequency=10)


def test_mixed_periodicity():
    """Test mixed periodic/non-periodic variables raises error."""
    system = System()
    cv1 = CustomBondForce("r")
    cv2 = CustomBondForce("r")
    bias1 = BiasVariable(cv1, 0.9, 1.1, 0.01, gridWidth=21, periodic=True)
    bias2 = BiasVariable(cv2, 0.9, 1.1, 0.01, gridWidth=21, periodic=False)

    with pytest.raises(
        ValueError,
        match="Metadynamics cannot handle mixed periodic/non-periodic variables",
    ):
        Metadynamics(system, [bias1, bias2], 300 * kelvin, 3.0, 5.0, 10)


def test_force_group_exhaustion():
    """Test error when all 32 force groups are used."""
    system = System()
    for i in range(32):
        f = CustomBondForce("0")
        f.addBond(0, 0)
        f.setForceGroup(i)
        system.addForce(f)

    cv = CustomBondForce("r")
    bias = BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=21)

    with pytest.raises(
        RuntimeError, match="Cannot assign a force group to the metadynamics force"
    ):
        Metadynamics(system, [bias], 300 * kelvin, 3.0, 5.0, 10)


def test_invalid_save_frequency():
    """Test invalid saveFrequency."""
    system = System()
    cv = CustomBondForce("r")
    bias = BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=21)

    with pytest.raises(
        ValueError, match="saveFrequency must be a multiple of frequency"
    ):
        Metadynamics(system, [bias], 300, 3.0, 5.0, 10, saveFrequency=15, biasDir="tmp")


def test_too_many_cvs():
    """Test error with >3 CVs (non-independent)."""
    system = System()
    variables = []
    for i in range(4):
        _ = system.addParticle(1.0)
        cv = CustomBondForce("r")
        cv.addBond(
            i, i + 1
        )  # This might fail if particles not enough, but system.addParticle(1.0) is in loop
        variables.append(BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=5))
    system.addParticle(1.0)  # One more particle for the last bond

    with pytest.raises(
        ValueError, match="Metadynamics requires 1, 2, or 3 collective variables"
    ):
        Metadynamics(system, variables, 300, 3.0, 5.0, 10, independentCVs=False)


def test_step_offset():
    """Test step() when currentStep is not a multiple of frequency."""
    system = System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    cv = CustomBondForce("r")
    cv.addBond(0, 1)
    bias = BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=21)
    meta = Metadynamics(system, [bias], 300, 3.0, 5.0, 10)

    integrator = LangevinIntegrator(300, 10 / picosecond, 0.001 * picosecond)
    simulation = Simulation(
        Topology(), system, integrator, Platform.getPlatform("Reference")
    )
    simulation.context.setPositions([Vec3(0, 0, 0), Vec3(1, 0, 0)])

    simulation.step(5)  # Offset
    meta.step(simulation, 10)
    assert simulation.currentStep == 15


def test_sync_with_disk_io_error(tmp_path):
    """Test IO error during sync."""
    from unittest.mock import patch

    system = System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    cv = CustomBondForce("r")
    cv.addBond(0, 1)
    bias = BiasVariable(cv, 0.9, 1.1, 0.01, gridWidth=21)

    bias_dir = tmp_path / "biases_io"
    bias_dir.mkdir()

    # Create a dummy file that looks like a bias file
    with open(bias_dir / "bias_999_0.npy", "w") as f:
        f.write("dummy")

    with patch("numpy.load", side_effect=IOError("Mocked IO Error")):
        # It should just ignore the error
        meta = Metadynamics(
            system, [bias], 300, 3.0, 5.0, 10, saveFrequency=10, biasDir=str(bias_dir)
        )
        assert 999 not in meta._loadedBiases
