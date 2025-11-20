"""
Functionality to obtain samples to fit the force field to.
"""

import copy
import functools
import pathlib
from typing import Callable, Protocol, TypedDict, Unpack

import datasets
import datasets.table
import descent.targets.energy
import loguru
import numpy
import numpy as np
import openff.interchange
import openff.toolkit
import openmm
import torch
from openff.units import unit as off_unit
from openmm import LangevinMiddleIntegrator
from openmm.app import PDBReporter, Simulation
from openmm.unit import Quantity, angstrom
from tqdm import tqdm

from . import mlp, settings
from .find_torsions import (
    _TORSIONS_TO_EXCLUDE_SMARTS,
    _TORSIONS_TO_INCLUDE_SMARTS,
    get_rot_torsions_by_rot_bond,
)
from .metadynamics import Metadynamics
from .outputs import OutputType
from .utils.register import get_registry_decorator

logger = loguru.logger

_ANGSTROM = off_unit.angstrom

_OMM_KELVIN = openmm.unit.kelvin
_OMM_PS = openmm.unit.picosecond
_OMM_ANGS = openmm.unit.angstrom
_OMM_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_OMM_KCAL_PER_MOL_ANGS = openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom


class SampleFnArgs(TypedDict):
    """Arguments for sampling functions."""

    mol: openff.toolkit.Molecule
    off_ff: openff.toolkit.ForceField
    device: torch.device
    settings: settings.SamplingSettings
    output_paths: dict[OutputType, pathlib.Path]


class SampleFn(Protocol):
    """A protocol for sampling functions."""

    def __call__(self, **kwargs: Unpack[SampleFnArgs]) -> datasets.Dataset: ...


_SAMPLING_FNS_REGISTRY: dict[type[settings.SamplingSettings], SampleFn] = {}
"""Registry of sampling functions for different sampling settings types."""

_register_sampling_fn = get_registry_decorator(_SAMPLING_FNS_REGISTRY)


def _copy_mol_and_add_conformers(
    mol: openff.toolkit.Molecule,
    n_conformers: int,
) -> openff.toolkit.Molecule:
    """Copy a molecule and add conformers to it."""
    mol = copy.deepcopy(mol)
    mol.generate_conformers(n_conformers=n_conformers, rms_cutoff=0.0 * _ANGSTROM)
    n_gen_conformers = len(mol.conformers)
    if n_gen_conformers < n_conformers:
        logger.warning(
            f"Only {n_gen_conformers} conformers were generated, which is less than the requested {n_conformers}."
            f" As a result, {n_gen_conformers / n_conformers * 100:.1f}% of the requested samples will be generated."
        )
    return mol


def _get_integrator(
    temp: openmm.unit.Quantity, timestep: openmm.unit.Quantity
) -> LangevinMiddleIntegrator:
    return LangevinMiddleIntegrator(temp, 1 / _OMM_PS, timestep)


def _run_md(
    mol: openff.toolkit.Molecule,
    simulation: Simulation,
    step_fn: Callable[[int], None],
    equilibration_n_steps_per_conformer: int,
    production_n_snapshots_per_conformer: int,
    production_n_steps_per_snapshot_per_conformer: int,
    pdb_reporter_path: str | None = None,
) -> datasets.Dataset:
    """Run MD on a molecule and return a dataset of the coordinates,
    energies, and forces of the snapshots.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The molecule to run MD on. Should have conformers already generated.

    simulation : openmm.app.Simulation
        The OpenMM simulation to use for MD.

    step_fn : Callable[[int], None]
        A function that takes the number of steps to run and runs them
        in the simulation. This is to allow for different types of MD
        (e.g. with or without metadynamics).

    equilibration_n_steps_per_conformer : int
        The number of equilibration steps to run per conformer.

    production_n_snapshots_per_conformer : int
        The number of production snapshots to take per conformer.

    production_n_steps_per_snapshot_per_conformer : int
        The number of production steps to run between each snapshot
        per conformer.

    pdb_reporter_path : str | None, optional
        The path to write a PDB trajectory of the MD
        simulation to. The frames saved correspond
        to the production snapshots. If None, no trajectory is saved.

    Returns
    -------
    datasets.Dataset
        The dataset of snapshots with coordinates, energies, and forces.
    """

    coords, energy, forces = [], [], []
    if pdb_reporter_path is not None:
        reporter = PDBReporter(
            pdb_reporter_path, production_n_steps_per_snapshot_per_conformer
        )

    for conf_idx, initial_positions in tqdm(
        enumerate(mol.conformers),
        leave=False,
        colour="green",
        desc="Generating Snapshots",
        total=len(mol.conformers),
    ):
        simulation.context.setPositions(initial_positions.to_openmm())

        # Equilibration
        simulation.minimizeEnergy(maxIterations=0)
        step_fn(equilibration_n_steps_per_conformer)

        # Production
        if pdb_reporter_path is not None:
            simulation.reporters.append(reporter)

        for _ in tqdm(
            range(production_n_snapshots_per_conformer),
            leave=False,
            colour="red",
            desc=f"Running MD for conformer {conf_idx + 1}",
        ):
            step_fn(production_n_steps_per_snapshot_per_conformer)
            state = simulation.context.getState(
                getEnergy=True, getForces=True, getPositions=True
            )
            coords.append(state.getPositions().value_in_unit(_OMM_ANGS))
            energy.append(state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL))
            forces.append(
                state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
            )

        # Remove the reporter to avoid the next round of equilibration sampling
        if pdb_reporter_path is not None:
            simulation.reporters.remove(reporter)

    # Return a Dataset with energies relative to the first snapshot
    smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
    coords_out = torch.tensor(np.array(coords))
    energy_0 = energy[0]
    energy_out = torch.tensor(np.array([x - energy_0 for x in energy]))
    forces_out = torch.tensor(np.array(forces))

    return descent.targets.energy.create_dataset(
        [
            {
                "smiles": smiles,
                "coords": coords_out,
                "energy": energy_out,
                "forces": forces_out,
            }
        ]
    )


def _get_ml_omm_system(
    mol: openff.toolkit.Molecule, mlp_name: mlp.AvailableModels
) -> openmm.System:
    """Get an OpenMM system for a molecule using a machine learning potential."""
    potential = mlp.get_mlp(mlp_name)
    # with open("/dev/null", "w") as f:
    #     with redirect_stdout(f):
    system = potential.createSystem(
        mol.to_topology().to_openmm(),
        charge=mol.total_charge.m_as(off_unit.e),
    )

    return system


def recalculate_energies_and_forces(
    dataset: datasets.Dataset, simulation: Simulation
) -> datasets.Dataset:
    """Recalculate energies and forces for a dataset using a given OpenMM simulation."""

    recalc_energies = []
    recalc_forces = []

    assert len(dataset) == 1, "Dataset should contain exactly one entry."

    entry = dataset[0]
    n_conf = len(entry["energy"])
    coords = entry["coords"].reshape(n_conf, -1, 3)

    for i in tqdm(
        range(n_conf),
        leave=False,
        colour="blue",
        desc="Recalculating energies and forces",
    ):
        my_pos = Quantity(numpy.array(coords[i]), angstrom)
        simulation.context.setPositions(my_pos)
        state = simulation.context.getState(getEnergy=True, getForces=True)
        recalc_energies.append(
            state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL)
        )
        recalc_forces.append(
            state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
        )

    return descent.targets.energy.create_dataset(
        [
            {
                "smiles": entry["smiles"],
                "coords": entry["coords"],
                "energy": torch.tensor(recalc_energies),
                "forces": torch.tensor(recalc_forces),
            }
        ]
    )


@_register_sampling_fn(settings.MMMDSamplingSettings)
def sample_mmmd(
    mol: openff.toolkit.Molecule,
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MMMDSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> datasets.Dataset:
    """Generate a dataset of samples from MD with the given MM force field,
    and energies and forces of snapshots from the ML potential.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The molecule to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field to use for sampling.
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : _SamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns
    -------
    datasets.Dataset
        The generated dataset of samples with energies and forces.
    """
    mol = _copy_mol_and_add_conformers(mol, settings.n_conformers)
    interchange = openff.interchange.Interchange.from_smirnoff(
        off_ff, openff.toolkit.Topology.from_molecules(mol)
    )
    system = interchange.to_openmm_system()
    simulation = Simulation(
        interchange.topology.to_openmm(),
        system,
        _get_integrator(settings.temperature, settings.timestep),
    )

    # First, generate the MD snapshots using the MM potential
    mm_dataset = _run_md(
        mol,
        simulation,
        simulation.step,
        settings.equilibration_n_steps_per_conformer,
        settings.production_n_snapshots_per_conformer,
        settings.production_n_steps_per_snapshot_per_conformer,
        str(output_paths.get(OutputType.PDB_TRAJECTORY, None)),
    )

    # Now, recalculate energies and forces using the ML potential
    ml_system = _get_ml_omm_system(mol, settings.ml_potential)
    ml_simulation = Simulation(
        interchange.topology,
        ml_system,
        _get_integrator(settings.temperature, settings.timestep),
    )
    ml_dataset = recalculate_energies_and_forces(mm_dataset, ml_simulation)

    return ml_dataset


@_register_sampling_fn(settings.MLMDSamplingSettings)
def sample_mlmd(
    mol: openff.toolkit.Molecule,
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MLMDSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> datasets.Dataset:
    """Generate a dataset of samples (with energies and forces) all
    from MD with an ML potential.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The molecule to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field. Kept for consistency with other sampling functions,
        but not used here.
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : _SamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns
    -------
    datasets.Dataset
        The generated dataset of samples with energies and forces.
    """
    mol = _copy_mol_and_add_conformers(mol, settings.n_conformers)
    ml_system = _get_ml_omm_system(mol, settings.ml_potential)
    integrator = _get_integrator(settings.temperature, settings.timestep)
    ml_simulation = Simulation(mol.to_topology().to_openmm(), ml_system, integrator)

    # Generate the MD snapshots using the ML potential
    ml_dataset = _run_md(
        mol,
        ml_simulation,
        ml_simulation.step,
        settings.equilibration_n_steps_per_conformer,
        settings.production_n_snapshots_per_conformer,
        settings.production_n_steps_per_snapshot_per_conformer,
        str(output_paths.get(OutputType.PDB_TRAJECTORY, None)),
    )

    return ml_dataset


def _get_torsion_bias_forces(
    mol: openff.toolkit.Molecule,
    torsions_to_include: list[str] = _TORSIONS_TO_INCLUDE_SMARTS,
    torsions_to_exclude: list[str] = _TORSIONS_TO_EXCLUDE_SMARTS,
    bias_width: float = np.pi / 10,
) -> list[openmm.app.metadynamics.BiasVariable]:
    """
    Find important torsions in a molecule and return a list of BiasVariable objects -
    one for each torsion.

    Args:
        mol: OpenFF Molecule.
        torsions_to_include: List of SMARTS patterns to include.
        torsions_to_exclude: List of SMARTS patterns to exclude.
        bias_width: Width of the bias to apply to each torsion.

    Returns:
        List of BiasVariable objects for each torsion.
    """
    torsions = get_rot_torsions_by_rot_bond(
        mol,
        include_smarts=torsions_to_include,
        exclude_smarts=torsions_to_exclude,
    )

    bias_variables = []

    for torsion in torsions.values():
        # Creat a custom torsion force for each torsion\
        torsion_force = openmm.CustomTorsionForce("theta")
        torsion_force.addTorsion(*torsion, [])

        # Create a BiasVariable for this torsion
        bias_variable = openmm.app.metadynamics.BiasVariable(
            force=torsion_force,
            biasWidth=bias_width,
            minValue=-numpy.pi,  # Torsions are periodic, so -pi to pi
            maxValue=numpy.pi,
            periodic=True,
        )

        bias_variables.append(bias_variable)

    return bias_variables


@_register_sampling_fn(settings.MMMDMetadynamicsSamplingSettings)
def sample_mmmd_metadynamics(
    mol: openff.toolkit.Molecule,
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MMMDMetadynamicsSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> datasets.Dataset:
    """Generate a dataset of samples from MD with the given MM force field
    with metadynamics samplings of the torsions. Each torsion is treated as an
    independent collective variable and biased independently. This function
    generates samples using the MM potential, and recalculates energies and
    forces of snapshots from the ML potential.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The molecule to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field to use for sampling.
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : settings.MMMDMetadynamicsSamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns
    -------
    datasets.Dataset
        The generated dataset of samples with energies and forces.
    """
    # Make sure we have all the required output paths and no others
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    mol = _copy_mol_and_add_conformers(mol, settings.n_conformers)
    interchange = openff.interchange.Interchange.from_smirnoff(
        off_ff, openff.toolkit.Topology.from_molecules(mol)
    )

    torsions = get_rot_torsions_by_rot_bond(mol)
    if not torsions:
        raise ValueError("No rotatable bonds found in the molecule.")

    # Configure metadynamics
    bias_variables = _get_torsion_bias_forces(
        mol,
        torsions_to_include=_TORSIONS_TO_INCLUDE_SMARTS,
        torsions_to_exclude=_TORSIONS_TO_EXCLUDE_SMARTS,
        bias_width=settings.bias_width,
    )

    system = interchange.to_openmm_system()

    bias_dir = output_paths[OutputType.METADYNAMICS_BIAS]
    bias_dir.mkdir()

    metad = Metadynamics(  # type: ignore[no-untyped-call]
        system=system,
        variables=bias_variables,
        temperature=settings.temperature,
        biasFactor=settings.bias_factor,
        height=settings.bias_height,
        frequency=settings.n_steps_per_bias,
        saveFrequency=settings.n_steps_per_bias_save,
        biasDir=bias_dir,
        independentCVs=True,
    )

    simulation = Simulation(
        interchange.topology.to_openmm(),
        system,
        _get_integrator(settings.temperature, settings.timestep),
    )

    step_fn = functools.partial(metad.step, simulation)

    # First, generate the MD snapshots using the MM potential
    mm_dataset = _run_md(
        mol,
        simulation,
        step_fn,
        settings.equilibration_n_steps_per_conformer,
        settings.production_n_snapshots_per_conformer,
        settings.production_n_steps_per_snapshot_per_conformer,
        str(output_paths.get(OutputType.PDB_TRAJECTORY, None)),
    )

    # Now, recalculate energies and forces using the ML potential
    ml_system = _get_ml_omm_system(mol, settings.ml_potential)
    ml_simulation = Simulation(
        interchange.topology.to_openmm(),
        ml_system,
        _get_integrator(settings.temperature, settings.timestep),
    )
    ml_dataset = recalculate_energies_and_forces(mm_dataset, ml_simulation)

    return ml_dataset


def _add_torsion_restraint(
    simulation: Simulation,
    torsion_atoms: tuple[int, int, int, int],
    target_angle: float,
    force_constant: openmm.unit.Quantity,
) -> int:
    """Add a harmonic torsion restraint to the simulation system.

    Parameters
    ----------
    simulation : Simulation
        OpenMM simulation object
    torsion_atoms : tuple[int, int, int, int]
        Indices of atoms defining the torsion to restrain
    target_angle : float
        Target angle (in radians) for the restraint
    force_constant : openmm.unit.Quantity
        Force constant (in kJ/mol/rad^2) for the restraint

    Returns
    -------
    int
        Index of the added force in the system
    """
    force_constant_in_kJ_per_mol_rad2 = force_constant / (
        openmm.unit.kilojoule_per_mole / openmm.unit.radian**2
    )

    restraint_force = openmm.CustomTorsionForce(
        f"0.5*{force_constant_in_kJ_per_mol_rad2}*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535"
    )
    restraint_force.addPerTorsionParameter("theta0")
    restraint_force.addTorsion(*torsion_atoms, [target_angle])
    force_idx: int = simulation.system.addForce(restraint_force)
    simulation.context.reinitialize(preserveState=True)
    return force_idx


def _remove_torsion_restraint(simulation: Simulation, force_idx: int) -> None:
    """Remove a torsion restraint from the simulation system.

    Parameters
    ----------
    simulation : Simulation
        OpenMM simulation object
    force_idx : int
        Index of the force to remove
    """
    simulation.system.removeForce(force_idx)
    simulation.context.reinitialize(preserveState=True)


def _calculate_torsion_angles(
    coords: torch.Tensor, torsion_atoms: tuple[int, int, int, int]
) -> torch.Tensor:
    """Calculate torsion angles for a given set of atom indices.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates tensor of shape (n_snapshots, n_atoms, 3)
    torsion_atoms : tuple[int, int, int, int]
        Indices of the four atoms defining the torsion

    Returns
    -------
    torch.Tensor
        Torsion angles in radians for each snapshot, shape (n_snapshots,)
    """
    i, j, k, m = torsion_atoms

    # Extract coordinates for the four atoms
    r1 = coords[:, i, :]
    r2 = coords[:, j, :]
    r3 = coords[:, k, :]
    r4 = coords[:, m, :]

    # Calculate vectors
    b1 = r2 - r1
    b2 = r3 - r2
    b3 = r4 - r3

    # Calculate normal vectors to the planes
    n1 = torch.cross(b1, b2, dim=1)
    n2 = torch.cross(b2, b3, dim=1)

    # Calculate the torsion angle
    m1 = torch.cross(n1, b2 / torch.norm(b2, dim=1, keepdim=True), dim=1)

    x = torch.sum(n1 * n2, dim=1)
    y = torch.sum(m1 * n2, dim=1)

    return torch.atan2(y, x)


def _select_diverse_samples_from_bin(
    sample_indices: list[int], max_samples: int
) -> list[int]:
    """Select diverse samples from a bin by maximizing spacing in trajectory.

    Parameters
    ----------
    sample_indices : list[int]
        Indices of samples in the bin
    max_samples : int
        Maximum number of samples to select

    Returns
    -------
    list[int]
        Selected sample indices
    """
    if len(sample_indices) <= max_samples:
        return sample_indices

    # Select samples evenly spaced in the trajectory
    # This ensures diversity by sampling from different parts of the
    # trajectory
    step = len(sample_indices) / max_samples
    selected = [sample_indices[int(i * step)] for i in range(max_samples)]

    return selected


def _select_seed_conformations_for_torsion(
    dataset: datasets.Dataset,
    torsion_atoms: tuple[int, int, int, int],
    n_angle_bins: int,
    n_samples_per_bin: int,
    simulation: Simulation,
    frozen_force_constant: openmm.unit.Quantity,
) -> dict[int, tuple[numpy.ndarray, float, float]]:
    """Select seed conformations for a torsion by binning, minimizing, and
    selecting lowest energy.

    Parameters
    ----------
    dataset : datasets.Dataset
        Dataset containing snapshots from metadynamics sampling
    torsion_atoms : tuple[int, int, int, int]
        Indices of atoms defining the torsion
    n_angle_bins : int
        Number of bins to divide the angle range into
    n_samples_per_bin : int
        Maximum number of diverse samples to select per bin for minimization
    simulation : Simulation
        OpenMM simulation object for minimization
    frozen_force_constant : openmm.unit.Quantity
        Force constant (in kJ/mol/rad^2) for freezing the torsion

    Returns
    -------
    dict[int, tuple[numpy.ndarray, float, float]]
        Dictionary mapping bin index to (coordinates, energy, target_angle)
        for the lowest energy conformation in that bin
    """
    assert len(dataset) == 1, "Dataset should contain exactly one entry."
    entry = dataset[0]

    n_snapshots = len(entry["energy"])
    coords = entry["coords"].reshape(n_snapshots, -1, 3)

    # Calculate torsion angles for all snapshots
    torsion_angles = _calculate_torsion_angles(coords, torsion_atoms)

    # Bin the samples
    bin_edges = torch.linspace(-numpy.pi, numpy.pi, n_angle_bins + 1)
    bin_indices = torch.searchsorted(bin_edges[:-1], torsion_angles) - 1
    bin_indices = torch.clamp(bin_indices, 0, n_angle_bins - 1)

    seed_conformations = {}

    for bin_idx in tqdm(
        range(n_angle_bins),
        leave=False,
        colour="magenta",
        desc=f"Selecting seeds for torsion {torsion_atoms}",
    ):
        # Get samples in this bin
        samples_in_bin = (bin_indices == bin_idx).nonzero(as_tuple=True)[0]
        if len(samples_in_bin) == 0:
            continue

        # Select diverse samples from the bin
        diverse_samples = _select_diverse_samples_from_bin(
            samples_in_bin.tolist(), n_samples_per_bin
        )

        # Calculate target angle as bin center
        target_angle = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]).item() / 2.0

        # Minimize each sample with frozen torsion and select lowest energy
        min_energy = float("inf")
        best_coords = None

        for sample_idx in diverse_samples:
            # Set positions
            sample_coords = coords[sample_idx].numpy()
            simulation.context.setPositions(Quantity(sample_coords, angstrom))

            # Get the initial energy
            state = simulation.context.getState(getEnergy=True)

            # Add restraint force to freeze torsion at target angle
            force_idx = _add_torsion_restraint(
                simulation, torsion_atoms, target_angle, frozen_force_constant
            )

            # Minimize
            simulation.minimizeEnergy(maxIterations=0)

            # Get energy
            state = simulation.context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL)

            if energy < min_energy:
                min_energy = energy
                best_coords = state.getPositions().value_in_unit(_OMM_ANGS)

            # Remove restraint force for next iteration
            _remove_torsion_restraint(simulation, force_idx)

        if best_coords is not None:
            seed_conformations[bin_idx] = (
                best_coords,
                min_energy,
                target_angle,
            )

    return seed_conformations


def _run_md_with_frozen_torsion(
    mol: openff.toolkit.Molecule,
    simulation: Simulation,
    torsion_atoms: tuple[int, int, int, int],
    target_angle: float,
    frozen_force_constant: openmm.unit.Quantity,
    initial_positions: numpy.ndarray,
    equilibration_n_steps: int,
    production_n_snapshots: int,
    production_n_steps_per_snapshot: int,
    pdb_reporter_path: str | None = None,
) -> datasets.Dataset:
    """Run MD with a frozen torsion and collect snapshots.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The molecule being sampled
    simulation : Simulation
        OpenMM simulation object
    torsion_atoms : tuple[int, int, int, int]
        Indices of atoms defining the torsion to freeze
    target_angle : float
        Target angle (in radians) to freeze the torsion at
    frozen_force_constant : openmm.unit.Quantity
        Force constant (in kJ/mol/rad^2) for freezing the torsion
    initial_positions : numpy.ndarray
        Initial coordinates
    equilibration_n_steps : int
        Number of equilibration steps
    production_n_snapshots : int
        Number of production snapshots
    production_n_steps_per_snapshot : int
        Number of steps between snapshots
    pdb_reporter_path : str | None, optional
        The path to write a PDB trajectory of the MD

    Returns
    -------
    datasets.Dataset
        Dataset containing the snapshots
    """
    # Add restraint force to freeze torsion
    force_idx = _add_torsion_restraint(
        simulation, torsion_atoms, target_angle, frozen_force_constant
    )

    # Create a temporary molecule with single conformer for _run_md
    temp_mol = copy.deepcopy(mol)

    # Convert initial positions to OpenFF Quantity and set as the only conformer
    temp_mol._conformers = [off_unit.Quantity(np.array(initial_positions), _ANGSTROM)]

    # Run MD using the standard function
    dataset = _run_md(
        temp_mol,
        simulation,
        simulation.step,
        equilibration_n_steps,
        production_n_snapshots,
        production_n_steps_per_snapshot,
        pdb_reporter_path=pdb_reporter_path,
    )

    # Remove restraint force
    _remove_torsion_restraint(simulation, force_idx)

    return dataset


@_register_sampling_fn(settings.MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings)
def sample_mmmd_metadynamics_seeded_frozen_torsions(
    mol: openff.toolkit.Molecule,
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> datasets.Dataset:
    """Generate a dataset using metadynamics to create seeds, then MD with
    frozen torsions.

    This function:
    1. Runs metadynamics on all rotatable bonds to generate initial samples
    2. For each torsion, bins samples by angle, selects diverse samples per
       bin, freezes the torsion, minimizes, and selects the lowest energy as
       the seed
    3. For each seed, runs MD with the torsion frozen to generate training
       samples

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The molecule to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field to use for sampling.
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : settings.MMMDMetadynamicsSeededFrozenTorsionsSamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns
    -------
    datasets.Dataset
        The generated dataset of samples with energies and forces.
    """
    # Make sure we have all the required output paths
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    logger.info("Step 1: Running metadynamics to generate initial samples")

    mol = _copy_mol_and_add_conformers(mol, settings.n_conformers)
    interchange = openff.interchange.Interchange.from_smirnoff(
        off_ff, openff.toolkit.Topology.from_molecules(mol)
    )

    torsions = get_rot_torsions_by_rot_bond(mol)
    if not torsions:
        raise ValueError("No rotatable bonds found in the molecule.")

    # Configure metadynamics
    bias_variables = _get_torsion_bias_forces(
        mol,
        torsions_to_include=_TORSIONS_TO_INCLUDE_SMARTS,
        torsions_to_exclude=_TORSIONS_TO_EXCLUDE_SMARTS,
        bias_width=settings.bias_width,
    )

    system = interchange.to_openmm_system()

    bias_dir = output_paths[OutputType.METADYNAMICS_BIAS]
    bias_dir.mkdir()

    metad = Metadynamics(  # type: ignore[no-untyped-call]
        system=system,
        variables=bias_variables,
        temperature=settings.temperature,
        biasFactor=settings.bias_factor,
        height=settings.bias_height,
        frequency=settings.n_steps_per_bias,
        saveFrequency=settings.n_steps_per_bias_save,
        biasDir=bias_dir,
        independentCVs=True,
    )

    simulation = Simulation(
        interchange.topology.to_openmm(),
        system,
        _get_integrator(settings.temperature, settings.timestep),
    )

    step_fn = functools.partial(metad.step, simulation)

    # Generate initial samples with metadynamics
    initial_dataset = _run_md(
        mol,
        simulation,
        step_fn,
        settings.equilibration_n_steps_per_conformer,
        settings.production_n_snapshots_per_conformer,
        settings.production_n_steps_per_snapshot_per_conformer,
        None,
    )

    logger.info("Step 2: Selecting seed conformations for each torsion and bin")

    # Create a fresh simulation for minimization (without metadynamics)
    minimization_system = interchange.to_openmm_system()
    minimization_simulation = Simulation(
        interchange.topology.to_openmm(),
        minimization_system,
        _get_integrator(settings.temperature, settings.timestep),
    )

    # Select seed conformations for each torsion
    torsion_seeds = {}
    for torsion_name, torsion_atoms in tqdm(
        torsions.items(),
        desc="Processing torsions",
        colour="cyan",
    ):
        seeds = _select_seed_conformations_for_torsion(
            initial_dataset,
            torsion_atoms,
            settings.n_angle_bins,
            settings.n_samples_per_bin,
            minimization_simulation,
            settings.frozen_torsion_force_constant,
        )
        torsion_seeds[torsion_name] = (torsion_atoms, seeds)

    logger.info("Step 3: Running MD with frozen torsions for each seed conformation")

    # Create a simulation for frozen torsion MD
    frozen_md_system = interchange.to_openmm_system()
    frozen_md_simulation = Simulation(
        interchange.topology.to_openmm(),
        frozen_md_system,
        _get_integrator(settings.temperature, settings.timestep),
    )

    # Collect all datasets from frozen MD runs
    all_datasets = []

    for _torsion_name, (torsion_atoms, seeds) in tqdm(
        torsion_seeds.items(),
        desc="Running frozen MD",
        colour="yellow",
    ):
        for bin_idx, (seed_coords, _seed_energy, target_angle) in seeds.items():
            dataset = _run_md_with_frozen_torsion(
                mol,
                frozen_md_simulation,
                torsion_atoms,
                target_angle,
                settings.frozen_torsion_force_constant,
                seed_coords,
                settings.frozen_equilibration_n_steps_per_seed,
                settings.frozen_production_n_snapshots_per_seed,
                settings.frozen_production_n_steps_per_snapshot_per_seed,
                str(output_paths.get(OutputType.PDB_TRAJECTORY, None))
                + f"_{torsion_atoms}_bin{bin_idx}.pdb",
            )
            all_datasets.append(dataset)

    # Concatenate all datasets
    if len(all_datasets) == 0:
        raise ValueError("No seed conformations were generated.")

    # Merge all datasets into one
    smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
    all_coords = []
    all_energies = []
    all_forces = []

    for ds in all_datasets:
        entry = ds[0]
        n_conf = len(entry["energy"])
        coords = entry["coords"].reshape(n_conf, -1, 3)
        all_coords.append(coords)
        all_energies.append(entry["energy"])
        all_forces.append(entry["forces"].reshape(n_conf, -1, 3))

    coords_out = torch.cat(all_coords, dim=0)
    energy_out = torch.cat(all_energies, dim=0)
    forces_out = torch.cat(all_forces, dim=0)

    # Make energies relative to minimum
    energy_out = energy_out - energy_out.min()

    mm_dataset = descent.targets.energy.create_dataset(
        [
            {
                "smiles": smiles,
                "coords": coords_out,
                "energy": energy_out,
                "forces": forces_out,
            }
        ]
    )

    logger.info("Step 4: Recalculating energies and forces with ML potential")

    # Recalculate energies and forces using ML potential
    ml_system = _get_ml_omm_system(mol, settings.ml_potential)
    ml_simulation = Simulation(
        interchange.topology.to_openmm(),
        ml_system,
        _get_integrator(settings.temperature, settings.timestep),
    )
    ml_dataset = recalculate_energies_and_forces(mm_dataset, ml_simulation)

    return ml_dataset
