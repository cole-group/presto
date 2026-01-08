"""Functionality for computing the Hessian, closely based on https://github.com/qubekit/QUBEKit/blob/bdccc28d1ad8c0fb4eeb01467bfe8365396c411c/qubekit/engines/openmm.py#L174."""

import numpy as np
from openmm import unit
from openmm.app import Simulation

_LENGTH_UNIT = unit.nanometers
_ENERGY_UNIT = unit.kilocalorie_per_mole
_FORCE_UNIT = _ENERGY_UNIT / _LENGTH_UNIT
_HESSIAN_UNIT = _ENERGY_UNIT / (_LENGTH_UNIT**2)

_DIMENSION_TO_OFFSET = {"x": 0, "y": 1, "z": 2}


def _get_forces(simulation: Simulation, positions: np.ndarray) -> np.ndarray:
    """
    Compute the forces on the system at the given positions
    """
    simulation.context.setPositions(positions * _LENGTH_UNIT)
    return (
        simulation.context.getState(getForces=True)
        .getForces(asNumpy=True)
        .value_in_unit(_FORCE_UNIT)
    )


def calculate_hessian(
    simulation: Simulation,
    input_coords: unit.Quantity,
    finite_step: unit.Quantity = 0.0005291772 * unit.nanometers,
) -> unit.Quantity:
    """
    Using finite displacement calculate the hessian matrix of the molecule
    using a seminumerical scheme. See https://mattermodeling.stackexchange.com/questions/10358/numerical-evaluation-of-hessian
    for a brief description of the method.

    Args:
        simulation (Simulation): The OpenMM Simulation object.
        input_coords (unit.Quantity): The coordinates of the molecule.
        finite_step (unit.Quantity): The finite step size for displacement.

    Returns:
        unit.Quantity: The Hessian matrix in appropriate units.
    """

    # Perform all operations in _LENGTH_UNIT
    input_coords = input_coords.value_in_unit(_LENGTH_UNIT)
    finite_step = finite_step.value_in_unit(_LENGTH_UNIT)
    n_atoms = simulation.topology.getNumAtoms()

    # Check we have coordinates corresponding to a single conformation
    if input_coords.shape != (n_atoms, 3):
        raise ValueError(
            f"Input coordinates must have shape ({n_atoms}, 3), got {input_coords.shape}"
        )

    # Calculate the full Hessian elements with finite difference
    hessian = np.zeros((3 * n_atoms, 3 * n_atoms))

    for j in range(0, n_atoms):
        for _dimension, offset_j in _DIMENSION_TO_OFFSET.items():
            coords = input_coords.copy()
            # Get gi + hej
            coords[j, offset_j] += finite_step
            gi_plus = -_get_forces(simulation, coords)
            # Get gi - hej
            coords[j, offset_j] -= 2 * finite_step
            gi_minus = -_get_forces(simulation, coords)

            for i in range(0, n_atoms):
                for _dimension, offset_i in _DIMENSION_TO_OFFSET.items():
                    # Compute hessian element
                    hessian[i * 3 + offset_i, j * 3 + offset_j] = (
                        gi_plus[i, offset_i] - gi_minus[i, offset_i]
                    ) / (2 * finite_step)

    # Symmetrise the Hessian by averaging with its transpose
    hessian = 0.5 * (hessian + hessian.T)

    return hessian * _HESSIAN_UNIT
