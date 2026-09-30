"""Generalised Born implicit solvent for the MM systems used during sampling.

OpenFF Interchange does not implement the SMIRNOFF ``GBSA`` handlers, so the solvent
term is added to the OpenMM ``System`` after it has been exported. The force is built
with `openmm.app.internal.customgbforces`, the same machinery
``AmberPrmtopFile.createSystem(implicitSolvent=...)`` uses, with radii and screening
parameters taken from the Amber tables.

Note that those tables only cover the elements commonly found in biomolecules. Atoms of
other elements (bromine, iodine, ...) fall back to a default radius of 1.5 Å and a
default screening parameter of 0.8.

This is only ever applied to the MM system used to generate configurations. See
`presto.settings.ImplicitSolventSettings`.
"""

import math

import loguru
import openmm
import openmm.app
from openmm.app.internal import customgbforces

from . import settings

logger = loguru.logger

_GB_FORCE_CLASSES: dict[str, type[customgbforces.CustomAmberGBForceBase]] = {
    "hct": customgbforces.GBSAHCTForce,
    "obc1": customgbforces.GBSAOBC1Force,
    "obc2": customgbforces.GBSAOBC2Force,
    "gbn": customgbforces.GBSAGBnForce,
    "gbn2": customgbforces.GBSAGBn2Force,
}
"""The generalised Born models which can be used, keyed by the identifier used in
`presto.settings.ImplicitSolventSettings.model`."""

_OMM_KELVIN = openmm.unit.kelvin
_OMM_MOLAR = openmm.unit.molar


def _get_kappa(
    salt_concentration: openmm.unit.Quantity,
    solvent_dielectric: float,
    temperature: openmm.unit.Quantity,
) -> float:
    """Convert a salt concentration to the Debye screening parameter kappa (1 / nm).

    This uses the same conversion as OpenMM's
    ``AmberPrmtopFile.createSystem(implicitSolventSaltConc=...)``.

    Parameters
    ----------
    salt_concentration : openmm.unit.Quantity
        The concentration of monovalent salt.

    solvent_dielectric : float
        The dielectric constant of the solvent.

    temperature : openmm.unit.Quantity
        The temperature the sampling is run at.

    Returns:
    -------
    float
        The Debye screening parameter in units of 1 / nm.
    """
    salt_molar = salt_concentration.value_in_unit(_OMM_MOLAR)
    temperature_kelvin = temperature.value_in_unit(_OMM_KELVIN)

    # The constant is Amber's kappa conversion factor (in 1 / angstrom); the factor of
    # 7.3 is 0.73 to account for ion exclusions and 10 to convert to 1 / nm.
    return (
        50.33355 * math.sqrt(salt_molar / solvent_dielectric / temperature_kelvin) * 7.3
    )


def _get_charges(system: openmm.System) -> list[float]:
    """Get the partial charges (in units of the proton charge) of each particle.

    The charges are read back from the system so that they always match the ones
    actually used by the MM force field, including any custom or library charges.
    """
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            return [
                force.getParticleParameters(i)[0].value_in_unit(
                    openmm.unit.elementary_charge
                )
                for i in range(force.getNumParticles())
            ]

    raise ValueError(
        "Cannot add an implicit solvent force to a system without a NonbondedForce, "
        "as the partial charges cannot be determined."
    )


def add_implicit_solvent_force(
    system: openmm.System,
    topology: openmm.app.Topology,
    implicit_solvent: settings.ImplicitSolventSettings,
    temperature: openmm.unit.Quantity,
) -> openmm.CustomGBForce:
    """Add a generalised Born implicit solvent force to an OpenMM system in place.

    Parameters
    ----------
    system : openmm.System
        The system to add the force to. Modified in place.

    topology : openmm.app.Topology
        The topology corresponding to `system`, used to assign the per-atom radii and
        screening parameters.

    implicit_solvent : settings.ImplicitSolventSettings
        The settings defining the generalised Born model to use.

    temperature : openmm.unit.Quantity
        The temperature the sampling is run at, used to convert the salt concentration
        to a Debye screening parameter.

    Returns:
    -------
    openmm.CustomGBForce
        The force which was added to the system.

    Raises:
    ------
    ValueError
        If the system has no `openmm.NonbondedForce`, or if the number of particles in
        the system does not match the number of atoms in the topology (for example if
        the force field adds virtual sites).
    """
    n_atoms = topology.getNumAtoms()
    if system.getNumParticles() != n_atoms:
        raise ValueError(
            f"Cannot add an implicit solvent force to a system with "
            f"{system.getNumParticles()} particles using a topology with {n_atoms} "
            "atoms. Implicit solvent is not supported for force fields with virtual "
            "sites."
        )

    charges = _get_charges(system)

    force_class = _GB_FORCE_CLASSES[implicit_solvent.model]
    force = force_class(
        solventDielectric=implicit_solvent.solvent_dielectric,
        soluteDielectric=implicit_solvent.solute_dielectric,
        SA=implicit_solvent.surface_area_model,
        # The systems sampled here are always non-periodic single molecules in a
        # (notional) infinite box, so no cutoff is applied.
        cutoff=None,
        kappa=_get_kappa(
            implicit_solvent.salt_concentration,
            implicit_solvent.solvent_dielectric,
            temperature,
        ),
    )

    # The standard parameters are the per-atom radii and screening parameters (plus
    # alpha, beta and gamma for the GBn2 model); the charge is prepended to these.
    standard_parameters = force_class.getStandardParameters(topology)
    for charge, parameters in zip(charges, standard_parameters, strict=True):
        force.addParticle([charge, *parameters])

    # Required by the custom GB forces before they can be added to a system.
    force.finalize()
    system.addForce(force)

    logger.debug(
        f"Added a {implicit_solvent.model} implicit solvent force with a solvent "
        f"dielectric of {implicit_solvent.solvent_dielectric} and a solute dielectric "
        f"of {implicit_solvent.solute_dielectric}."
    )

    return force
