"""Additive energy and force corrections, such as implicit-solvent terms.

A correction adds an energy and forces to either the reference (e.g. MLP) or the MM
model. Each correction can be used in two independent ways:

- During sampling, as an OpenMM force added to the systems which propagate dynamics
  or run minimisations (`add_correction_forces`). These forces are placed in their
  own force group so that recorded energies and forces can exclude them.
- In the fit, evaluated on the final coordinates of a dataset and stored in its
  correction columns (`apply_corrections_to_dataset`), which
  `presto.loss.predict_with_weights` adds to the reference and MM values.

New corrections are added by defining a settings class in `presto.settings` and
registering an `EnergyCorrection` subclass for it with `_register_correction`.
"""

from __future__ import annotations

import functools
import importlib
import typing
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Literal

import datasets
import loguru
import numpy as np
import openmm
import openmm.unit
import smee
import torch
from openff.toolkit import Molecule
from openff.units import unit as off_unit
from openmm.app import element as omm_element
from openmm.app.internal.customgbforces import GBSAGBn2Force
from openmm.app.internal.lcpo import getLCPOParamsTopology

from . import settings as presto_settings
from ._exceptions import MissingOptionalDependencyError
from .data_utils import (
    WeightedEntry,
    create_weighted_dataset,
    get_corrections_from_entry,
)
from .utils.gpu import free_gpu_memory
from .utils.register import get_registry_decorator

logger = loguru.logger

CorrectionSettings = (
    presto_settings.ReferenceCorrectionSettings | presto_settings.MMCorrectionSettings
)

CORRECTION_FORCE_NAME_PREFIX = "presto.correction."
"""Prefix of the names of every force added by `add_correction_forces`."""

_EV_TO_KCAL_PER_MOL = 23.060548
_KCAL_TO_KJ = 4.184
_NM_TO_ANGSTROM = 10.0

_OMM_NM = openmm.unit.nanometer
_OMM_KJ_PER_MOL = openmm.unit.kilojoules_per_mole
_OMM_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_OMM_KCAL_PER_MOL_ANGS = openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom


class EnergyCorrection(ABC):
    """An additive energy and force correction for one molecule.

    Subclasses must implement `create_openmm_forces`. The default `compute`
    evaluates those forces with OpenMM; subclasses may override it with a faster
    direct implementation.
    """

    side: ClassVar[Literal["reference", "mm"]]
    """Whether the correction applies to the reference or the MM model."""

    def __init__(
        self,
        settings: CorrectionSettings,
        mol: Molecule,
        *,
        device: torch.device,
        charges: np.ndarray | None = None,
    ) -> None:
        """Create the correction.

        Args:
            settings: The settings for this correction.
            mol: The molecule the correction is applied to.
            device: The device to evaluate the correction on.
            charges: The partial charges [e] of the MM model, one per atom, for
                corrections which depend on them.
        """
        self.settings = settings
        self.mol = mol
        self.device = device
        self.charges = None if charges is None else np.asarray(charges, dtype=float)
        self._context: openmm.Context | None = None

    @abstractmethod
    def create_openmm_forces(self) -> list[openmm.Force]:
        """Return new OpenMM forces implementing this correction for the molecule."""

    def compute(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the correction for a set of conformations.

        Args:
            coords: Coordinates [Å] with ``shape=(n_confs, n_atoms, 3)``.

        Returns:
            Tuple of energies [kcal/mol] with ``shape=(n_confs,)`` and forces
            [kcal/mol/Å] with ``shape=(n_confs, n_atoms, 3)``.
        """
        coords = np.asarray(coords, dtype=float).reshape(len(coords), -1, 3)
        context = self._get_context()
        energies, forces = [], []
        for conformer in coords:
            context.setPositions(conformer / _NM_TO_ANGSTROM)
            state = context.getState(getEnergy=True, getForces=True)
            energies.append(state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL))
            forces.append(
                state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
            )
        return np.array(energies), np.array(forces).reshape(coords.shape)

    def _get_context(self) -> openmm.Context:
        """Build (once) an OpenMM context containing only this correction's forces."""
        if self._context is None:
            system = openmm.System()
            for _ in range(self.mol.n_atoms):
                system.addParticle(1.0)
            for force in self.create_openmm_forces():
                system.addForce(force)
            self._context = _create_context(system, self.device)
        return self._context

    @classmethod
    def prepare(cls, settings: CorrectionSettings) -> None:
        """Check that the correction can be evaluated and fetch anything it needs.

        Called once in the parent process before any sampling, so that missing
        dependencies fail fast and downloads are not raced by worker processes.
        """
        return None

    @classmethod
    def release(cls) -> None:
        """Release any cached models or memory held by this correction type."""
        return None


_CORRECTIONS_REGISTRY: dict[type, type[EnergyCorrection]] = {}
"""Registry of correction implementations for different correction settings types."""

_register_correction = get_registry_decorator(_CORRECTIONS_REGISTRY)


def _create_context(system: openmm.System, device: torch.device) -> openmm.Context:
    """Create a context on the platform matching ``device``, falling back to CPU."""
    integrator = openmm.VerletIntegrator(1.0)
    if device.type == "cuda":
        try:
            properties = {"Precision": "mixed"}
            if device.index is not None:
                properties["DeviceIndex"] = str(device.index)
            return openmm.Context(
                system,
                integrator,
                openmm.Platform.getPlatformByName("CUDA"),
                properties,
            )
        except Exception as exc:
            logger.warning(
                f"Could not evaluate corrections with the OpenMM CUDA platform ({exc}). "
                "Falling back to the CPU platform."
            )
            integrator = openmm.VerletIntegrator(1.0)
    return openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))


def get_system_charges(system: openmm.System) -> np.ndarray:
    """Return the partial charges [e] of every particle in a system's NonbondedForce."""
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            return np.array(
                [
                    force.getParticleParameters(i)[0].value_in_unit(
                        openmm.unit.elementary_charge
                    )
                    for i in range(force.getNumParticles())
                ]
            )
    raise ValueError("The system has no NonbondedForce to take charges from.")


def get_smee_charges(
    topology: smee.TensorTopology, force_field: smee.TensorForceField
) -> np.ndarray:
    """Return the partial charges [e] smee assigns to each particle of a topology.

    These are the charges used for the MM model in the fit, so MM corrections which
    depend on charges are evaluated with them.
    """
    potential = force_field.potentials_by_type["Electrostatics"]
    parameter_map = topology.parameters["Electrostatics"]
    parameters = (
        parameter_map.assignment_matrix.to_dense().to(potential.parameters)
        @ potential.parameters.detach()
    )
    charge_col = potential.parameter_cols.index("charge")
    return parameters[:, charge_col].cpu().numpy().astype(float)


def build_correction(
    settings: CorrectionSettings,
    mol: Molecule,
    *,
    device: torch.device,
    charges: np.ndarray | None = None,
) -> EnergyCorrection:
    """Create the correction implementation registered for ``settings``."""
    return _CORRECTIONS_REGISTRY[type(settings)](
        settings, mol, device=device, charges=charges
    )


def prepare_corrections(corrections: Iterable[CorrectionSettings]) -> None:
    """Check every correction can be evaluated, failing fast if one cannot."""
    for correction in corrections:
        _CORRECTIONS_REGISTRY[type(correction)].prepare(correction)


def release_corrections(corrections: Iterable[CorrectionSettings]) -> None:
    """Release cached models and memory held by the given corrections."""
    for correction_cls in {_CORRECTIONS_REGISTRY[type(c)] for c in corrections}:
        correction_cls.release()


def resolve_sampling_corrections(
    value: Sequence[CorrectionSettings] | Literal["from_fit"],
    fit_corrections: Sequence[CorrectionSettings],
) -> list[CorrectionSettings]:
    """Resolve a sampling stage's corrections, replacing 'from_fit' by the fit's."""
    return list(fit_corrections) if value == "from_fit" else list(value)


def add_correction_forces(
    system: openmm.System,
    mol: Molecule,
    corrections: Sequence[CorrectionSettings],
    *,
    device: torch.device,
    charges: np.ndarray | None = None,
) -> int | None:
    """Add the forces of each correction to a system, in a dedicated force group.

    Args:
        system: The system to add the forces to.
        mol: The molecule the system describes.
        corrections: The corrections to add.
        device: The device to evaluate any ML corrections on.
        charges: The partial charges [e] for corrections which depend on them.

    Returns:
        The force group the corrections were added to, or None if there were none.
    """
    if not corrections:
        return None

    used_groups = {force.getForceGroup() for force in system.getForces()}
    free_groups = [group for group in range(32) if group not in used_groups]
    if not free_groups:
        raise RuntimeError("All force groups (0-31) are in use")
    group = free_groups[0]

    for correction_settings in corrections:
        correction = build_correction(
            correction_settings, mol, device=device, charges=charges
        )
        for force in correction.create_openmm_forces():
            force.setName(
                f"{CORRECTION_FORCE_NAME_PREFIX}{correction_settings.correction_type}"
            )
            force.setForceGroup(group)
            system.addForce(force)

    logger.debug(f"Added {len(corrections)} correction(s) to force group {group}")
    return group


def correction_force_groups(system: openmm.System) -> set[int]:
    """Return the force groups holding forces added by `add_correction_forces`."""
    return {
        force.getForceGroup()
        for force in system.getForces()
        if force.getName().startswith(CORRECTION_FORCE_NAME_PREFIX)
    }


def compute_corrections(
    corrections: Sequence[EnergyCorrection], coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum the energies [kcal/mol] and forces [kcal/mol/Å] of several corrections."""
    coords = np.asarray(coords, dtype=float).reshape(len(coords), -1, 3)
    energies = np.zeros(len(coords))
    forces = np.zeros(coords.shape)
    for correction in corrections:
        correction_energies, correction_forces = correction.compute(coords)
        energies += correction_energies
        forces += correction_forces
    return energies, forces


def apply_corrections_to_dataset(
    dataset: datasets.Dataset,
    mol: Molecule,
    *,
    reference_corrections: Sequence[presto_settings.ReferenceCorrectionSettings],
    mm_corrections: Sequence[presto_settings.MMCorrectionSettings],
    device: torch.device,
    mm_charges: np.ndarray | None = None,
    reference_mode: Literal["keep", "recompute", "zero"] = "recompute",
) -> datasets.Dataset:
    """Set the correction columns of a dataset from the given corrections.

    The ``energy`` and ``forces`` columns are left untouched. MM corrections are
    always recomputed (zero if there are none).

    Args:
        dataset: The weighted dataset for ``mol``.
        mol: The molecule the dataset describes.
        reference_corrections: The corrections to the reference.
        mm_corrections: The corrections to the MM model.
        device: The device to evaluate the corrections on.
        mm_charges: The MM partial charges [e], one per atom, used by MM
            corrections which depend on them.
        reference_mode: 'recompute' evaluates ``reference_corrections``, 'keep' keeps
            the stored reference corrections and 'zero' sets them to zero.

    Returns:
        A new in-memory dataset with the correction columns set.
    """
    reference_terms = [
        build_correction(c, mol, device=device) for c in reference_corrections
    ]
    mm_terms = [
        build_correction(c, mol, device=device, charges=mm_charges)
        for c in mm_corrections
    ]

    entries = []
    for entry in dataset:
        n_confs = len(entry["energy"])
        coords = torch.as_tensor(entry["coords"]).cpu().reshape(n_confs, -1, 3).numpy()
        ref_e, ref_f, _, _ = get_corrections_from_entry(entry)

        if reference_mode == "recompute":
            ref_e_np, ref_f_np = compute_corrections(reference_terms, coords)
            # The reference energies may have been evaluated at other coordinates
            # than the forces (see `WeightedEntry.reference_energy_coords`)
            energy_coords = entry.get("reference_energy_coords")
            if energy_coords is not None and reference_terms:
                energy_coords = torch.as_tensor(energy_coords).cpu().numpy()
                ref_e_np, _ = compute_corrections(
                    reference_terms, energy_coords.reshape(n_confs, -1, 3)
                )
            ref_e, ref_f = torch.from_numpy(ref_e_np), torch.from_numpy(ref_f_np)
        elif reference_mode == "zero":
            ref_e, ref_f = torch.zeros_like(ref_e), torch.zeros_like(ref_f)
        elif reference_mode != "keep":
            raise ValueError(f"Unknown reference correction mode {reference_mode}")

        mm_e, mm_f = compute_corrections(mm_terms, coords)

        new_entry = {
            key: (value.cpu() if isinstance(value, torch.Tensor) else value)
            for key, value in entry.items()
        }
        new_entry["reference_energy_correction"] = ref_e
        new_entry["reference_forces_correction"] = ref_f
        new_entry["mm_energy_correction"] = torch.from_numpy(mm_e)
        new_entry["mm_forces_correction"] = torch.from_numpy(mm_f)
        entries.append(new_entry)

    return create_weighted_dataset(typing.cast(list[WeightedEntry], entries))


# ---------------------------------------------------------------------------
# GBneck2 + SASA (MM side)
# ---------------------------------------------------------------------------

_GBN2_PARAMETERISED_ELEMENTS = {
    omm_element.hydrogen,
    omm_element.carbon,
    omm_element.nitrogen,
    omm_element.oxygen,
    omm_element.sulfur,
}


@_register_correction(presto_settings.GBn2SASACorrectionSettings)
class GBn2SASACorrection(EnergyCorrection):
    """GBneck2 (Amber igb=8) generalised Born energy plus a surface area term."""

    side = "mm"
    settings: presto_settings.GBn2SASACorrectionSettings

    def __init__(
        self,
        settings: presto_settings.GBn2SASACorrectionSettings,
        mol: Molecule,
        *,
        device: torch.device,
        charges: np.ndarray | None = None,
    ) -> None:
        """Create the correction. ``charges`` (one per atom) are required."""
        super().__init__(settings, mol, device=device, charges=charges)
        if self.charges is None:
            raise ValueError("The GBn2 correction requires the MM partial charges.")
        if len(self.charges) != mol.n_atoms:
            raise ValueError(
                f"The GBn2 correction requires one charge per atom, but got "
                f"{len(self.charges)} charges for {mol.n_atoms} atoms. Virtual sites "
                "are not supported."
            )

    def create_openmm_forces(self) -> list[openmm.Force]:
        """Return the GBneck2 force and, if requested, the LCPO surface area force."""
        settings = self.settings
        topology = self.mol.to_topology().to_openmm()

        unparameterised = sorted(
            {
                atom.element.symbol
                for atom in topology.atoms()
                if atom.element not in _GBN2_PARAMETERISED_ELEMENTS
            }
        )
        if unparameterised:
            logger.warning(
                f"GBneck2 has no alpha/beta/gamma parameters for {unparameterised}, so "
                "OpenMM's defaults are used for these elements."
            )

        gb_force = GBSAGBn2Force(
            solventDielectric=settings.solvent_dielectric,
            soluteDielectric=settings.solute_dielectric,
            SA="ACE" if settings.sasa_model == "ace" else None,
            cutoff=None,
            kappa=0.0,
        )
        gb_parameters = GBSAGBn2Force.getStandardParameters(topology)
        assert self.charges is not None  # Checked on construction
        gb_force.addParticles(
            np.column_stack([self.charges, np.asarray(gb_parameters)])
        )
        gb_force.finalize()
        forces: list[openmm.Force] = [gb_force]

        if settings.sasa_model == "lcpo":
            forces.append(self._create_lcpo_force(topology))

        return forces

    def _create_lcpo_force(self, topology: openmm.app.Topology) -> openmm.LCPOForce:
        """Create an LCPO force, as in `openmm.app.internal.lcpo.addLCPOForce`."""
        try:
            lcpo_parameters = getLCPOParamsTopology(topology)
        except ValueError as exc:
            raise ValueError(
                f"LCPO surface area parameters are not available for this molecule: "
                f"{exc}. Set sasa_model to 'ace' or None to use a different surface "
                "area term."
            ) from exc

        probe_radius = self.settings.probe_radius.value_in_unit(_OMM_NM)
        force = openmm.LCPOForce()
        force.setSurfaceTension(
            self.settings.surface_tension.value_in_unit(_OMM_KJ_PER_MOL / _OMM_NM**2)
        )
        for atom_radius, p1, p2, p3, p4 in lcpo_parameters:
            radius = atom_radius.value_in_unit(_OMM_NM)
            force.addParticle(
                radius + probe_radius if radius else 0.0,
                p1,
                p2,
                p3,
                p4.value_in_unit(_OMM_NM**-2),
            )
        force.setUsesPeriodicBoundaryConditions(False)
        return force


# ---------------------------------------------------------------------------
# AnySolv SMD (reference side)
# ---------------------------------------------------------------------------

_ANYSOLV_INSTALL_HINT = (
    "AnySolv is an optional dependency which is not available from conda-forge. "
    'Install it with `pip install "anysolv[hub]"`, or use a pixi environment with '
    "the 'anysolv' feature."
)
_ANYSOLV_HF_REPO = "antonknee/anysolv"
_ANYSOLV_MOE_LICENCE_HINT = (
    "The 'model_moe' weights are gated: request access and accept the FAIR Chemistry "
    f"License at https://huggingface.co/{_ANYSOLV_HF_REPO}, then log in with "
    "`hf auth login`. Alternatively, set checkpoint to 'model_compact' (bundled with "
    "AnySolv) or to the path of a downloaded .pt file."
)


def _import_optional(module: str, hint: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            f"Could not import {module}. {hint}"
        ) from exc


@functools.cache
def _resolve_anysolv_checkpoint(checkpoint: str) -> str:
    """Return the checkpoint to pass to AnySolv, downloading 'model_moe' if needed.

    AnySolv itself only looks for named checkpoints inside its package directory, and
    silently falls back to 'model_compact' when given no checkpoint, so the 'model_moe'
    weights are fetched from Hugging Face here and passed as an explicit path.
    """
    if Path(checkpoint).is_file() or checkpoint != "model_moe":
        return checkpoint

    anysolv_model = _import_optional("anysolv.model", _ANYSOLV_INSTALL_HINT)
    bundled = Path(getattr(anysolv_model, "_CKPT_DIR", "")) / "model_moe.pt"
    if bundled.is_file():
        return str(bundled)

    hub = _import_optional(
        "huggingface_hub",
        "huggingface_hub is required to download the AnySolv 'model_moe' weights. "
        'Install it with `pip install "anysolv[hub]"`.',
    )
    cached = hub.try_to_load_from_cache(_ANYSOLV_HF_REPO, "model_moe.pt")
    if isinstance(cached, str):
        return cached

    logger.info(f"Downloading the AnySolv 'model_moe' weights from {_ANYSOLV_HF_REPO}")
    try:
        return str(hub.hf_hub_download(_ANYSOLV_HF_REPO, "model_moe.pt"))
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the AnySolv 'model_moe' weights ({exc}). "
            f"{_ANYSOLV_MOE_LICENCE_HINT}"
        ) from exc


class _AnySolvForceCallback:
    """Computes AnySolv energies and forces for an OpenMM `PythonForce`."""

    def __init__(self, correction: AnySolvSMDCorrection) -> None:
        self.correction = correction

    def __call__(self, state: openmm.State) -> tuple[float, np.ndarray]:
        positions = state.getPositions(asNumpy=True).value_in_unit(_OMM_NM)
        energy, forces = self.correction.predict(positions * _NM_TO_ANGSTROM)
        # kcal/mol -> kJ/mol and kcal/mol/Å -> kJ/mol/nm
        return (
            energy * _KCAL_TO_KJ,
            forces * _KCAL_TO_KJ * _NM_TO_ANGSTROM,
        )


@_register_correction(presto_settings.AnySolvSMDCorrectionSettings)
class AnySolvSMDCorrection(EnergyCorrection):
    """SMD implicit-solvent energy and force corrections predicted by AnySolv."""

    side = "reference"
    settings: presto_settings.AnySolvSMDCorrectionSettings

    def __init__(
        self,
        settings: presto_settings.AnySolvSMDCorrectionSettings,
        mol: Molecule,
        *,
        device: torch.device,
        charges: np.ndarray | None = None,
    ) -> None:
        """Create the correction. ``charges`` are not used."""
        super().__init__(settings, mol, device=device, charges=charges)
        self.atomic_numbers = np.array([atom.atomic_number for atom in mol.atoms])
        # OpenFF molecules are closed-shell, so the spin multiplicity is always 1
        self.total_charge = round(mol.total_charge.m_as(off_unit.elementary_charge))

    def predict(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        """Predict the correction for one conformation.

        Args:
            positions: Coordinates [Å] with ``shape=(n_atoms, 3)``.

        Returns:
            Tuple of the energy [kcal/mol] and forces [kcal/mol/Å].
        """
        anysolv = _import_optional("anysolv", _ANYSOLV_INSTALL_HINT)
        delta_e, delta_f = anysolv.predict_solvation_energy(
            (self.atomic_numbers, np.asarray(positions, dtype=float)),
            charge=self.total_charge,
            spin=1,
            solvent=self.settings.solvent,
            checkpoint=_resolve_anysolv_checkpoint(self.settings.checkpoint),
            device=str(self.device),
            dtype=getattr(torch, self.settings.dtype),
            inference_settings=self.settings.inference_settings,
        )
        return (
            float(delta_e) * _EV_TO_KCAL_PER_MOL,
            np.asarray(delta_f, dtype=float) * _EV_TO_KCAL_PER_MOL,
        )

    def compute(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the correction directly, without OpenMM."""
        coords = np.asarray(coords, dtype=float).reshape(len(coords), -1, 3)
        energies, forces = [], []
        for conformer in coords:
            energy, conformer_forces = self.predict(conformer)
            energies.append(energy)
            forces.append(conformer_forces)
        return np.array(energies), np.array(forces).reshape(coords.shape)

    def create_openmm_forces(self) -> list[openmm.Force]:
        """Return a `PythonForce` evaluating AnySolv on ``device``."""
        return [openmm.PythonForce(_AnySolvForceCallback(self))]

    @classmethod
    def prepare(cls, settings: CorrectionSettings) -> None:
        """Check that AnySolv is installed and fetch the checkpoint."""
        assert isinstance(settings, presto_settings.AnySolvSMDCorrectionSettings)
        _import_optional("anysolv", _ANYSOLV_INSTALL_HINT)
        _resolve_anysolv_checkpoint(settings.checkpoint)

    @classmethod
    def release(cls) -> None:
        """Clear AnySolv's model cache and free GPU memory."""
        try:
            predict = importlib.import_module("anysolv.predict")
        except ImportError:
            return
        model_cache = getattr(predict, "_MODEL_CACHE", None)
        if isinstance(model_cache, dict):
            model_cache.clear()
        free_gpu_memory()
