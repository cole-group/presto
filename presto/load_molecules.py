"""Helpers for loading molecules from parameterisation inputs."""

import copy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from openff.toolkit import Molecule
from openff.units import Quantity
from openff.units import unit as off_unit
from rdkit import Chem

from .utils.typing import PathLike

MoleculeInputType = Literal["smiles", "sdf"]

MoleculeLoader = Callable[[str], list[Molecule]]

PROBLEMATIC_FUNCTIONAL_GROUP_WARNINGS: dict[str, str] = {
    "[#15]": (
        "Phosphorus-containing molecules may undergo connectivity changes or proton "
        "hopping during MLP minimisations. Set "
        "`training_sampling_settings.sampling_protocol` to `mm_md_metadynamics` to "
        "avoid minimisation-enabled training sampling, and disable MSM by setting "
        "`param_settings.msm_settings` to `null` (YAML) or `None` (Python), because "
        "MSM also performs MLP minimisation."
    ),
    "[SX4](=[OX1])(=[OX1])[NX3]": (
        "Sulfonamides with this environment may have problematic S-N bond "
        "initialisation during MSM. Disable MSM by setting "
        "`param_settings.msm_settings` to `null` (YAML) or `None` (Python)."
    ),
    "[SX4](=[OX1])(=[OX1])[NX2]": (
        "Sulfonamides with this environment may have problematic S-N bond "
        "initialisation during MSM. Disable MSM by setting "
        "`param_settings.msm_settings` to `null` (YAML) or `None` (Python)."
    ),
}


def _molecule_identity(molecule: Chem.Mol) -> str:
    return Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True)


def load_smiles_molecules(input_value: str) -> list[Molecule]:
    """Load a single OpenFF Molecule from a SMILES string."""
    try:
        molecule = Molecule.from_smiles(input_value, allow_undefined_stereo=True)
    except Exception as exc:
        raise ValueError(f"Invalid SMILES string: {input_value}") from exc

    return [molecule]


def load_sdf_molecules(input_value: str) -> list[Molecule]:
    """Load one or more unique OpenFF Molecules from an SDF file."""
    path = Path(input_value)

    if not path.exists():
        raise ValueError(f"SDF file does not exist: {path}")

    if path.suffix.lower() != ".sdf":
        raise ValueError(f"Expected an SDF file path ending in .sdf: {path}")

    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    except Exception as exc:
        raise ValueError(f"Failed to read SDF file: {path}") from exc

    rdkit_molecules = [molecule for molecule in supplier if molecule is not None]
    if not rdkit_molecules:
        raise ValueError(f"No molecules found in SDF file: {path}")

    seen_identities: set[str] = set()
    molecules: list[Molecule] = []

    for rdkit_molecule in rdkit_molecules:
        identity = _molecule_identity(rdkit_molecule)
        if identity in seen_identities:
            raise ValueError(
                f"SDF file contains duplicate molecule entries: {path} ({identity})"
            )
        seen_identities.add(identity)

        try:
            molecule = Molecule.from_rdkit(rdkit_molecule, allow_undefined_stereo=True)
        except Exception as exc:
            raise ValueError(
                f"Failed to convert SDF to OpenFF Molecule: {path}"
            ) from exc

        molecules.append(molecule)

    return molecules


MOLECULE_LOADERS: dict[MoleculeInputType, MoleculeLoader] = {
    "smiles": load_smiles_molecules,
    "sdf": load_sdf_molecules,
}


def _open_sdf_supplier(path: Path) -> Chem.SDMolSupplier:
    """Open an SDF file, raising a clear ``ValueError`` for path-level problems.

    Kept separate from :func:`load_conformers_for_molecule` so callers which read the
    same file for many molecules can report a missing or unreadable file once, rather
    than once per molecule.

    Raises:
    ------
    ValueError
        If the path does not exist, does not end in ``.sdf``, or cannot be read.
    """
    if not path.exists():
        raise ValueError(f"SDF file does not exist: {path}")

    if path.suffix.lower() != ".sdf":
        raise ValueError(f"Expected an SDF file path ending in .sdf: {path}")

    try:
        return Chem.SDMolSupplier(str(path), removeHs=False)
    except Exception as exc:
        raise ValueError(f"Failed to read SDF file: {path}") from exc


def load_conformers_for_molecule(
    molecule: Molecule, sdf_path: PathLike
) -> list[Quantity]:
    """Load the conformers of ``molecule`` from an SDF, aligned to its atom ordering.

    Every record in the SDF that is graph-isomorphic to ``molecule`` is treated as a
    conformer of it. Because the atom ordering in the SDF need not match ``molecule``,
    each matching record is remapped onto ``molecule``'s atom ordering before its
    coordinates are extracted, guaranteeing the returned conformers are valid starting
    positions for a topology built from ``molecule``. Records that do not match are
    ignored (they may belong to another molecule in a multi-molecule SDF).

    Parameters
    ----------
    molecule : openff.toolkit.Molecule
        The molecule whose conformers should be loaded. Defines the canonical atom
        ordering the returned conformers are aligned to.
    sdf_path : PathLike
        Path to an SDF file containing one or more conformers of ``molecule`` (and,
        optionally, of other molecules).

    Returns:
    -------
    list[openff.units.Quantity]
        The matching conformers, each aligned to ``molecule``'s atom ordering.

    Raises:
    ------
    ValueError
        If the path does not exist, does not end in ``.sdf``, or contains no record
        matching ``molecule``.
    """
    path = Path(sdf_path)
    supplier = _open_sdf_supplier(path)

    conformers: list[Quantity] = []

    for rdkit_molecule in supplier:
        if rdkit_molecule is None:
            continue

        try:
            record = Molecule.from_rdkit(rdkit_molecule, allow_undefined_stereo=True)
        except Exception:
            # Records that cannot be interpreted as molecules cannot be a conformer of
            # ``molecule``, so skip them rather than failing the whole load.
            continue

        matched, atom_map = Molecule.are_isomorphic(
            record, molecule, return_atom_map=True
        )
        if not matched or atom_map is None:
            continue

        # ``atom_map`` maps record atom indices -> ``molecule`` atom indices, so remapping
        # the record with ``current_to_new`` yields a molecule in ``molecule``'s ordering.
        aligned = record.remap(atom_map, current_to_new=True)
        conformers.extend(aligned.conformers)

    if not conformers:
        raise ValueError(
            f"SDF file {path} contains no conformers matching the molecule "
            f"{_molecule_identity(molecule.to_rdkit())}."
        )

    return conformers


def molecule_description(molecule: Molecule, index: int) -> str:
    """Describe a molecule for user-facing error messages."""
    # Implicit hydrogens keep this close to what the user wrote in their input.
    smiles = molecule.to_smiles(explicit_hydrogens=False)
    name = molecule.name
    if name:
        return f"molecule {index} ({name}, {smiles})"
    return f"molecule {index} ({smiles})"


_TOOLKIT_REGISTRY_HEADER = "No registered toolkits can provide the capability"
_TOOLKIT_LIST_HEADER = "Available toolkits are:"
_MAX_SUMMARY_LINES = 8


def _summarise_error(exc: Exception, max_lines: int = _MAX_SUMMARY_LINES) -> str:
    """Condense an exception message to the part a user needs.

    ``ToolkitRegistry`` re-raises the underlying failure with the full call signature and
    a list of every registered toolkit prepended, which buries the actual cause, so for
    those only the per-toolkit error lines are kept (all of them: two toolkits can fail
    for different reasons). Every other message is kept in full, because exceptions such
    as ``UnassignedProperTorsionParameterException`` explain themselves on the first line
    and list the offending terms afterwards. Long messages are truncated so a report
    covering many molecules stays readable.
    """
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return type(exc).__name__

    if lines[0].startswith(_TOOLKIT_REGISTRY_HEADER):
        toolkit_list = next(
            (
                position
                for position, line in enumerate(lines)
                if line.startswith(_TOOLKIT_LIST_HEADER)
            ),
            None,
        )
        if toolkit_list is not None and lines[toolkit_list + 1 :]:
            lines = lines[toolkit_list + 1 :]

    if len(lines) > max_lines:
        omitted = len(lines) - max_lines
        lines = [*lines[:max_lines], f"... ({omitted} more lines omitted)"]

    return "\n".join(lines)


def find_conformer_generation_failures(
    molecules: Sequence[Molecule],
) -> dict[int, tuple[str, str]]:
    """Find molecules for which ETKDG cannot generate a conformer.

    Workflow stages that do not use supplied starting conformers need at least one
    ETKDG conformer. Charge assignment is deliberately outside this helper: modern
    graph-based models do not need coordinates, while methods such as AM1BCC do.

    A single conformer is a sufficient probe. ``RDKitToolkitWrapper.generate_conformers``
    embeds with ``randomSeed=1`` (so the result is deterministic) and only raises when
    embedding yields *zero* conformers, so success here guarantees the real call will
    not raise however many conformers it asks for.

    Parameters
    ----------
    molecules : Sequence[openff.toolkit.Molecule]
        The molecules to check. They are not modified.

    Returns:
    -------
    dict[int, tuple[str, str]]
        Mapping of index in ``molecules`` to ``(description, error)`` for each molecule
        whose conformer generation failed. Empty if every molecule can be embedded.
    """
    failures: dict[int, tuple[str, str]] = {}

    for index, molecule in enumerate(molecules):
        # Work on a copy so the caller's molecules do not gain an incidental conformer.
        candidate = copy.deepcopy(molecule)

        try:
            candidate.generate_conformers(
                n_conformers=1, rms_cutoff=0.0 * off_unit.angstrom
            )
        except Exception as exc:
            # The RDKit wrapper raises ConformerGenerationError, but the toolkit
            # registry re-wraps it as a plain ValueError, so neither type alone is
            # enough to catch.
            failures[index] = (
                molecule_description(molecule, index),
                _summarise_error(exc),
            )
            continue

        if not candidate.conformers:
            failures[index] = (
                molecule_description(molecule, index),
                "conformer generation returned no conformers",
            )

    return failures


def find_problematic_functional_groups(
    molecules: Sequence[Molecule],
) -> dict[str, list[str]]:
    """Collect molecule descriptions for each known problematic SMARTS."""
    matches: dict[str, list[str]] = {}
    for smarts in PROBLEMATIC_FUNCTIONAL_GROUP_WARNINGS:
        descriptions = [
            molecule_description(molecule, index)
            for index, molecule in enumerate(molecules)
            if molecule.chemical_environment_matches(smarts)
        ]
        if descriptions:
            matches[smarts] = descriptions
    return matches
