"""Create new tagged SMARTS parameter types for molecules of interest."""

from __future__ import annotations

import copy
import warnings
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import openff.toolkit
from loguru import logger
from openff.units import Quantity, unit
from rdkit import Chem

from .settings import TypeGenerationSettings
from .utils.typing import NonLinearValenceType


_SUPPORTED_CUT_BOND_TYPES = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}
_MAX_MASK_MATCHES = 10_000


def _parameter_fingerprint(
    parameter: openff.toolkit.typing.engines.smirnoff.parameters.ParameterType,
) -> tuple[tuple[str, str], ...]:
    """Return a stable, diagnostic fingerprint for a force-field parameter."""
    return tuple(
        sorted((key, str(value)) for key, value in parameter.to_dict().items())
    )


def _molecule_label(mol: openff.toolkit.Molecule, index: int | None = None) -> str:
    """Return a stable human-readable molecule label for diagnostics."""
    prefix = f"molecule {index}" if index is not None else "molecule"
    return f"{prefix} ({mol.name or mol.to_smiles(explicit_hydrogens=False)})"


def _find_atoms_to_remove(
    mol: openff.toolkit.Molecule,
    remove_atom_smarts: list[str],
    handler_name: str,
) -> set[int]:
    """Match and validate mapped terminal atom-removal patterns on ``mol``."""
    if not remove_atom_smarts:
        return set()

    rd_mol = mol.to_rdkit()
    selections: list[tuple[str, frozenset[int]]] = []

    for smarts in remove_atom_smarts:
        query = Chem.MolFromSmarts(smarts)
        assert query is not None  # Statically validated by TypeGenerationSettings.
        mapped_query_indices = [
            atom.GetIdx()
            for atom in query.GetAtoms()  # type: ignore[no-untyped-call]
            if atom.GetAtomMapNum() > 0
        ]
        matches = rd_mol.GetSubstructMatches(
            query,
            uniquify=False,
            useChirality=False,
            maxMatches=_MAX_MASK_MATCHES,
        )
        if len(matches) == _MAX_MASK_MATCHES:
            raise ValueError(
                f"Removal SMARTS {smarts!r} reached the {_MAX_MASK_MATCHES} match "
                f"limit for {mol.name or mol.to_smiles()}; make the pattern more specific."
            )

        distinct = {
            frozenset(match[query_idx] for query_idx in mapped_query_indices)
            for match in matches
        }
        if not distinct:
            raise ValueError(
                f"Removal SMARTS {smarts!r} did not match molecule "
                f"{mol.name or mol.to_smiles()} for handler {handler_name}."
            )

        logger.info(
            f"Removal SMARTS {smarts!r} produced {len(matches)} raw matches and "
            f"{len(distinct)} distinct deletion sets for handler {handler_name} on "
            f"{mol.name or mol.to_smiles()}."
        )

        for selected in distinct:
            expanded = set(selected)
            for atom_idx in selected:
                atom = rd_mol.GetAtomWithIdx(atom_idx)
                expanded.update(
                    neighbor.GetIdx()
                    for neighbor in atom.GetNeighbors()
                    if neighbor.GetAtomicNum() == 1 and neighbor.GetDegree() == 1
                )
            selections.append((smarts, frozenset(expanded)))

    for index, (smarts, selected) in enumerate(selections):
        selected_mol = Chem.PathToSubmol(
            rd_mol,
            [
                bond.GetIdx()
                for bond in rd_mol.GetBonds()
                if bond.GetBeginAtomIdx() in selected
                and bond.GetEndAtomIdx() in selected
            ],
        )
        # A single selected atom has no bonds but is still connected.
        if len(selected) > 1 and selected_mol.GetNumAtoms() != len(selected):
            raise ValueError(
                f"Mapped atoms selected by removal SMARTS {smarts!r} are not connected "
                f"in molecule {mol.name or mol.to_smiles()}."
            )

        boundary_bonds = [
            bond
            for bond in rd_mol.GetBonds()
            if (bond.GetBeginAtomIdx() in selected)
            != (bond.GetEndAtomIdx() in selected)
        ]
        if len(boundary_bonds) != 1:
            raise ValueError(
                f"Removal SMARTS {smarts!r} must select a terminal component with "
                f"exactly one attachment bond, but found {len(boundary_bonds)} on "
                f"{mol.name or mol.to_smiles()}."
            )
        if boundary_bonds[0].GetBondType() not in _SUPPORTED_CUT_BOND_TYPES:
            raise ValueError(
                f"Removal SMARTS {smarts!r} cuts unsupported bond type "
                f"{boundary_bonds[0].GetBondType()} on {mol.name or mol.to_smiles()}."
            )

        for other_smarts, other_selected in selections[:index]:
            if selected & other_selected:
                raise ValueError(
                    f"Removal SMARTS {smarts!r} and {other_smarts!r} select "
                    f"overlapping atoms on {mol.name or mol.to_smiles()}."
                )
            if any(
                (
                    bond.GetBeginAtomIdx() in selected
                    and bond.GetEndAtomIdx() in other_selected
                )
                or (
                    bond.GetEndAtomIdx() in selected
                    and bond.GetBeginAtomIdx() in other_selected
                )
                for bond in rd_mol.GetBonds()
            ):
                raise ValueError(
                    f"Removal SMARTS {smarts!r} and {other_smarts!r} select adjacent "
                    f"components on {mol.name or mol.to_smiles()}."
                )

    return set().union(*(selected for _, selected in selections))


def _add_parameter_with_overwrite(
    handler: openff.toolkit.typing.engines.smirnoff.parameters.ParameterHandler,
    parameter_dict: Mapping[str, str | Quantity],
) -> None:
    """Add a parameter to a handler, overwriting any existing parameter with the same smirks."""
    old_parameter = handler.get_parameter({"smirks": parameter_dict["smirks"]})
    new_parameter = handler._INFOTYPE(**parameter_dict)
    if old_parameter:
        assert len(old_parameter) == 1
        old_parameter = old_parameter[0]
        # Keep the old ID if overwriting
        new_parameter.id = old_parameter.id
        logger.debug(
            f"Overwriting existing parameter with id {new_parameter.id} with smirks {parameter_dict['smirks']}."
        )
        idx = handler._index_of_parameter(old_parameter)
        handler._parameters[idx] = new_parameter
    else:
        handler._parameters.append(new_parameter)


def _create_smarts(
    mol: openff.toolkit.Molecule,
    idxs: tuple[int, ...],
    max_extend_distance: int = -1,
    atoms_to_remove: set[int] | None = None,
) -> str:
    """Create a mapped SMARTS representation of a molecule.

    Crucially, this uses MergeQueryHs to merge non-mapped
    hydrogens into their heavy atom. This dramatically increases
    the speed of SMARTS matching in RDKit for complex SMARTS patterns
    (thanks to Niels Maeder for suggesting this!).

    Parameters
    ----------
    mol: openff.toolkit.Molecule
        The molecule to create SMARTS for.
    idxs: tuple[int, ...]
        Indices of the atoms to map (and from which to extend).
    max_extend_distance: int, default -1
        Maximum number of bonds to extend from the mapped atoms.
        If -1, include the entire molecule.
    atoms_to_remove: set[int] | None, default None
        Source-molecule atom indices to remove from the generated pattern. Each
        represented cut bond is terminated by an untagged wildcard atom.

    Returns:
    -------
    str
        The SMARTS pattern with atom maps.
    """
    if atoms_to_remove:
        return _create_masked_smarts(
            mol,
            idxs,
            max_extend_distance=max_extend_distance,
            atoms_to_remove=atoms_to_remove,
        )

    mol_rdkit = mol.to_rdkit()

    # Determine which atoms to include in the SMARTS
    if max_extend_distance == -1:
        # Include all atoms
        atoms_to_include = set(range(mol_rdkit.GetNumAtoms()))
    else:
        # Include atoms within max_extend_distance bonds from the mapped atoms
        atoms_to_include = set(idxs)
        for _ in range(max_extend_distance):
            new_atoms = set()
            for atom_idx in atoms_to_include:
                atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    new_atoms.add(neighbor.GetIdx())
            atoms_to_include.update(new_atoms)

    # Create a copy of the molecule with only the atoms to include
    if max_extend_distance != -1 and atoms_to_include != set(
        range(mol_rdkit.GetNumAtoms())
    ):
        # Create an editable molecule
        edit_mol = Chem.RWMol(mol_rdkit)

        # Remove atoms not in atoms_to_include (reverse order for indices)
        excluded_atom_indices = sorted(
            [i for i in range(mol_rdkit.GetNumAtoms()) if i not in atoms_to_include],
            reverse=True,
        )
        for atom_idx in excluded_atom_indices:
            edit_mol.RemoveAtom(atom_idx)

        # Create mapping from old to new indices
        old_to_new = {}
        new_idx = 0
        for old_idx in range(mol_rdkit.GetNumAtoms()):
            if old_idx in atoms_to_include:
                old_to_new[old_idx] = new_idx
                new_idx += 1

        mol_rdkit = edit_mol.GetMol()
        idxs = tuple(old_to_new[idx] for idx in idxs)

    # Set atom maps for the key atoms
    for i, idx in enumerate(idxs):
        atom = mol_rdkit.GetAtomWithIdx(idx)
        atom.SetAtomMapNum(i + 1)

    # Merge non-mapped hydrogens into their heavy atoms to
    # speed up SMARTS matching
    h_merged_mol_rdkit = Chem.MergeQueryHs(mol_rdkit, True)
    smarts = Chem.MolToSmarts(h_merged_mol_rdkit)

    return smarts


def _create_masked_smarts(
    mol: openff.toolkit.Molecule,
    idxs: tuple[int, ...],
    max_extend_distance: int,
    atoms_to_remove: set[int],
) -> str:
    """Create SMARTS after pruning selected source atoms and adding wildcard cuts."""
    mol_rdkit = mol.to_rdkit()
    for atom in mol_rdkit.GetAtoms():
        atom.SetAtomMapNum(0)

    if max_extend_distance == -1:
        atoms_to_include = set(range(mol_rdkit.GetNumAtoms()))
    else:
        atoms_to_include = set(idxs)
        for _ in range(max_extend_distance):
            atoms_to_include.update(
                neighbor.GetIdx()
                for atom_idx in tuple(atoms_to_include)
                for neighbor in mol_rdkit.GetAtomWithIdx(atom_idx).GetNeighbors()
            )

    represented_removals = atoms_to_include & atoms_to_remove
    retained_atoms = atoms_to_include - represented_removals
    assert not (set(idxs) & atoms_to_remove)

    cut_bonds: list[tuple[int, Chem.BondType]] = []
    for bond in mol_rdkit.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in represented_removals and end in retained_atoms:
            cut_bonds.append((end, bond.GetBondType()))
        elif end in represented_removals and begin in retained_atoms:
            cut_bonds.append((begin, bond.GetBondType()))

    edit_mol = Chem.RWMol(mol_rdkit)
    for atom_idx in sorted(
        set(range(mol_rdkit.GetNumAtoms())) - retained_atoms,
        reverse=True,
    ):
        edit_mol.RemoveAtom(atom_idx)

    old_to_new = {
        old_idx: new_idx for new_idx, old_idx in enumerate(sorted(retained_atoms))
    }
    for retained_idx, bond_type in cut_bonds:
        wildcard_idx = edit_mol.AddAtom(Chem.AtomFromSmarts("*"))
        edit_mol.AddBond(old_to_new[retained_idx], wildcard_idx, bond_type)

    mol_rdkit = edit_mol.GetMol()
    for map_number, old_idx in enumerate(idxs, start=1):
        mol_rdkit.GetAtomWithIdx(old_to_new[old_idx]).SetAtomMapNum(map_number)

    return Chem.MolToSmarts(Chem.MergeQueryHs(mol_rdkit, True))


def _remove_redundant_smarts(
    mols: openff.toolkit.Molecule | list[openff.toolkit.Molecule],
    force_field: openff.toolkit.ForceField,
    id_substring: str | None = None,
) -> openff.toolkit.ForceField:
    """Remove redundant SMARTS parameters that are not used by any molecule.

    This function labels all molecules with the force field and identifies which
    parameters are actually applied. Parameters that are not used by any molecule
    and have an ID containing the specified substring are removed. This works because
    the a given substructure should always be matched by the last equivalent mapped-SMARTS
    in the force field.

    Parameters
    ----------
    mols : openff.toolkit.Molecule | list[openff.toolkit.Molecule]
        Molecule or list of molecules to check parameter usage against
    force_field : openff.toolkit.ForceField
        Force field to remove redundant parameters from
    id_substring : str | None, default None
        Only remove parameters whose ID contains this substring.
        If None, no parameters are removed.

    Returns:
    -------
    openff.toolkit.ForceField
        Force field with redundant parameters removed
    """
    if id_substring is None:
        return force_field

    # Convert single molecule to list
    if isinstance(mols, openff.toolkit.Molecule):
        mols = [mols]

    # Create a copy to avoid modifying the original
    ff_copy = copy.deepcopy(force_field)

    # Label all molecules and collect used parameter IDs for each handler
    used_param_ids: dict[str, set[str]] = defaultdict(set)

    for mol in mols:
        labels = ff_copy.label_molecules(mol.to_topology())[0]
        for handler_name, param_dict in labels.items():
            for param in param_dict.values():
                used_param_ids[handler_name].add(param.id)

    # If no molecules, we need to check all handlers for bespoke parameters
    if not mols:
        # Get all handler names from the force field
        for handler_name in ff_copy.registered_parameter_handlers:
            used_param_ids[handler_name] = set()

    # Remove unused parameters that contain the id_substring
    for handler_name, used_ids in used_param_ids.items():
        handler = ff_copy.get_parameter_handler(handler_name)
        params_to_remove = []

        for param in handler.parameters:
            # Check if parameter has id_substring and is not used
            if id_substring in param.id and param.id not in used_ids:
                params_to_remove.append(param)

        # Remove the parameters
        for param in params_to_remove:
            handler._parameters.remove(param)
            logger.debug(
                f"Removed unused parameter {param.id} with SMIRKS {param.smirks} from {handler_name}"
            )

    return ff_copy


def _remove_stereochemical_information(
    mol: openff.toolkit.Molecule,
) -> openff.toolkit.Molecule:
    """Return a copy of ``mol`` with atom and bond stereochemistry removed."""
    mol_copy = copy.deepcopy(mol)

    had_atom_stereo = any(atom.stereochemistry is not None for atom in mol_copy.atoms)
    had_bond_stereo = any(
        getattr(bond, "_stereochemistry", None) is not None for bond in mol_copy.bonds
    )

    if had_atom_stereo or had_bond_stereo:
        warnings.warn(
            (
                "Input molecule contains stereochemical information that will be "
                "removed before bespoke type generation. This avoids toolkit "
                "disagreements between OpenEye and RDKit (see "
                "https://github.com/openforcefield/openff-toolkit/issues/146) that "
                "can otherwise cause type generation failures. The resulting types "
                "will match alternative stereoisomers, which should not be an issue "
                "for enantiomers unless torsion phase shifts are being trained. This "
                "may introduce some errors for diastereomers."
            ),
            UserWarning,
            stacklevel=2,
        )

    for atom in mol_copy.atoms:
        atom.stereochemistry = None

    for bond in mol_copy.bonds:
        bond._stereochemistry = None

    return mol_copy


def add_types_to_forcefield(
    mols: openff.toolkit.Molecule | list[openff.toolkit.Molecule],
    force_field: openff.toolkit.ForceField,
    type_generation_settings: dict[NonLinearValenceType, TypeGenerationSettings],
) -> openff.toolkit.ForceField:
    """Add bespoke types to a force field based on multiple molecules and type generation settings.

    Parameters
    ----------
    mols : openff.toolkit.Molecule | list[openff.toolkit.Molecule]
        Molecule or list of molecules to parameterize
    force_field : openff.toolkit.ForceField
        The base force field to add bespoke parameters to
    type_generation_settings : dict[NonLinearValenceType, TypeGenerationSettings]
        Settings for generating tagged SMARTS types for each valence type

    Returns:
    -------
    openff.toolkit.ForceField
        Force field with bespoke parameters added, deduplicated across all molecules
    """
    # Convert single molecule to list
    if isinstance(mols, openff.toolkit.Molecule):
        mols = [mols]

    mols_for_typing = [_remove_stereochemical_information(mol) for mol in mols]

    # Create a copy of the force field to avoid modifying the original
    ff_copy = copy.deepcopy(force_field)

    if not mols_for_typing and any(
        settings.remove_atom_smarts for settings in type_generation_settings.values()
    ):
        raise ValueError("Cannot apply remove_atom_smarts without any molecules.")

    for handler_name, settings in type_generation_settings.items():
        parameter_handler = ff_copy.get_parameter_handler(handler_name)
        existing_smirks = {param.smirks for param in parameter_handler.parameters}

        # Collect all SMARTS patterns from all molecules
        all_bespoke_smarts: list[str] = []
        smarts_to_param: dict[
            str, openff.toolkit.typing.engines.smirnoff.parameters.ParameterType
        ] = {}
        smarts_provenance: dict[str, dict[str, Any]] = {}
        skipped_terms: list[
            tuple[int, tuple[int, ...], tuple[tuple[str, str], ...], str]
        ] = []
        contained_count = 0
        crossing_count = 0

        for mol_index, mol in enumerate(mols_for_typing):
            atoms_to_remove = _find_atoms_to_remove(
                mol, settings.remove_atom_smarts, handler_name
            )
            # Find all matches for this handler on the molecule
            matches = parameter_handler.find_matches(mol.to_topology())

            for match_key, match in matches.items():
                param = match.parameter_type
                atom_indices = match_key

                removed_term_atoms = set(atom_indices) & atoms_to_remove
                if removed_term_atoms:
                    classification = (
                        "cap-contained"
                        if removed_term_atoms == set(atom_indices)
                        else "cap-crossing"
                    )
                    if classification == "cap-contained":
                        contained_count += 1
                    else:
                        crossing_count += 1
                    skipped_terms.append(
                        (
                            mol_index,
                            atom_indices,
                            _parameter_fingerprint(param),
                            classification,
                        )
                    )
                    continue

                # Get the original parameter's SMIRKS
                original_smirks = param.smirks

                # Check if this parameter should be excluded
                if settings.exclude and original_smirks in settings.exclude:
                    continue

                # Check if this parameter should be included (if include list exists)
                if settings.include and original_smirks not in settings.include:
                    continue

                # Create bespoke SMARTS pattern
                bespoke_smarts = _create_smarts(
                    mol,
                    atom_indices,
                    settings.max_extend_distance,
                    atoms_to_remove=atoms_to_remove,
                )

                provenance = {
                    "molecule": _molecule_label(mol, mol_index),
                    "handler": handler_name,
                    "term": atom_indices,
                    "parameter_id": param.id,
                    "parameter_smirks": param.smirks,
                    "fingerprint": _parameter_fingerprint(param),
                }

                if settings.remove_atom_smarts and bespoke_smarts in existing_smirks:
                    raise ValueError(
                        f"Masked SMARTS {bespoke_smarts!r} collides with an existing "
                        f"{handler_name} parameter; source provenance: {provenance}."
                    )

                if bespoke_smarts in smarts_to_param:
                    previous = smarts_provenance[bespoke_smarts]
                    if (
                        settings.remove_atom_smarts
                        and previous["fingerprint"] != provenance["fingerprint"]
                    ):
                        raise ValueError(
                            f"Masked SMARTS {bespoke_smarts!r} was generated from "
                            f"different source parameters: {previous} and {provenance}."
                        )
                else:
                    all_bespoke_smarts.append(bespoke_smarts)
                    smarts_to_param[bespoke_smarts] = param
                    smarts_provenance[bespoke_smarts] = provenance

        logger.info(
            f"Generated {len(all_bespoke_smarts)} bespoke SMARTS patterns for handler {handler_name} across {len(mols_for_typing)} molecules."
        )
        if settings.remove_atom_smarts:
            logger.info(
                f"Skipped {contained_count} cap-contained and {crossing_count} "
                f"cap-crossing {handler_name} terms."
            )

        # Add the SMARTS patterns to the handler
        handler_copy = copy.deepcopy(parameter_handler)

        for bespoke_smarts in all_bespoke_smarts:
            param = smarts_to_param[bespoke_smarts]

            # Create a new parameter dict based on the original parameter
            new_param_dict = {"smirks": bespoke_smarts}

            # Copy over all parameter attributes from the original
            for attr_name in param.to_dict().keys():
                if attr_name not in ["smirks", "id"]:
                    attr_value = getattr(param, attr_name)
                    new_param_dict[attr_name] = attr_value

            # Generate a unique ID for the new parameter
            counter = len(handler_copy.parameters) + 1
            new_param_dict["id"] = f"{handler_name[0].lower()}-bespoke-{counter}"

            # Add the new parameter to the handler
            _add_parameter_with_overwrite(handler_copy, new_param_dict)

        # Update the force field with the modified parameter handler
        ff_copy.deregister_parameter_handler(handler_name)
        ff_copy.register_parameter_handler(handler_copy)

        # A skipped cap term must retain precisely the parameter it had before any
        # bespoke patterns were appended. A broad generated pattern must not silently
        # override this fallback.
        if skipped_terms:
            final_handler = ff_copy.get_parameter_handler(handler_name)
            final_matches_by_mol = [
                final_handler.find_matches(mol.to_topology()) for mol in mols_for_typing
            ]
            for mol_index, term, expected, classification in skipped_terms:
                assigned = final_matches_by_mol[mol_index][term].parameter_type
                if _parameter_fingerprint(assigned) != expected:
                    raise ValueError(
                        f"A bespoke {handler_name} pattern overrode {classification} "
                        f"term {term} on {_molecule_label(mols_for_typing[mol_index], mol_index)}. "
                        f"Expected base parameter {dict(expected)}, but received "
                        f"{assigned.id} ({assigned.smirks})."
                    )

    # Remove redundant parameters that are not used by any molecule
    ff_copy = _remove_redundant_smarts(mols_for_typing, ff_copy, id_substring="bespoke")

    return ff_copy


def add_library_charges_to_forcefield(
    mols: openff.toolkit.Molecule | list[openff.toolkit.Molecule],
    force_field: openff.toolkit.ForceField,
) -> openff.toolkit.ForceField:
    """Write per-atom ``LibraryCharges`` from molecules' partial charges into a force field.

    For each atom of each molecule a bespoke single-tagged-atom SMARTS spanning the
    whole molecule is generated using the same machinery as the valence types (see
    :func:`add_types_to_forcefield`). The charge assigned to each SMARTS is the mean
    of the partial charges of all atoms that produce it (i.e. symmetry-equivalent
    atoms), which both symmetrises equivalent atoms and preserves the total molecular
    charge exactly.

    Because the SMARTS spans the whole molecule, each one only matches its own
    symmetry class, so every atom is covered by exactly one library charge and the net
    charge of the molecule is reproduced. This is required because OpenFF interchange
    does not renormalise charges: it only applies a ``LibraryCharges`` handler if it
    covers every atom of the molecule, and otherwise raises if the assigned charges do
    not sum to the formal charge. The non-tagged hydrogens are merged onto their heavy
    atoms by ``MergeQueryHs`` for fast SMARTS matching.

    Parameters
    ----------
    mols : openff.toolkit.Molecule | list[openff.toolkit.Molecule]
        Molecule or molecules with ``partial_charges`` set.
    force_field : openff.toolkit.ForceField
        The base force field to add the library charges to.

    Returns:
    -------
    openff.toolkit.ForceField
        A copy of the force field with bespoke library charges added, deduplicated
        across all molecules.
    """
    # Convert single molecule to list
    if isinstance(mols, openff.toolkit.Molecule):
        mols = [mols]

    # Validate partial charges before doing any work
    charges_per_mol: list[list[Quantity]] = []
    for mol in mols:
        if mol.partial_charges is None:
            raise ValueError(
                f"Molecule {mol.to_smiles(explicit_hydrogens=False)} is missing "
                "partial charges. Set Molecule.partial_charges before generating "
                "library charges."
            )

        charges = list(mol.partial_charges)
        charge_sum = sum(c.m_as(unit.elementary_charge) for c in charges)
        formal_sum = mol.total_charge.m_as(unit.elementary_charge)
        if abs(charge_sum - formal_sum) > 0.01:
            raise ValueError(
                f"Partial charges of molecule {mol.to_smiles(explicit_hydrogens=False)} "
                f"sum to {charge_sum:.4f} e, which differs from its formal charge of "
                f"{formal_sum:.4f} e by more than 0.01 e. Library charges would not "
                "produce an integral net charge. Please ensure that the partial charges "
                "are consistent with the formal charge of the molecule."
            )
        charges_per_mol.append(charges)

    # Strip stereochemistry so generated SMARTS match either stereoisomer (atom index
    # order is preserved by the copy, so the captured charges stay aligned).
    mols_for_typing = [_remove_stereochemical_information(mol) for mol in mols]

    ff_copy = copy.deepcopy(force_field)
    parameter_handler = ff_copy.get_parameter_handler("LibraryCharges")

    # Collect, for each unique whole-molecule SMARTS, the charges of every atom that
    # produces it (across all molecules) so they can be averaged. The dict preserves
    # insertion order, giving deterministic parameter ordering.
    charges_by_smarts: dict[str, list[float]] = defaultdict(list)

    for mol, charges in zip(mols_for_typing, charges_per_mol, strict=True):
        for atom_index in range(mol.n_atoms):
            smarts = _create_smarts(mol, (atom_index,), max_extend_distance=-1)
            charges_by_smarts[smarts].append(
                float(charges[atom_index].m_as(unit.elementary_charge))
            )

    logger.info(
        f"Generated {len(charges_by_smarts)} bespoke library charge SMARTS patterns "
        f"across {len(mols_for_typing)} molecules."
    )

    handler_copy = copy.deepcopy(parameter_handler)

    for smarts, atom_charges in charges_by_smarts.items():
        mean_charge = sum(atom_charges) / len(atom_charges)

        counter = len(handler_copy.parameters) + 1
        new_param_dict = {
            "smirks": smarts,
            "charge1": mean_charge * unit.elementary_charge,
            "id": f"l-bespoke-{counter}",  # l for library charge
        }

        _add_parameter_with_overwrite(handler_copy, new_param_dict)

    ff_copy.deregister_parameter_handler("LibraryCharges")
    ff_copy.register_parameter_handler(handler_copy)

    # Remove any redundant parameters that are not used by any molecule
    ff_copy = _remove_redundant_smarts(mols_for_typing, ff_copy, id_substring="bespoke")

    return ff_copy
