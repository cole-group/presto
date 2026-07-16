"""Create new tagged SMARTS parameter types for molecules of interest."""

from __future__ import annotations

import copy
import warnings
from collections import defaultdict
from collections.abc import Mapping

import openff.toolkit
from loguru import logger
from openff.units import Quantity, unit
from rdkit import Chem

from .settings import TypeGenerationSettings
from .utils.typing import NonLinearValenceType


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

    Returns:
    -------
    str
        The SMARTS pattern with atom maps.
    """
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
        atoms_to_remove = sorted(
            [i for i in range(mol_rdkit.GetNumAtoms()) if i not in atoms_to_include],
            reverse=True,
        )
        for atom_idx in atoms_to_remove:
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
        If None, no parameters are removed. Parameters without an ID are never
        removed, as the ID is optional in the SMIRNOFF spec and such parameters
        cannot have been generated by PRESTO.

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
            # Parameters without an ID were not generated by PRESTO, so are never
            # removal candidates. The ID is optional in the SMIRNOFF spec and is
            # omitted by some force fields, e.g. double-exponential ones.
            if param.id and id_substring in param.id and param.id not in used_ids:
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

    for handler_name, settings in type_generation_settings.items():
        parameter_handler = ff_copy.get_parameter_handler(handler_name)

        # Collect all SMARTS patterns from all molecules
        all_bespoke_smarts: list[str] = []
        smarts_to_param: dict[
            str, openff.toolkit.typing.engines.smirnoff.parameters.ParameterType
        ] = {}

        for mol in mols_for_typing:
            # Find all matches for this handler on the molecule
            matches = parameter_handler.find_matches(mol.to_topology())

            for match_key, match in matches.items():
                param = match.parameter_type
                atom_indices = match_key

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
                    mol, atom_indices, settings.max_extend_distance
                )

                if bespoke_smarts not in smarts_to_param:
                    all_bespoke_smarts.append(bespoke_smarts)
                    smarts_to_param[bespoke_smarts] = param

        logger.info(
            f"Generated {len(all_bespoke_smarts)} bespoke SMARTS patterns for handler {handler_name} across {len(mols_for_typing)} molecules."
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
