"""Create new tagged SMARTS parameter types for molecules of interest."""

import copy
from collections import defaultdict
from collections.abc import Mapping
from typing import TypeVar

import openff.toolkit
from loguru import logger
from openff.units import Quantity
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

    Returns
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
        If None, no parameters are removed.

    Returns
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


_T = TypeVar(
    "_T",
    bound=openff.toolkit.typing.engines.smirnoff.parameters.ParameterHandler,
)


def _add_types_to_parameter_handler(
    mol: openff.toolkit.Molecule,
    parameter_handler: _T,
    handler_name: str,
    max_extend_distance: int = -1,
    excluded_smirks: list[str] | None = None,
    included_smirks: list[str] | None = None,
) -> _T:
    """
    Add bespoke parameters to the parameter handler based on the molecule.

    This:

    a) Labels the molecule with the original parameter types from the
       parameter handler (using `find_matches`).
    b) For each set of atoms labelled using the original parameters:
        i) Check that it is not excluded (if `excluded_smirks` is
           provided) or is included (if `included_smirks` is provided).
        ii) Create a SMARTS pattern that extends from the labelled atoms up
            to `max_extend_distance` bonds.
        iii) Add a new parameter to the parameter handler with the created
             SMARTS pattern. If a parameter with the same SMARTS pattern
             already exists, log a warning and skip adding the new parameter.

    Parameters
    ----------
    mol: openff.toolkit.Molecule
        The molecule to parameterise.
    parameter_handler: (
        openff.toolkit.typing.engines.smirnoff.parameters.ParameterHandler
    )
        The parameter handler to add bespoke parameters to.
    handler_name: str
        The name of the parameter handler.
    max_extend_distance: int = -1
        The maximum distance (in bonds) to extend the SMARTS patterns from
        the atoms which determine the energy for the parameter. If -1, the
        SMARTS patterns will include the entire molecule.
    excluded_smirks: list[str] | None, default None
        A list of SMIRKS patterns to exclude from bespoke type generation.
        This is mutually exclusive with `included_smirks`.
    included_smirks: list[str] | None, default None
        A list of SMIRKS patterns to include in bespoke type generation.
        This is mutually exclusive with `excluded_smirks`.

    Returns
    -------
    openff.toolkit.typing.engines.smirnoff.parameters.ParameterHandler
        The parameter handler with bespoke parameters added.
    """

    # Validate that excluded_smirks and included_smirks are mutually exclusive
    if excluded_smirks and included_smirks:
        raise ValueError(
            "excluded_smirks and included_smirks are mutually exclusive. "
            "Please provide only one."
        )

    excluded_smirks = excluded_smirks or []
    included_smirks = included_smirks or []

    # Create a copy of the parameter handler to avoid modifying the original
    handler_copy = copy.deepcopy(parameter_handler)

    # Find all matches for this handler on the molecule
    matches = handler_copy.find_matches(mol.to_topology())

    # First pass: collect all bespoke SMARTS patterns and their corresponding parameters
    bespoke_smarts_list: list[str] = []
    smarts_to_param: dict[
        str, openff.toolkit.typing.engines.smirnoff.parameters.ParameterType
    ] = {}

    for match_key, match in matches.items():
        # match_key is a tuple of atom indices
        atom_indices = match_key

        # Get the parameter from the match object
        param = match.parameter_type

        # Get the original parameter's SMIRKS
        original_smirks = param.smirks

        # Check if this parameter should be excluded
        if excluded_smirks and original_smirks in excluded_smirks:
            continue

        # Check if this parameter should be included (if include list exists)
        if included_smirks and original_smirks not in included_smirks:
            continue

        # Create a bespoke SMARTS pattern for this specific set of atoms
        bespoke_smarts = _create_smarts(mol, atom_indices, max_extend_distance)

        # Store the SMARTS and associated parameter
        if bespoke_smarts not in smarts_to_param:
            bespoke_smarts_list.append(bespoke_smarts)
            smarts_to_param[bespoke_smarts] = param

    logger.info(
        f"Generated {len(bespoke_smarts_list)} bespoke SMARTS patterns for handler {handler_name}."
    )

    # Second pass: add the SMARTS patterns to the handler
    for bespoke_smarts in bespoke_smarts_list:
        # Get the original parameter to copy attributes from
        param = smarts_to_param[bespoke_smarts]

        # Create a new parameter dict based on the original parameter
        new_param_dict = {"smirks": bespoke_smarts}

        # Copy over all the parameter attributes from the original
        for attr_name in param.to_dict().keys():
            if attr_name not in ["smirks", "id"]:
                attr_value = getattr(param, attr_name)
                new_param_dict[attr_name] = attr_value

        # Generate a unique ID for the new parameter
        counter = len(handler_copy.parameters) + 1
        new_param_dict["id"] = f"{handler_name[0].lower()}-bespoke-{counter}"

        # Add the new parameter to the handler
        _add_parameter_with_overwrite(handler_copy, new_param_dict)

    return handler_copy


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

    Returns
    -------
    openff.toolkit.ForceField
        Force field with bespoke parameters added, deduplicated across all molecules
    """
    # Convert single molecule to list for backward compatibility
    if isinstance(mols, openff.toolkit.Molecule):
        mols = [mols]

    # Create a copy of the force field to avoid modifying the original
    ff_copy = copy.deepcopy(force_field)

    for handler_name, settings in type_generation_settings.items():
        parameter_handler = ff_copy.get_parameter_handler(handler_name)

        # Collect all SMARTS patterns from all molecules
        all_bespoke_smarts: list[str] = []
        smarts_to_param: dict[
            str, openff.toolkit.typing.engines.smirnoff.parameters.ParameterType
        ] = {}

        for mol in mols:
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
            f"Generated {len(all_bespoke_smarts)} bespoke SMARTS patterns for handler {handler_name} across {len(mols)} molecules."
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
    ff_copy = _remove_redundant_smarts(mols, ff_copy, id_substring="bespoke")

    return ff_copy
