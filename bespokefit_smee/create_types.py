"""Create new tagged SMARTS parameter types for molecules of interest."""

import copy
from collections.abc import Mapping
from typing import Any, Iterator, TypeVar

import openff.toolkit
from loguru import logger
from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
from openff.units import Quantity

from .settings import TypeGenerationSettings
from .utils.typing import NonLinearValenceType


class ReversibleTuple:
    def __init__(self, *items: Any) -> None:
        self.items = tuple(items)
        # Normalize to the lexicographically smaller of the tuple and its reverse
        self.canonical = min(self.items, tuple(reversed(self.items)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReversibleTuple):
            return NotImplemented
        return self.canonical == other.canonical

    def __hash__(self) -> int:
        return hash(self.canonical)

    def __repr__(self) -> str:
        return f"ReversibleTuple{self.items}"

    def __iter__(self) -> Iterator[Any]:
        return iter(self.items)

    def __getitem__(self, index: int) -> Any:
        return self.items[index]


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
        # Ensure we don't change the name
        new_parameter.id = old_parameter.id
        logger.info(
            f"Overwriting existing parameter with smirks {parameter_dict['smirks']}."
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
    from rdkit import Chem

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

    smarts = Chem.MolToSmarts(mol_rdkit)
    return smarts


def _deduplicate_symmetry_related_smarts(
    smarts_list: list[str],
) -> list[str]:
    """
    Deduplicate SMARTS patterns that are symmetry-equivalent.

    Parameters
    ----------
    smarts_list: list[str]
        List of SMARTS patterns to deduplicate.

    Returns
    -------
    list[str]
        Deduplicated list of SMARTS patterns.
    """
    from rdkit import Chem

    unique_smarts: list[str] = []
    unique_mols: list[Chem.Mol] = []

    for smarts in smarts_list:
        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            raise ValueError(f"Invalid SMARTS pattern: {smarts}")

        is_duplicate = False
        for unique_mol, unique_smarts_pattern in zip(
            unique_mols, unique_smarts, strict=True
        ):
            # Cannot match if different number of atoms
            if not mol.GetNumAtoms() == unique_mol.GetNumAtoms():
                continue

            # Check that both SMARTS match the same sets of atoms in the same molecule
            matches_new_smarts = [
                ReversibleTuple(*match)
                for match in RDKitToolkitWrapper._find_smarts_matches(mol, smarts)
            ]
            matches_unique_smarts = [
                ReversibleTuple(*match)
                for match in RDKitToolkitWrapper._find_smarts_matches(
                    mol, unique_smarts_pattern
                )
            ]
            if set(matches_new_smarts) == set(matches_unique_smarts):
                is_duplicate = True
                break

        if not is_duplicate:
            unique_smarts.append(smarts)
            unique_mols.append(mol)

    return unique_smarts


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

    # Deduplicate symmetry-related SMARTS patterns
    unique_smarts = _deduplicate_symmetry_related_smarts(bespoke_smarts_list)
    logger.info(
        f"Generated {len(unique_smarts)} unique bespoke SMARTS patterns for handler {handler_name}."
    )

    # Second pass: add the unique SMARTS patterns to the handler
    for bespoke_smarts in unique_smarts:
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
    mol: openff.toolkit.Molecule,
    force_field: openff.toolkit.ForceField,
    type_generation_settings: dict[NonLinearValenceType, TypeGenerationSettings],
) -> openff.toolkit.ForceField:
    """Add bespoke types to a force field based on the given molecule and type generation settings."""
    # Create a copy of the force field to avoid modifying the original
    ff_copy = copy.deepcopy(force_field)

    for handler_name, settings in type_generation_settings.items():
        parameter_handler = ff_copy.get_parameter_handler(handler_name)

        updated_handler = _add_types_to_parameter_handler(
            mol,
            parameter_handler,
            handler_name=handler_name,
            max_extend_distance=settings.max_extend_distance,
            excluded_smirks=settings.exclude,
            included_smirks=settings.include,
        )

        # Update the force field with the modified parameter handler
        ff_copy.deregister_parameter_handler(handler_name)
        ff_copy.register_parameter_handler(updated_handler)

    return ff_copy
