"""
Bespoke RDKit toolkit wrapper optimized for SMIRNOFF force field assignment.
"""

import inspect
import threading
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Generator, ParamSpec, TypeVar

import loguru
import numpy as np
from openff.toolkit.utils.exceptions import (
    ChargeMethodUnavailableError,
)
from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
from openff.toolkit.utils.toolkits import ToolkitRegistry

if TYPE_CHECKING:
    from openff.toolkit.topology.molecule import Molecule

logger = loguru.logger


_registry_lock = threading.Lock()

if TYPE_CHECKING:
    from rdkit import Chem


@contextmanager
def use_bespoke_rdkit_toolkit() -> Generator[ToolkitRegistry, None, None]:
    """
    Context manager that temporarily registers the RDKitBespokeToolkitWrapper (
    which ensures fast SMARTS matching for highly symmetric molecules) with the
    OpenFF toolkit registry and restores the original registry on exit.

    Yields
    ------
    GLOBAL_TOOLKIT_REGISTRY
        The OpenFF GLOBAL_TOOLKIT_REGISTRY while the context is active.
    """
    try:
        from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
    except Exception as e:
        raise RuntimeError(
            "Cannot use RDKitBespokeToolkitWrapper context manager: "
            "openff toolkit GLOBAL_TOOLKIT_REGISTRY not available."
        ) from e

    with _registry_lock:
        # Snapshot the current registry list contents (shallow copy).
        original_toolkits = list(GLOBAL_TOOLKIT_REGISTRY._toolkits)

        # Find any existing bespoke instances.
        existing_indices = [
            i
            for i, t in enumerate(GLOBAL_TOOLKIT_REGISTRY._toolkits)
            if isinstance(t, RDKitBespokeToolkitWrapper)
        ]

        if not existing_indices:
            # No bespoke toolkit present: create and insert/append as requested.
            bespoke = RDKitBespokeToolkitWrapper()
            GLOBAL_TOOLKIT_REGISTRY._toolkits.insert(0, bespoke)
        else:
            # Bespoke already present. Move the first one to front.
            idx = existing_indices[0]
            if idx != 0:
                item = GLOBAL_TOOLKIT_REGISTRY._toolkits.pop(idx)
                GLOBAL_TOOLKIT_REGISTRY._toolkits.insert(0, item)

    try:
        yield GLOBAL_TOOLKIT_REGISTRY

    finally:
        # Restore the registry contents in-place so any references to the list remain valid.
        with _registry_lock:
            GLOBAL_TOOLKIT_REGISTRY._toolkits[:] = original_toolkits


P = ParamSpec("P")
R = TypeVar("R")


def use_bespoke_rdkit_toolkit_decorator(_func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that runs the wrapped synchronous function inside the
    `use_bespoke_rdkit_toolkit()` context manager.

    Usage:
      - @use_bespoke_rdkit_toolkit_decorator

    Note: async functions are not supported and will raise TypeError.
    """

    def _decorate(func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):
            raise TypeError(
                "use_bespoke_rdkit_toolkit_decorator does not support async functions."
            )

        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with use_bespoke_rdkit_toolkit():
                return func(*args, **kwargs)

        return _wrapper

    return _decorate(_func)


class RDKitBespokeToolkitWrapper(RDKitToolkitWrapper):  # type: ignore[misc]
    """
    RDKit toolkit wrapper optimized for SMIRNOFF force field parameter assignment.

    This wrapper uses symmetry perception to dramatically speed up SMARTS matching
    for highly symmetric molecules by deduplicating chemically equivalent matches.

    Key differences from RDKitToolkitWrapper:
    - SMARTS matching uses symmetry perception to avoid exponential match explosion
    - Partial charge assignment methods are disabled (use specialized toolkits instead)

    Examples
    --------
    For a highly symmetric molecule with 24,000 SMARTS matches, this toolkit will
    return only the ~10-50 unique chemical environments instead.

    >>> toolkit = RDKitBespokeToolkitWrapper()
    >>> matches = toolkit.find_smarts_matches(molecule, '[C:1]-[C:2]')

    Notes
    -----
    This is specifically designed for force field parameter assignment workflows where
    you only need to know which unique chemical environments exist, not enumerate all
    symmetry-equivalent permutations.
    """

    _toolkit_name = "The RDKit (Bespoke)"

    # Remove charge assignment methods - use specialized toolkits instead
    _supported_charge_methods: dict[str, dict[Any, Any]] = {}

    SUPPORTED_CHARGE_METHODS: set[str] = set(_supported_charge_methods.keys())

    def __init__(self) -> None:
        super().__init__()
        logger.info(
            "Using RDKitBespokeToolkitWrapper with symmetry-aware SMARTS matching. "
            "This will significantly speed up matching for molecules where each type "
            " matches only one set of symmetry-equivalent atoms."
        )

    @staticmethod
    def _find_smarts_matches(
        rdmol: "Chem.Mol",
        smarts: str,
        aromaticity_model: str = "OEAroModel_MDL",
        unique: bool = False,
    ) -> list[tuple[int, ...]]:
        """
        Find all sets of atoms in the provided RDKit molecule that match the provided
        SMARTS string, using symmetry perception to avoid exponential match explosion.
        If the SMARTS pattern contains all the atoms in the molecule, this method is
        used as there can only be one set of symmetry-equivalent matches; otherwise,
        the original RDKit behavior is used.

        This method uses a fast path optimized for force field assignment:
        1. Gets symmetry classes for all atoms once
        2. Finds matches with uniquify=True (fast internal deduplication)
        3. Deduplicates based on symmetry signatures (chemical equivalence)

        For highly symmetric molecules, this reduces 24,000+ matches to ~10-50 unique
        chemical environments.

        Parameters
        ----------
        rdmol
            RDKit molecule to search for matches
        smarts
            SMARTS string with sequentially tagged atoms (1..N)
        aromaticity_model
            OpenEye aromaticity model designation (only "OEAroModel_MDL" supported)
        unique
            Passed to RDKit's GetSubstructMatches as uniquify parameter

        Returns
        -------
        matches
            List of N-tuples of atom indices, one per unique chemical environment

        Raises
        ------
        ChemicalEnvironmentParsingError
            If the SMARTS string is malformed

        Notes
        -----
        - For force field assignment, you typically want use_symmetry=True (default)
        - Matches are not guaranteed to be in any particular order
        - Each match represents a unique chemical environment based on atom symmetry
        """
        from openff.toolkit.utils.exceptions import ChemicalEnvironmentParsingError
        from rdkit import Chem

        # Parse the SMARTS pattern
        qmol = Chem.MolFromSmarts(smarts)
        if qmol is None:
            raise ChemicalEnvironmentParsingError(
                f'RDKit could not parse the SMARTS/SMIRKS string "{smarts}"'
            )

        # Create atom mapping for query molecule
        idx_map: dict[int, int] = {}
        for atom in qmol.GetAtoms():  # type: ignore[no-untyped-call]
            smirks_index = atom.GetAtomMapNum()
            if smirks_index != 0:
                idx_map[smirks_index - 1] = atom.GetIdx()
        map_list = [idx_map[x] for x in sorted(idx_map)]

        # If the SMARTS pattern contains all atoms in the molecule, use symmetry-based matching
        if qmol.GetNumAtoms() == rdmol.GetNumAtoms():
            # Compute symmetry classes once - this is fast
            symmetry_ids = list(Chem.CanonicalRankAtoms(rdmol, breakTies=False))

            # Get only a single match and get the idxs of the atoms involved in the potential
            first_full_match = rdmol.GetSubstructMatches(
                qmol, uniquify=False, maxMatches=1, useChirality=True
            )[0]
            first_match = [first_full_match[x] for x in map_list]
            first_symmetry_ids = [symmetry_ids[x] for x in first_match]
            # Get a list of the number of bonds separating the matched atoms

            def get_shortest_bond_separation(
                rdmol: Chem.Mol, atom1: int, atom2: int
            ) -> int:
                return len(Chem.GetShortestPath(rdmol, atom1, atom2)) - 1

            bond_separations = {}  # Index is the index of the reference atom
            for i in range(1, len(first_match)):
                other_atoms = first_match[:i]
                bond_separations[i] = [
                    get_shortest_bond_separation(rdmol, first_match[i], atom)
                    for atom in other_atoms
                ]

            # Now, for each idx in symmetry_ids matching first_symmetry_ids, find the
            # next idx matching the next symmetry id which is the correct no of bonds away.
            # Do this recursively so it works for any length of match
            matched_idxs = []

            def find_symmetric_matches(
                current_idx_in_first_symmetry_ids: int,
                current_match: list[int],
            ) -> None:
                # Base case: if we've matched all atoms, we're done
                if current_idx_in_first_symmetry_ids == len(first_symmetry_ids) - 1:
                    matched_idxs.append(tuple(current_match))
                    return

                # Find the next symmetry id to match
                next_sym_id = first_symmetry_ids[current_idx_in_first_symmetry_ids + 1]
                for i in range(len(symmetry_ids)):
                    if not symmetry_ids[i] == next_sym_id:
                        continue

                    if i in current_match:
                        continue

                    current_bond_seps = [
                        get_shortest_bond_separation(rdmol, i, j) for j in current_match
                    ]
                    if (
                        not current_bond_seps
                        == bond_separations[current_idx_in_first_symmetry_ids + 1]
                    ):
                        continue

                    find_symmetric_matches(
                        current_idx_in_first_symmetry_ids + 1, current_match + [i]
                    )

            first_idx_molecule_matches = [
                i
                for i in range(len(symmetry_ids))
                if symmetry_ids[i] == first_symmetry_ids[0]
            ]

            for molecule_idx in first_idx_molecule_matches:
                find_symmetric_matches(0, [molecule_idx])

            return matched_idxs

        # Otherwise, do not assume only one set of symmetry-equivalent matches
        # and use the original RDKit behavior (very slow for symmetric molecules
        # with large SMIRKS patterns)
        else:
            max_matches = np.iinfo(np.uintc).max
            full_matches = rdmol.GetSubstructMatches(
                qmol, uniquify=unique, maxMatches=max_matches, useChirality=True
            )
            matches = [tuple(match[x] for x in map_list) for match in full_matches]
            return matches

    def assign_partial_charges(
        self,
        molecule: "Molecule",
        partial_charge_method: str | None = None,
        use_conformers: bool | None = None,
        strict_n_conformers: bool = False,
        normalize_partial_charges: bool = True,
        _cls: type | None = None,
    ) -> None:
        """
        Raises an error - use specialized charge assignment toolkits instead.

        The RDKitBespokeToolkitWrapper is optimized for SMARTS matching in force field
        parameter assignment, not charge calculation. For partial charges, use:
        - AmberToolsToolkitWrapper for AM1-BCC charges
        - OpenEyeToolkitWrapper for high-quality AM1-BCC charges
        - NAGLToolkitWrapper for graph neural network charges

        Raises
        ------
        ChargeMethodUnavailableError
            Always raised to redirect users to appropriate charge methods
        """
        raise ChargeMethodUnavailableError(
            "RDKitBespokeToolkitWrapper does not support partial charge assignment. "
            "This toolkit is optimized for fast SMARTS matching in force field parameter "
            "assignment. For partial charges, please use:\n"
            "  - AmberToolsToolkitWrapper for AM1-BCC charges\n"
            "  - OpenEyeToolkitWrapper for high-quality AM1-BCC charges\n"
            "  - NAGLToolkitWrapper for graph neural network charges"
        )

    def __repr__(self) -> str:
        return (
            f"RDKitBespokeToolkitWrapper around {self.toolkit_name} "
            f"version {self.toolkit_version} (symmetry-optimized)"
        )
