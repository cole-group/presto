"""Helpers for loading molecules from parameterisation inputs."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from openff.toolkit import Molecule
from rdkit import Chem

MoleculeInputType = Literal["smiles", "sdf"]

MoleculeLoader = Callable[[str], list[Molecule]]


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
