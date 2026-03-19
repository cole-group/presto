"""Provides a custom RDKit toolkit wrapper that allows for other aromaticity models to be used when matching SMARTS patterns."""

from cachetools import LRUCache, cached
from openff.toolkit import Molecule, RDKitToolkitWrapper, unit
from openff.toolkit.utils import base_wrapper
from openff.toolkit.utils.constants import (
    DEFAULT_AROMATICITY_MODEL,
)
from openff.toolkit.utils.exceptions import InvalidAromaticityModelError


class PermissiveAromaticityRDKitToolkitWrapper(RDKitToolkitWrapper):  # type: ignore[misc]
    """A custom RDKit toolkit wrapper that allows for other aromaticity models to be used when matching SMARTS"""

    @cached(LRUCache(maxsize=4096), key=base_wrapper._mol_to_ctab_and_aro_key)
    def _connection_table_to_rdkit(
        self, molecule: Molecule, aromaticity_model: str = DEFAULT_AROMATICITY_MODEL
    ) -> "Chem.Mol":
        from rdkit import Chem

        # Create an editable RDKit molecule
        rdmol = Chem.RWMol()

        _bondtypes = {
            1: Chem.BondType.SINGLE,
            1.5: Chem.BondType.AROMATIC,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.QUADRUPLE,
            5: Chem.BondType.QUINTUPLE,
            6: Chem.BondType.HEXTUPLE,
            7: Chem.BondType.ONEANDAHALF,
        }

        for index, atom in enumerate(molecule.atoms):
            rdatom = Chem.Atom(atom.atomic_number)
            rdatom.SetFormalCharge(atom.formal_charge.m_as(unit.elementary_charge))
            rdatom.SetIsAromatic(atom.is_aromatic)

            # Stereo handling code moved to after bonds are added
            if atom.stereochemistry == "S":
                rdatom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
            elif atom.stereochemistry == "R":
                rdatom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)

            # Stop rdkit from adding implicit hydrogens
            rdatom.SetNoImplicit(True)

            rd_index = rdmol.AddAtom(rdatom)

            # Let's make sure al the atom indices in the two molecules
            # are the same, otherwise we need to create an atom map.
            assert index == atom.molecule_atom_index
            assert index == rd_index

        for bond in molecule.bonds:
            atom_indices = (
                bond.atom1.molecule_atom_index,
                bond.atom2.molecule_atom_index,
            )
            rdmol.AddBond(*atom_indices)
            rdbond = rdmol.GetBondBetweenAtoms(*atom_indices)
            # Assign bond type, which is based on order unless it is aromatic
            if bond.is_aromatic:
                rdbond.SetBondType(_bondtypes[1.5])
                rdbond.SetIsAromatic(True)
            else:
                rdbond.SetBondType(_bondtypes[bond.bond_order])
                rdbond.SetIsAromatic(False)

        Chem.SanitizeMol(
            rdmol,
            Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS ^ Chem.SANITIZE_SETAROMATICITY,
        )

        try:
            aromaticity_model_enum = getattr(Chem.AromaticityModel, aromaticity_model)
        except AttributeError:
            raise InvalidAromaticityModelError(
                f"Invalid aromaticity model: {aromaticity_model}. "
                f"Allowed models are: {Chem.AromaticityModel.names.keys()}."
            ) from None

        Chem.SetAromaticity(rdmol, aromaticity_model_enum)

        # Assign atom stereochemsitry and collect atoms for which RDKit
        # can't figure out chirality. The _CIPCode property of these atoms
        # will be forcefully set to the stereo we want (see #196).
        undefined_stereo_atoms = {}
        for index, atom in enumerate(molecule.atoms):
            rdatom = rdmol.GetAtomWithIdx(index)

            # Skip non-chiral atoms.
            if atom.stereochemistry is None:
                continue

            # Let's randomly assign this atom's (local) stereo to CW
            # and check if this causes the (global) stereo to be set
            # to the desired one (S or R).
            rdatom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
            # We need to do force and cleanIt to recalculate CIP stereo.
            Chem.AssignStereochemistry(rdmol, force=True, cleanIt=True)
            # If our random initial assignment worked, then we're set.
            if (
                rdatom.HasProp("_CIPCode")
                and rdatom.GetProp("_CIPCode") == atom.stereochemistry
            ):
                continue

            # Otherwise, set it to CCW.
            rdatom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
            # We need to do force and cleanIt to recalculate CIP stereo.
            Chem.AssignStereochemistry(rdmol, force=True, cleanIt=True)
            # Hopefully this worked, otherwise something's wrong
            if (
                rdatom.HasProp("_CIPCode")
                and rdatom.GetProp("_CIPCode") == atom.stereochemistry
            ):
                continue

            # Keep track of undefined stereo atoms. We'll force stereochemistry
            # at the end to avoid the next AssignStereochemistry to overwrite.
            if not rdatom.HasProp("_CIPCode"):
                undefined_stereo_atoms[rdatom] = atom.stereochemistry
                continue

            # Something is wrong.
            err_msg = (
                "Unknown atom stereochemistry encountered in to_rdkit. "
                f"Desired stereochemistry: {atom.stereochemistry}. "
                f"Set stereochemistry {rdatom.GetProp('_CIPCode')}"
            )
            raise RuntimeError(err_msg)

        # Copy bond stereo info from molecule to rdmol.
        self._assign_rdmol_bonds_stereo(molecule, rdmol)

        # Cleanup the rdmol
        rdmol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(rdmol)

        # Forcefully assign stereo information on the atoms that RDKit
        # can't figure out. This must be done last as calling AssignStereochemistry
        # again will delete these properties (see #196).
        for rdatom, stereochemistry in undefined_stereo_atoms.items():
            rdatom.SetProp("_CIPCode", stereochemistry)

        return rdmol
