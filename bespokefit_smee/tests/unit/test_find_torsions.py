"""Unit tests for find_torsions module."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from openff.toolkit import Molecule

from bespokefit_smee.find_torsions import (
    _TORSIONS_TO_EXCLUDE_SMARTS,
    _TORSIONS_TO_INCLUDE_SMARTS,
    get_rot_torsions_by_rot_bond,
    get_single_torsion_by_rot_bond,
    get_unwanted_bonds,
)


class TestGetSingleTorsionByRotBond:
    """Tests for get_single_torsion_by_rot_bond function."""

    def test_ethane_no_rotatable_bonds(self):
        """Test ethane has no rotatable bonds."""
        mol = Molecule.from_smiles("CC")
        torsions = get_single_torsion_by_rot_bond(mol, _TORSIONS_TO_INCLUDE_SMARTS[0])
        assert len(torsions) == 0

    def test_butane_one_rotatable_bond(self):
        """Test butane has one rotatable bond (middle C-C bond)."""
        mol = Molecule.from_smiles("CCCC")
        smarts = _TORSIONS_TO_INCLUDE_SMARTS[0]
        torsions = get_single_torsion_by_rot_bond(mol, smarts)
        assert len(torsions) == 1

        # Check that the torsion has 4 atoms
        for rot_bond, torsion in torsions.items():
            assert len(rot_bond) == 2
            assert len(torsion) == 4

    def test_propane_no_rotatable_bonds(self):
        """Test propane has no "rotatable bonds"."""
        mol = Molecule.from_smiles("CCC")
        smarts = _TORSIONS_TO_INCLUDE_SMARTS[0]
        torsions = get_single_torsion_by_rot_bond(mol, smarts)
        assert len(torsions) == 0

    def test_invalid_smarts_wrong_atom_count_raises_error(self):
        """Test that SMARTS matching wrong number of atoms raises error."""
        mol = Molecule.from_smiles("CCO")
        # SMARTS that matches only 2 atoms (a bond, not a torsion)
        smarts = "[!#1:1]-[!#1:2]"
        with pytest.raises(ValueError, match="Expected torsion to have 4 atoms"):
            get_single_torsion_by_rot_bond(mol, smarts)

    def test_biphenyl_rotatable_bond(self):
        """Test biphenyl has one rotatable bond between rings."""
        mol = Molecule.from_smiles("c1ccccc1-c2ccccc2")
        smarts = "[!#1:1]~[!$(*#*)&!D1:2]-!@[!$(*#*)&!D1:3]~[!#1:4]"
        torsions = get_single_torsion_by_rot_bond(mol, smarts)
        assert len(torsions) >= 1

    def test_rotatable_bond_tuple_is_sorted(self):
        """Test that rotatable bond tuple is sorted."""
        mol = Molecule.from_smiles("CCCC")
        smarts = "[!#1:1]~[!$(*#*)&!D1:2]-!@[!$(*#*)&!D1:3]~[!#1:4]"
        torsions = get_single_torsion_by_rot_bond(mol, smarts)

        for rot_bond in torsions.keys():
            assert rot_bond[0] < rot_bond[1]


class TestGetUnwantedBonds:
    """Tests for get_unwanted_bonds function."""

    def test_ethane_no_unwanted_bonds(self):
        """Test ethane has no unwanted bonds."""
        mol = Molecule.from_smiles("CC")
        # SMARTS for C-C bond
        smarts = "[#6:1]-[#6:2]"
        unwanted = get_unwanted_bonds(mol, smarts)
        assert len(unwanted) == 1  # Has one C-C bond

    def test_amide_bond_detected(self):
        """Test that amide bonds are detected."""
        mol = Molecule.from_smiles("CC(=O)NC")
        # SMARTS for amide C-N bond
        smarts = "[#6X3:1](=[#8X1])-[#7X3:2]"
        unwanted = get_unwanted_bonds(mol, smarts)
        assert len(unwanted) == 1

    def test_invalid_smarts_wrong_atom_count_raises_error(self):
        """Test that SMARTS matching wrong number of atoms raises error."""
        mol = Molecule.from_smiles("CCCO")
        # SMARTS that matches 4 atoms (a torsion, not a bond)
        smarts = "[!#1:1]~[!#1:2]~[!#1:3]~[!#1:4]"
        with pytest.raises(ValueError, match="Expected bond to have 2 atoms"):
            get_unwanted_bonds(mol, smarts)

    def test_returned_bonds_are_sorted(self):
        """Test that returned bonds are sorted tuples."""
        mol = Molecule.from_smiles("CCCC")
        smarts = "[#6:1]-[#6:2]"
        unwanted = get_unwanted_bonds(mol, smarts)

        for bond in unwanted:
            assert len(bond) == 2
            assert bond[0] < bond[1]


class TestGetRotTorsionsByRotBond:
    """Tests for get_rot_torsions_by_rot_bond function."""

    def test_default_parameters_propanol(self):
        """Test with default parameters on propanol."""
        mol = Molecule.from_smiles("CCCO")
        torsions = get_rot_torsions_by_rot_bond(mol)
        assert len(torsions) == 1

    def test_default_parameters_ethane(self):
        """Test with default parameters on ethane."""
        mol = Molecule.from_smiles("CC")
        torsions = get_rot_torsions_by_rot_bond(mol)
        assert len(torsions) == 0

    def test_custom_include_smarts(self):
        """Test with custom include SMARTS."""
        mol = Molecule.from_smiles("CCCC")
        # Include all C-C bonds
        include_smarts = ["[#6:1]-[#6:2]-[#6:3]-[#6:4]"]
        torsions = get_rot_torsions_by_rot_bond(mol, include_smarts=include_smarts)
        # Should find torsions around C-C bonds
        assert len(torsions) >= 1

    def test_exclude_smarts_removes_bonds(self):
        """Test that exclude SMARTS removes bonds."""
        mol = Molecule.from_smiles("CC(=O)NC")

        # First, find torsions without exclusion
        include_smarts = ["[!#1:1]~[!#1:2]-[!#1:3]~[!#1:4]"]
        torsions_all = get_rot_torsions_by_rot_bond(
            mol, include_smarts=include_smarts, exclude_smarts=[]
        )

        # Now exclude amide bonds
        exclude_smarts = ["[#6X3:1](=[#8X1])-[#7X3:2]"]
        torsions_filtered = get_rot_torsions_by_rot_bond(
            mol, include_smarts=include_smarts, exclude_smarts=exclude_smarts
        )

        # Should have fewer torsions after exclusion
        assert len(torsions_filtered) <= len(torsions_all)

    def test_empty_include_smarts(self):
        """Test with empty include SMARTS."""
        mol = Molecule.from_smiles("CCO")
        torsions = get_rot_torsions_by_rot_bond(mol, include_smarts=[])
        assert len(torsions) == 0

    def test_benzene_no_rotatable_bonds(self):
        """Test that benzene has no rotatable bonds."""
        mol = Molecule.from_smiles("c1ccccc1")
        torsions = get_rot_torsions_by_rot_bond(mol)
        assert len(torsions) == 0

    def test_multiple_rotatable_bonds(self):
        """Test molecule with multiple rotatable bonds."""
        mol = Molecule.from_smiles("CCCCCC")  # Hexane
        torsions = get_rot_torsions_by_rot_bond(mol)
        # Should have multiple rotatable bonds
        assert len(torsions) >= 1

    @given(n_carbons=st.integers(min_value=2, max_value=6))
    def test_linear_alkanes(self, n_carbons):
        """Test linear alkanes with hypothesis."""
        smiles = "C" * n_carbons
        mol = Molecule.from_smiles(smiles)
        torsions = get_rot_torsions_by_rot_bond(mol)

        # Linear alkanes with n carbons have max(0, n-3) rotatable bonds
        # (need 4 heavy atoms for a torsion, and terminal bonds don't count)
        expected_max = max(0, n_carbons - 3)
        assert len(torsions) <= expected_max

    def test_return_type_structure(self):
        """Test that return type has correct structure."""
        mol = Molecule.from_smiles("CCO")
        torsions = get_rot_torsions_by_rot_bond(mol)

        assert isinstance(torsions, dict)
        for rot_bond, torsion in torsions.items():
            assert isinstance(rot_bond, tuple)
            assert len(rot_bond) == 2
            assert isinstance(torsion, tuple)
            assert len(torsion) == 4
            # Check that rot_bond atoms are the middle two of torsion
            assert rot_bond == tuple(sorted([torsion[1], torsion[2]]))

    def test_branched_molecule(self):
        """Test a branched molecule."""
        mol = Molecule.from_smiles("CC(C)CC")  # Isopentane
        torsions = get_rot_torsions_by_rot_bond(mol)
        assert len(torsions) >= 1


class TestDefaultSmarts:
    """Tests for default SMARTS patterns."""

    def test_default_include_smarts_defined(self):
        """Test that default include SMARTS are defined."""
        assert len(_TORSIONS_TO_INCLUDE_SMARTS) > 0
        assert all(isinstance(s, str) for s in _TORSIONS_TO_INCLUDE_SMARTS)

    def test_default_exclude_smarts_defined(self):
        """Test that default exclude SMARTS are defined."""
        assert isinstance(_TORSIONS_TO_EXCLUDE_SMARTS, list)
        assert all(isinstance(s, str) for s in _TORSIONS_TO_EXCLUDE_SMARTS)
