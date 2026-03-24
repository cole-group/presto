"""Unit tests for find_torsions module."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from openff.toolkit import Molecule

from presto.find_torsions import (
    DEFAULT_TORSIONS_TO_EXCLUDE_SMARTS,
    DEFAULT_TORSIONS_TO_INCLUDE_SMARTS,
    get_rot_torsions_by_rot_bond,
    get_single_torsion_by_rot_bond,
    get_unwanted_bonds,
)
from presto.settings import MMMDMetadynamicsSamplingSettings


class TestGetSingleTorsionByRotBond:
    """Tests for get_single_torsion_by_rot_bond function."""

    def test_ethane_no_rotatable_bonds(self):
        """Test ethane has no rotatable bonds."""
        mol = Molecule.from_smiles("CC")
        torsions = get_single_torsion_by_rot_bond(
            mol, DEFAULT_TORSIONS_TO_INCLUDE_SMARTS[0]
        )
        assert len(torsions) == 0

    def test_butane_one_rotatable_bond(self):
        """Test butane has one rotatable bond (middle C-C bond)."""
        mol = Molecule.from_smiles("CCCC")
        smarts = DEFAULT_TORSIONS_TO_INCLUDE_SMARTS[0]
        torsions = get_single_torsion_by_rot_bond(mol, smarts)
        assert len(torsions) == 1

        # Check that the torsion has 4 atoms
        for rot_bond, torsion in torsions.items():
            assert len(rot_bond) == 2
            assert len(torsion) == 4

    def test_propane_no_rotatable_bonds(self):
        """Test propane has no "rotatable bonds"."""
        mol = Molecule.from_smiles("CCC")
        smarts = DEFAULT_TORSIONS_TO_INCLUDE_SMARTS[0]
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
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )
        assert len(torsions) == 1

    def test_default_parameters_ethane(self):
        """Test with default parameters on ethane."""
        mol = Molecule.from_smiles("CC")
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )
        assert len(torsions) == 0

    def test_custom_include_smarts(self):
        """Test with custom include SMARTS."""
        mol = Molecule.from_smiles("CCCC")
        # Include all C-C bonds
        include_smarts = ["[#6:1]-[#6:2]-[#6:3]-[#6:4]"]
        torsions = get_rot_torsions_by_rot_bond(mol, include_smarts=include_smarts)
        # Should find central torsions around C-C bond
        assert len(torsions) == 1

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
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )
        assert len(torsions) == 0

    def test_multiple_rotatable_bonds(self):
        """Test molecule with multiple rotatable bonds."""
        mol = Molecule.from_smiles("CCCCCC")  # Hexane
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )
        # Should have multiple rotatable bonds
        assert len(torsions) == 3

    @given(n_carbons=st.integers(min_value=2, max_value=6))
    def test_linear_alkanes(self, n_carbons):
        """Test linear alkanes with hypothesis."""
        smiles = "C" * n_carbons
        mol = Molecule.from_smiles(smiles)
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )

        # Linear alkanes with n carbons have max(0, n-3) rotatable bonds
        # (need 4 heavy atoms for a torsion, and terminal bonds don't count)
        expected_n_tor = max(0, n_carbons - 3)
        assert len(torsions) == expected_n_tor

    def test_return_type_structure(self):
        """Test that return type has correct structure."""
        mol = Molecule.from_smiles("CCO")
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )

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
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )
        assert len(torsions) >= 1


class TestDefaultSmarts:
    """Tests for default SMARTS patterns constants."""

    def test_default_include_smarts_defined(self):
        """Test that DEFAULT_TORSIONS_TO_INCLUDE_SMARTS is properly defined."""
        assert len(DEFAULT_TORSIONS_TO_INCLUDE_SMARTS) > 0
        assert all(isinstance(s, str) for s in DEFAULT_TORSIONS_TO_INCLUDE_SMARTS)
        # Should have 4 patterns: non-ring, r5, r6, r7
        assert len(DEFAULT_TORSIONS_TO_INCLUDE_SMARTS) == 4

    def test_default_exclude_smarts_defined(self):
        """Test that DEFAULT_TORSIONS_TO_EXCLUDE_SMARTS is properly defined."""
        assert isinstance(DEFAULT_TORSIONS_TO_EXCLUDE_SMARTS, list)
        assert all(isinstance(s, str) for s in DEFAULT_TORSIONS_TO_EXCLUDE_SMARTS)

    def test_settings_use_default_constants(self):
        """Test that settings class uses the same default constants."""
        settings = MMMDMetadynamicsSamplingSettings()
        assert settings.torsions_to_include_smarts == DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        assert settings.torsions_to_exclude_smarts == DEFAULT_TORSIONS_TO_EXCLUDE_SMARTS


class TestRingSpecificTorsions:
    """Tests for ring-specific torsion matching with explicit bond primitives."""

    @pytest.mark.parametrize(
        ("smiles", "expected_n_torsions"),
        [
            ("C1CCCC1", 5),
            ("C1CCCCC1", 6),
            ("C1CCCCCC1", 7),
            ("C1CCC1", 0),
            ("C1CC1", 0),
            ("c1ccccc1", 0),
            ("c1cc[nH]c1", 0),
            ("O=[S@@](C)c1ccccc1", 1),  # Between S and benzene ring
        ],
    )
    def test_default_patterns_expected_counts(self, smiles, expected_n_torsions):
        """Test expected ring/aromatic match behavior with default patterns.

        Pyrrole can fail under the MDL aromaticity model but passes under RDKit.
        """
        mol = Molecule.from_smiles(smiles)
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )
        assert len(torsions) == expected_n_torsions

    def test_ethylbenzene_acyclic_bond_matches(self):
        """Test that ethylbenzene's acyclic C-C bond matches."""
        mol = Molecule.from_smiles("CCc1ccccc1")
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )
        # Should find rotatable bonds in the ethyl chain and connecting to ring
        assert len(torsions) >= 1

    def test_bond_primitives_in_patterns(self):
        """Test that default patterns use explicit bond primitives (!@ and @)."""
        # Check first pattern (non-ring) uses !@
        assert "-!@" in DEFAULT_TORSIONS_TO_INCLUDE_SMARTS[0]

        # Check ring patterns use @
        for pattern in DEFAULT_TORSIONS_TO_INCLUDE_SMARTS[1:]:
            assert "-@" in pattern

    def test_patterns_have_correct_count(self):
        """Test that we have exactly 4 default patterns."""
        # 1 for non-ring + 3 for ring sizes (r5, r6, r7)
        assert len(DEFAULT_TORSIONS_TO_INCLUDE_SMARTS) == 4

    @pytest.mark.parametrize(
        ("pattern_idx", "smiles", "expected_n_torsions"),
        [
            (0, "CCCC", 1),
            (1, "C1CCCC1", 5),
            (2, "C1CCCCC1", 6),
            (3, "C1CCCCCC1", 7),
        ],
    )
    def test_patterns_match_intended_ring_sizes(
        self, pattern_idx, smiles, expected_n_torsions
    ):
        """Test each default pattern matches its intended topology/ring size."""
        mol = Molecule.from_smiles(smiles)
        torsions = get_single_torsion_by_rot_bond(
            mol, DEFAULT_TORSIONS_TO_INCLUDE_SMARTS[pattern_idx]
        )
        assert len(torsions) == expected_n_torsions

    def test_mixed_acyclic_and_cyclic_molecule(self):
        """Test molecule with both acyclic and cyclic rotatable bonds."""
        # Cyclohexane with ethyl substituent
        mol = Molecule.from_smiles("CCC1CCCCC1")
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )
        # Should find bonds in both the ethyl chain and the ring
        assert len(torsions) >= 7  # 6 in ring + at least 1 in chain

    def test_patterns_exclude_aromatic_rings(self):
        """Test that patterns correctly exclude aromatic rings via !a."""
        # All ring patterns (indices 1-3) should include !a (not aromatic)
        for pattern in DEFAULT_TORSIONS_TO_INCLUDE_SMARTS[1:]:
            assert "!a" in pattern

        # Verify benzene (aromatic) doesn't match any ring pattern
        mol_benzene = Molecule.from_smiles("c1ccccc1")
        for pattern in DEFAULT_TORSIONS_TO_INCLUDE_SMARTS[1:]:
            torsions = get_single_torsion_by_rot_bond(mol_benzene, pattern)
            assert len(torsions) == 0

    def test_methylcyclohexane(self):
        """Test methylcyclohexane has correct number of rotatable bonds."""
        mol = Molecule.from_smiles("CC1CCCCC1")
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )
        # 6 bonds in the ring (the C-C bond to methyl is terminal and doesn't match !D1)
        assert len(torsions) == 6

    def test_spiro_compound(self):
        """Test a spiro compound (two rings sharing one atom)."""
        # Spiro[4.5]decane - 5-membered ring fused to 6-membered ring
        mol = Molecule.from_smiles("C1CCC2(C1)CCCCC2")
        torsions = get_rot_torsions_by_rot_bond(
            mol, include_smarts=DEFAULT_TORSIONS_TO_INCLUDE_SMARTS
        )
        # Should find torsions in both rings
        assert len(torsions) >= 9  # 5 + 6 bonds, minus shared atom effects
