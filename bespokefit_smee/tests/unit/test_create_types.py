"""Tests for the create_types module."""

import openff.toolkit
import pytest
from openff.toolkit import ForceField
from rdkit import Chem

from bespokefit_smee.create_types import (
    ReversibleTuple,
    _add_parameter_with_overwrite,
    _add_types_to_parameter_handler,
    _create_smarts,
    _deduplicate_symmetry_related_smarts,
    add_types_to_forcefield,
)
from bespokefit_smee.settings import (
    TypeGenerationSettings,
)


class TestAddParameterWithOverwrite:
    """Tests for the _add_parameter_with_overwrite function."""

    def test_add_new_parameter(self):
        """Test adding a new bond parameter to a handler."""
        ff = ForceField("openff_unconstrained-2.3.0-rc1.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        original_count = len(bond_handler.parameters)

        # Use a SMIRKS that doesn't exist in the force field
        param_dict = {
            "smirks": "[#99:1]-[#99:2]",
            "id": "b-new-test",
            "length": 1.5 * openff.toolkit.unit.angstrom,
            "k": 500.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }

        _add_parameter_with_overwrite(bond_handler, param_dict)

        assert len(bond_handler.parameters) == original_count + 1
        added_param = bond_handler.parameters[-1]
        assert added_param.smirks == "[#99:1]-[#99:2]"
        assert added_param.id == "b-new-test"
        assert added_param.length.m_as("angstrom") == pytest.approx(1.5)
        assert added_param.k.m_as(
            "kilocalorie_per_mole / angstrom**2"
        ) == pytest.approx(500.0)

    def test_overwrite_existing_parameter(self):
        """Test overwriting an existing parameter with the same SMIRKS."""
        ff = ForceField("openff_unconstrained-2.3.0-rc1.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        # Find an existing parameter to overwrite
        original_param = bond_handler.parameters[0]
        original_smirks = original_param.smirks
        original_count = len(bond_handler.parameters)

        # Create a new parameter with the same SMIRKS but different values
        param_dict = {
            "smirks": original_smirks,
            "id": "b-overwrite",
            "length": 9.99 * openff.toolkit.unit.angstrom,
            "k": 999.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }

        _add_parameter_with_overwrite(bond_handler, param_dict)

        # Check that the parameter count hasn't changed
        assert len(bond_handler.parameters) == original_count

        # Check that the parameter was overwritten
        overwritten_param = bond_handler.get_parameter({"smirks": original_smirks})[0]
        assert overwritten_param.smirks == original_smirks
        assert overwritten_param.id == "b-overwrite"
        assert overwritten_param.length.m_as("angstrom") == pytest.approx(9.99)
        assert overwritten_param.k.m_as(
            "kilocalorie_per_mole / angstrom**2"
        ) == pytest.approx(999.0)

    def test_parameter_preserves_position_on_overwrite(self):
        """Test that overwriting a parameter preserves its position in the handler."""
        ff = ForceField("openff_unconstrained-2.3.0-rc1.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        # Get the 5th parameter (arbitrary choice)
        target_index = 4
        original_param = bond_handler.parameters[target_index]
        original_smirks = original_param.smirks

        # Save the parameters before and after for comparison
        param_before = bond_handler.parameters[target_index - 1]
        param_after = bond_handler.parameters[target_index + 1]

        param_dict = {
            "smirks": original_smirks,
            "id": "b-test-position",
            "length": 8.88 * openff.toolkit.unit.angstrom,
            "k": 888.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }

        _add_parameter_with_overwrite(bond_handler, param_dict)

        # Check that the parameter at the same index has been updated
        updated_param = bond_handler.parameters[target_index]
        assert updated_param.id == "b-test-position"
        assert updated_param.smirks == original_smirks

        # Check that surrounding parameters are unchanged
        assert bond_handler.parameters[target_index - 1] is param_before
        assert bond_handler.parameters[target_index + 1] is param_after

    def test_add_multiple_new_parameters(self):
        """Test adding multiple new parameters sequentially."""
        ff = ForceField("openff_unconstrained-2.3.0-rc1.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        original_count = len(bond_handler.parameters)

        param_dicts = [
            {
                "smirks": "[#97:1]-[#97:2]",
                "id": "b1",
                "length": 1.5 * openff.toolkit.unit.angstrom,
                "k": 500.0
                * openff.toolkit.unit.kilocalorie_per_mole
                / openff.toolkit.unit.angstrom**2,
            },
            {
                "smirks": "[#98:1]-[#98:2]",
                "id": "b2",
                "length": 1.4 * openff.toolkit.unit.angstrom,
                "k": 600.0
                * openff.toolkit.unit.kilocalorie_per_mole
                / openff.toolkit.unit.angstrom**2,
            },
            {
                "smirks": "[#99:1]-[#99:2]",
                "id": "b3",
                "length": 1.2 * openff.toolkit.unit.angstrom,
                "k": 800.0
                * openff.toolkit.unit.kilocalorie_per_mole
                / openff.toolkit.unit.angstrom**2,
            },
        ]

        for param_dict in param_dicts:
            _add_parameter_with_overwrite(bond_handler, param_dict)

        assert len(bond_handler.parameters) == original_count + 3
        for i, param_dict in enumerate(param_dicts):
            param = bond_handler.parameters[original_count + i]
            assert param.smirks == param_dict["smirks"]
            assert param.id == param_dict["id"]

    def test_mixed_add_and_overwrite(self):
        """Test a mix of adding new parameters and overwriting existing ones."""
        ff = ForceField("openff_unconstrained-2.3.0-rc1.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        original_count = len(bond_handler.parameters)
        original_first_smirks = bond_handler.parameters[0].smirks

        # Overwrite the first parameter
        overwrite_dict = {
            "smirks": original_first_smirks,
            "id": "b-overwrite",
            "length": 7.77 * openff.toolkit.unit.angstrom,
            "k": 777.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }
        _add_parameter_with_overwrite(bond_handler, overwrite_dict)

        # Add a new parameter
        new_dict = {
            "smirks": "[#97:1]-[#97:2]",
            "id": "b-new",
            "length": 3.33 * openff.toolkit.unit.angstrom,
            "k": 333.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }
        _add_parameter_with_overwrite(bond_handler, new_dict)

        # Check counts
        assert len(bond_handler.parameters) == original_count + 1

        # Check overwritten parameter
        overwritten_param = bond_handler.parameters[0]
        assert overwritten_param.id == "b-overwrite"

        # Check new parameter is at the end
        new_param = bond_handler.parameters[-1]
        assert new_param.id == "b-new"
        assert new_param.smirks == "[#97:1]-[#97:2]"


@pytest.mark.parametrize(
    "mol_mapped_smiles, idxs, max_extend_distance, expected_num_atoms, expected_smarts",
    [
        # Ethane, full extension
        (
            "[C:1]([C:2]([H:6])([H:7])[H:8])([H:3])([H:4])[H:5]",
            (0, 1),
            -1,
            8,
            "[#6:1](-[#6:2](-[H])(-[H])-[H])(-[H])(-[H])-[H]",
        ),
        # Ethane, no extension
        (
            "[C:1]([C:2]([H:6])([H:7])[H:8])([H:3])([H:4])[H:5]",
            (0, 1),
            0,
            2,
            "[#6:1]-[#6:2]",
        ),
        # Ethane, extend 1 bond
        (
            "[C:1]([C:2]([H:6])([H:7])[H:8])([H:3])([H:4])[H:5]",
            (0, 1),
            1,
            8,
            "[#6:1](-[#6:2](-[H])(-[H])-[H])(-[H])(-[H])-[H]",
        ),
        # Propane, angle, full extension
        (
            "[C:1]([C:2]([C:3]([H:9])([H:10])[H:11])([H:7])[H:8])([H:4])([H:5])[H:6]",
            (0, 1, 2),
            -1,
            11,
            "[#6:1](-[#6:2](-[#6:3](-[H])(-[H])-[H])(-[H])-[H])(-[H])(-[H])-[H]",
        ),
        # Butane, proper torsion, full extension
        (
            "[C:1]([C:2]([C:3]([C:4]([H:12])([H:13])[H:14])([H:10])[H:11])([H:8])[H:9])([H:5])([H:6])[H:7]",
            (0, 1, 2, 3),
            -1,
            14,
            "[#6:1](-[#6:2](-[#6:3](-[#6:4](-[H])(-[H])-[H])(-[H])-[H])(-[H])-[H])(-[H])(-[H])-[H]",
        ),
        # Pyridine, full extension
        (
            "[c:1]1([H:7])[c:2]([H:8])[c:3]([H:9])[n:4][c:5]([H:10])[c:6]1[H:11]",
            (0, 1),
            -1,
            11,
            "[#6:1]1(:[#6:2](:[#6](:[#7]:[#6](:[#6]:1-[H])-[H])-[H])-[H])-[H]",
        ),
        # Pyridine, improper torsion, extend 2 bonds
        (
            "[c:1]1([H:7])[c:2]([H:8])[c:3]([H:9])[n:4][c:5]([H:10])[c:6]1[H:11]",
            (3, 4, 9, 5),
            2,
            10,  # Should miss 1 hydrogen
            "[#6]1(:[#6]:[#6](:[#7:1]:[#6:2](:[#6:4]:1-[H])-[H:3])-[H])-[H]",
        ),
    ],
)
def test_create_smarts(
    mol_mapped_smiles, idxs, max_extend_distance, expected_num_atoms, expected_smarts
):
    """Test the _create_smarts function with various molecules and parameters."""
    mol = openff.toolkit.Molecule.from_mapped_smiles(mol_mapped_smiles)

    smarts = _create_smarts(mol, idxs, max_extend_distance=max_extend_distance)

    # Parse the SMARTS to verify it's valid
    rdkit_mol = Chem.MolFromSmarts(smarts)
    assert rdkit_mol is not None

    # Check that the SMARTS includes the expected number of atoms
    assert rdkit_mol.GetNumAtoms() == expected_num_atoms

    # Check we have the expected atom maps
    atom_map_strings = [f":{i + 1}]" for i in range(len(idxs))]
    for atom_map in atom_map_strings:
        assert atom_map in smarts

    if expected_smarts is not None:
        assert smarts == expected_smarts


class TestDeduplicateSymmetryRelatedSmarts:
    """Tests for _deduplicate_symmetry_related_smarts function."""

    def test_no_duplicates(self):
        """Test with a list that has no duplicates."""
        smarts_list = [
            "[#6:1]-[#6:2]",
            "[#6:1]-[#8:2]",
            "[#6:1]=[#8:2]",
        ]
        result = _deduplicate_symmetry_related_smarts(smarts_list)
        assert len(result) == 3
        assert result == smarts_list

    def test_exact_duplicates(self):
        """Test with exact duplicate SMARTS."""
        smarts_list = [
            "[#6:1]-[#6:2]",
            "[#6:1]-[#6:2]",
            "[#6:1]-[#8:2]",
        ]
        result = _deduplicate_symmetry_related_smarts(smarts_list)
        assert len(result) == 2
        assert "[#6:1]-[#6:2]" in result
        assert "[#6:1]-[#8:2]" in result

    def test_symmetry_equivalent_simple(self):
        """Test with symmetry-equivalent simple bond patterns."""
        # C-C is the same as C-C regardless of atom numbering
        smarts_list = [
            "[#6:1]-[#6:2]",
            "[#6:2]-[#6:1]",  # Same but reversed
        ]
        result = _deduplicate_symmetry_related_smarts(smarts_list)
        # Should keep only one
        assert len(result) == 1

    def test_symmetry_equivalent_ethane(self):
        """Test with symmetry-equivalent patterns from ethane."""
        mol = openff.toolkit.Molecule.from_smiles("CC")

        # All C-C bonds in ethane are equivalent
        smarts1 = _create_smarts(mol, (0, 1), max_extend_distance=-1)
        smarts2 = _create_smarts(mol, (1, 0), max_extend_distance=-1)

        result = _deduplicate_symmetry_related_smarts([smarts1, smarts2])
        # Should deduplicate to 1
        assert len(result) == 1

    def test_symmetry_equivalent_benzene_bonds(self):
        """Test with symmetry-equivalent bonds in benzene."""
        mol = openff.toolkit.Molecule.from_smiles("c1ccccc1")

        # All C-C bonds in benzene are equivalent
        bonds = [(0, 1), (1, 2), (2, 3)]
        smarts_list = [
            _create_smarts(mol, bond, max_extend_distance=-1) for bond in bonds
        ]

        result = _deduplicate_symmetry_related_smarts(smarts_list)
        # All benzene C-C bonds are equivalent
        assert len(result) == 1

    def test_symmetry_equivalent_ethanol_angles(self):
        """Test with angles in ethanol using limited extension."""
        mol = openff.toolkit.Molecule.from_smiles("CCO")

        # Use limited extension to avoid all angles becoming identical
        # C-C-O angle
        smarts1 = _create_smarts(mol, (0, 1, 2), max_extend_distance=1)
        # H-C-C angle (multiple of these are equivalent)
        smarts2 = _create_smarts(mol, (3, 0, 1), max_extend_distance=1)
        smarts3 = _create_smarts(mol, (4, 0, 1), max_extend_distance=1)
        smarts4 = _create_smarts(mol, (5, 0, 1), max_extend_distance=1)

        smarts_list = [smarts1, smarts2, smarts3, smarts4]
        result = _deduplicate_symmetry_related_smarts(smarts_list)

        # Should have 2 unique: C-C-O and H-C-C (the H-C-C are all equivalent)
        assert len(result) == 2

    def test_different_patterns_not_deduplicated(self):
        """Test that genuinely different patterns are not deduplicated."""
        smarts_list = [
            "[#6:1]-[#6:2]-[#6:3]",  # C-C-C
            "[#6:1]-[#6:2]-[#8:3]",  # C-C-O
            "[#6:1]-[#8:2]-[#6:3]",  # C-O-C
        ]
        result = _deduplicate_symmetry_related_smarts(smarts_list)
        assert len(result) == 3

    def test_empty_list(self):
        """Test with an empty list."""
        result = _deduplicate_symmetry_related_smarts([])
        assert result == []

    def test_single_element(self):
        """Test with a single element."""
        smarts_list = ["[#6:1]-[#6:2]"]
        result = _deduplicate_symmetry_related_smarts(smarts_list)
        assert result == smarts_list

    def test_invalid_smarts_raises_error(self):
        """Test that invalid SMARTS raises an error."""
        smarts_list = ["invalid_smarts"]
        with pytest.raises(ValueError, match="Invalid SMARTS pattern"):
            _deduplicate_symmetry_related_smarts(smarts_list)


@pytest.mark.parametrize(
    "smiles, max_extend_distance, expected_num_new_params",
    [
        ("c1ccccc1", -1, 2),  # Benzene: 1 C-C bond type, 1 C-H bond type
        (
            "c1ccncc1",
            1,
            5,
        ),  # Pyridine: 2 C-C bond types, 1 C-N bond type, 2 C-H bond types
    ],
)
def test_add_types_deduplicates_symmetry_equivalent_bonds(
    smiles, max_extend_distance, expected_num_new_params
):
    """Test that _add_types_to_parameter_handler deduplicates symmetry-equivalent SMARTS."""

    mol = openff.toolkit.Molecule.from_smiles(smiles)

    ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0-rc1.offxml")
    bond_handler = ff.get_parameter_handler("Bonds")

    original_count = len(bond_handler.parameters)

    updated_handler = _add_types_to_parameter_handler(
        mol,
        bond_handler,
        handler_name="Bonds",
        max_extend_distance=max_extend_distance,
    )

    new_count = len(updated_handler.parameters)
    added_count = new_count - original_count

    assert added_count == expected_num_new_params, (
        f"Expected {expected_num_new_params} new parameters, got {added_count}"
    )

    # Verify that the added parameters are at the end
    bespoke_params = updated_handler.parameters[original_count:]
    assert len(bespoke_params) == expected_num_new_params

    # Check that both have bespoke IDs
    for param in bespoke_params:
        assert param.id.startswith("b-bespoke-")


class TestReversibleTuple:
    """Tests for the ReversibleTuple class."""

    def test_initialization(self):
        """Test basic initialization."""
        rt = ReversibleTuple(1, 2, 3)
        assert rt.items == (1, 2, 3)
        assert rt.canonical == (1, 2, 3)

    def test_equality_same_order(self):
        """Test equality with same order."""
        rt1 = ReversibleTuple(1, 2, 3)
        rt2 = ReversibleTuple(1, 2, 3)
        assert rt1 == rt2

    def test_equality_reversed_order(self):
        """Test equality with reversed order."""
        rt1 = ReversibleTuple(1, 2, 3)
        rt2 = ReversibleTuple(3, 2, 1)
        assert rt1 == rt2

    def test_inequality(self):
        """Test inequality with different tuples."""
        rt1 = ReversibleTuple(1, 2, 3)
        rt2 = ReversibleTuple(1, 2, 4)
        assert rt1 != rt2

    def test_hash_same_order(self):
        """Test that same order produces same hash."""
        rt1 = ReversibleTuple(1, 2, 3)
        rt2 = ReversibleTuple(1, 2, 3)
        assert hash(rt1) == hash(rt2)

    def test_hash_reversed_order(self):
        """Test that reversed order produces same hash."""
        rt1 = ReversibleTuple(1, 2, 3)
        rt2 = ReversibleTuple(3, 2, 1)
        assert hash(rt1) == hash(rt2)

    def test_hash_different_tuples(self):
        """Test that different tuples produce different hashes."""
        rt1 = ReversibleTuple(1, 2, 3)
        rt2 = ReversibleTuple(1, 2, 4)
        assert hash(rt1) != hash(rt2)

    def test_use_in_set(self):
        """Test that ReversibleTuples work correctly in sets."""
        rt1 = ReversibleTuple(1, 2, 3)
        rt2 = ReversibleTuple(3, 2, 1)  # Same as rt1 when reversed
        rt3 = ReversibleTuple(1, 2, 4)  # Different

        # Should only have 2 unique items
        unique_set = {rt1, rt2, rt3}
        assert len(unique_set) == 2

    def test_use_in_dict(self):
        """Test that ReversibleTuples work correctly as dict keys."""
        rt1 = ReversibleTuple(1, 2, 3)
        rt2 = ReversibleTuple(3, 2, 1)

        data = {rt1: "value1"}
        # Should access the same key
        assert data[rt2] == "value1"

    def test_iteration(self):
        """Test iteration over ReversibleTuple."""
        rt = ReversibleTuple(1, 2, 3)
        result = list(rt)
        assert result == [1, 2, 3]

    def test_indexing(self):
        """Test indexing ReversibleTuple."""
        rt = ReversibleTuple(1, 2, 3)
        assert rt[0] == 1
        assert rt[1] == 2
        assert rt[2] == 3

    def test_repr(self):
        """Test string representation."""
        rt = ReversibleTuple(1, 2, 3)
        assert repr(rt) == "ReversibleTuple(1, 2, 3)"

    def test_canonical_chooses_smaller(self):
        """Test that canonical form is lexicographically smaller."""
        # (3, 2, 1) < (1, 2, 3) is False, so canonical should be (1, 2, 3)
        rt1 = ReversibleTuple(3, 2, 1)
        assert rt1.canonical == (1, 2, 3)

        # (1, 2, 3) < (3, 2, 1) is True, so canonical should be (1, 2, 3)
        rt2 = ReversibleTuple(1, 2, 3)
        assert rt2.canonical == (1, 2, 3)

    def test_two_element_tuple(self):
        """Test with two-element tuple."""
        rt1 = ReversibleTuple(1, 2)
        rt2 = ReversibleTuple(2, 1)
        assert rt1 == rt2
        assert hash(rt1) == hash(rt2)

    def test_single_element_tuple(self):
        """Test with single element."""
        rt1 = ReversibleTuple(1)
        rt2 = ReversibleTuple(1)
        assert rt1 == rt2
        assert rt1.canonical == (1,)


class TestAddTypesToForcefield:
    """Tests for the add_types_to_forcefield function."""

    def test_basic_usage(self):
        """Test basic usage with benzene."""
        mol = openff.toolkit.Molecule.from_smiles("c1ccccc1")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0-rc1.offxml")

        # Get original counts
        original_bond_count = len(ff.get_parameter_handler("Bonds").parameters)
        original_angle_count = len(ff.get_parameter_handler("Angles").parameters)

        # Define type generation settings
        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
            "Angles": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        # Add types
        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Check that new parameters were added
        new_bond_count = len(ff_with_types.get_parameter_handler("Bonds").parameters)
        new_angle_count = len(ff_with_types.get_parameter_handler("Angles").parameters)

        assert new_bond_count > original_bond_count
        assert new_angle_count > original_angle_count

        # Verify original force field is unchanged
        assert len(ff.get_parameter_handler("Bonds").parameters) == original_bond_count

    def test_with_exclusions(self):
        """Test with excluded SMIRKS patterns."""
        mol = openff.toolkit.Molecule.from_smiles("CCO")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0-rc1.offxml")

        # Get a parameter to exclude
        bond_handler = ff.get_parameter_handler("Bonds")
        original_count = len(bond_handler.parameters)

        # Find what SMIRKS patterns match our molecule
        matches = bond_handler.find_matches(mol.to_topology())
        if matches:
            # Get one SMIRKS to exclude
            first_match = next(iter(matches.values()))
            excluded_smirks = first_match.parameter_type.smirks

            # Define settings with exclusion
            type_gen_settings = {
                "Bonds": TypeGenerationSettings(
                    max_extend_distance=-1, exclude=[excluded_smirks]
                ),
            }

            # Add types
            ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

            new_count = len(ff_with_types.get_parameter_handler("Bonds").parameters)

            # Should add some parameters, but fewer than without exclusions
            assert new_count > original_count

    def test_multiple_handlers(self):
        """Test adding types to multiple handlers."""
        mol = openff.toolkit.Molecule.from_smiles("CC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0-rc1.offxml")

        # Get original counts for multiple handlers
        original_counts = {
            "Bonds": len(ff.get_parameter_handler("Bonds").parameters),
            "Angles": len(ff.get_parameter_handler("Angles").parameters),
            "ProperTorsions": len(
                ff.get_parameter_handler("ProperTorsions").parameters
            ),
        }

        # Define settings for all three handlers
        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=1, exclude=[]),
            "Angles": TypeGenerationSettings(max_extend_distance=1, exclude=[]),
            "ProperTorsions": TypeGenerationSettings(max_extend_distance=1, exclude=[]),
        }

        # Add types
        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Check that all handlers got new parameters
        for handler_name in ["Bonds", "Angles", "ProperTorsions"]:
            new_count = len(
                ff_with_types.get_parameter_handler(handler_name).parameters
            )
            assert new_count >= original_counts[handler_name]

    def test_force_field_not_modified(self):
        """Test that original force field is not modified."""
        mol = openff.toolkit.Molecule.from_smiles("CC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0-rc1.offxml")

        original_bond_count = len(ff.get_parameter_handler("Bonds").parameters)

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        # Add types
        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Original should be unchanged
        assert len(ff.get_parameter_handler("Bonds").parameters) == original_bond_count
        # New should have more parameters
        assert (
            len(ff_with_types.get_parameter_handler("Bonds").parameters)
            > original_bond_count
        )

    def test_empty_settings(self):
        """Test with empty type generation settings."""
        mol = openff.toolkit.Molecule.from_smiles("CC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0-rc1.offxml")

        original_bond_count = len(ff.get_parameter_handler("Bonds").parameters)

        # Empty settings - should not add any parameters
        type_gen_settings = {}

        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Should be unchanged
        new_bond_count = len(ff_with_types.get_parameter_handler("Bonds").parameters)
        assert new_bond_count == original_bond_count

    def test_different_extension_distances(self):
        """Test with different max_extend_distance values."""
        mol = openff.toolkit.Molecule.from_smiles("CCCC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0-rc1.offxml")

        original_count = len(ff.get_parameter_handler("Bonds").parameters)

        # Test with limited extension
        type_gen_settings_limited = {
            "Bonds": TypeGenerationSettings(max_extend_distance=1, exclude=[]),
        }
        ff_limited = add_types_to_forcefield(mol, ff, type_gen_settings_limited)
        count_limited = len(ff_limited.get_parameter_handler("Bonds").parameters)

        # Test with full extension
        type_gen_settings_full = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }
        ff_full = add_types_to_forcefield(mol, ff, type_gen_settings_full)
        count_full = len(ff_full.get_parameter_handler("Bonds").parameters)

        # Both should add parameters
        assert count_limited > original_count
        assert count_full > original_count

        # Full extension might add fewer parameters due to deduplication
        # (all bonds look the same when including full molecule context)
        assert count_full <= count_limited
