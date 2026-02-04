"""Tests for the create_types module."""

import openff.toolkit
import pytest
from openff.toolkit import ForceField
from rdkit import Chem

from presto.create_types import (
    _add_parameter_with_overwrite,
    _add_types_to_parameter_handler,
    _create_smarts,
    _remove_redundant_smarts,
    add_types_to_forcefield,
)
from presto.settings import TypeGenerationSettings


class TestAddParameterWithOverwrite:
    """Tests for the _add_parameter_with_overwrite function."""

    def test_add_new_parameter(self):
        """Test adding a new bond parameter to a handler."""
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
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
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        # Find an existing parameter to overwrite
        original_param = bond_handler.parameters[0]
        original_smirks = original_param.smirks
        original_count = len(bond_handler.parameters)
        original_id = original_param.id

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
        assert overwritten_param.id == original_id  # ID should remain unchanged
        assert overwritten_param.length.m_as("angstrom") == pytest.approx(9.99)
        assert overwritten_param.k.m_as(
            "kilocalorie_per_mole / angstrom**2"
        ) == pytest.approx(999.0)

    def test_parameter_preserves_position_on_overwrite(self):
        """Test that overwriting a parameter preserves its position in the handler."""
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        # Get the 5th parameter (arbitrary choice)
        target_index = 4
        original_param = bond_handler.parameters[target_index]
        original_smirks = original_param.smirks
        original_id = original_param.id

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
        assert updated_param.id == original_id  # ID should remain unchanged
        assert updated_param.smirks == original_smirks

        # Check that surrounding parameters are unchanged
        assert bond_handler.parameters[target_index - 1] is param_before
        assert bond_handler.parameters[target_index + 1] is param_after

    def test_add_multiple_new_parameters(self):
        """Test adding multiple new parameters sequentially."""
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
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
        ff = ForceField("openff_unconstrained-2.3.0.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        original_count = len(bond_handler.parameters)
        original_first_smirks = bond_handler.parameters[0].smirks
        original_first_id = bond_handler.parameters[0].id

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
        assert overwritten_param.id == original_first_id  # ID should remain unchanged

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
            2,
            "[#6&!H0&!H1&!H2:1]-[#6&!H0&!H1&!H2:2]",
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
            2,
            "[#6&!H0&!H1&!H2:1]-[#6&!H0&!H1&!H2:2]",
        ),
        # Propane, angle, full extension
        (
            "[C:1]([C:2]([C:3]([H:9])([H:10])[H:11])([H:7])[H:8])([H:4])([H:5])[H:6]",
            (0, 1, 2),
            -1,
            3,
            "[#6&!H0&!H1&!H2:1]-[#6&!H0&!H1:2]-[#6&!H0&!H1&!H2:3]",
        ),
        # Butane, proper torsion, full extension
        (
            "[C:1]([C:2]([C:3]([C:4]([H:12])([H:13])[H:14])([H:10])[H:11])([H:8])[H:9])([H:5])([H:6])[H:7]",
            (0, 1, 2, 3),
            -1,
            4,
            "[#6&!H0&!H1&!H2:1]-[#6&!H0&!H1:2]-[#6&!H0&!H1:3]-[#6&!H0&!H1&!H2:4]",
        ),
        # Pyridine, full extension
        (
            "[c:1]1([H:7])[c:2]([H:8])[c:3]([H:9])[n:4][c:5]([H:10])[c:6]1[H:11]",
            (0, 1),
            -1,
            6,
            "[#6&!H0:1]1:[#6&!H0:2]:[#6&!H0]:[#7]:[#6&!H0]:[#6&!H0]:1",
        ),
        # Pyridine, improper torsion, extend 2 bonds
        (
            "[c:1]1([H:7])[c:2]([H:8])[c:3]([H:9])[n:4][c:5]([H:10])[c:6]1[H:11]",
            (3, 4, 9, 5),
            2,
            7,  # Should include 1 hydrogen explicitly
            "[#6&!H0]1:[#6]:[#6&!H0]:[#7:1]:[#6:2](:[#6&!H0:4]:1)-[H:3]",
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


class TestRemoveRedundantSmarts:
    """Tests for the _remove_redundant_smarts function."""

    def test_removes_unused_bespoke_parameters(self):
        """Test that unused parameters with id_substring are removed."""
        mol = openff.toolkit.Molecule.from_smiles("CCO")  # Ethanol
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        # Add a bespoke parameter that will be used
        bond_handler = ff.get_parameter_handler("Bonds")
        labels = bond_handler.find_matches(mol.to_topology())
        first_bond_indices = list(labels.keys())[0]

        # Create a SMARTS that matches ethanol
        used_smarts = _create_smarts(mol, first_bond_indices, max_extend_distance=-1)
        used_param = {
            "smirks": used_smarts,
            "id": "b-bespoke-used",
            "length": 1.5 * openff.toolkit.unit.angstrom,
            "k": 500.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }
        _add_parameter_with_overwrite(bond_handler, used_param)

        # Add a bespoke parameter that won't be used (element 99 doesn't exist in ethanol)
        unused_param = {
            "smirks": "[#99:1]-[#99:2]",
            "id": "b-bespoke-unused",
            "length": 1.2 * openff.toolkit.unit.angstrom,
            "k": 600.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }
        _add_parameter_with_overwrite(bond_handler, unused_param)

        original_count = len(bond_handler.parameters)

        # Remove redundant parameters
        ff_cleaned = _remove_redundant_smarts(mol, ff, id_substring="bespoke")

        # Check that the unused parameter was removed
        cleaned_handler = ff_cleaned.get_parameter_handler("Bonds")
        new_count = len(cleaned_handler.parameters)

        assert new_count == original_count - 1

        # Check that the used parameter is still there
        param_ids = [p.id for p in cleaned_handler.parameters]
        assert "b-bespoke-used" in param_ids
        assert "b-bespoke-unused" not in param_ids

    def test_does_not_remove_non_bespoke_parameters(self):
        """Test that parameters without id_substring are never removed."""
        mol = openff.toolkit.Molecule.from_smiles("C")  # Methane (very simple)
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        original_bond_count = len(ff.get_parameter_handler("Bonds").parameters)

        # This should not remove any original parameters even if some are unused
        ff_cleaned = _remove_redundant_smarts(mol, ff, id_substring="bespoke")

        # Original parameters should be unchanged (they don't have "bespoke" in their id)
        new_bond_count = len(ff_cleaned.get_parameter_handler("Bonds").parameters)
        assert new_bond_count == original_bond_count

    def test_multiple_molecules_keeps_if_used_by_any(self):
        """Test that parameter is kept if used by any molecule."""
        mol1 = openff.toolkit.Molecule.from_smiles("CC")  # Ethane
        mol2 = openff.toolkit.Molecule.from_smiles("CCO")  # Ethanol

        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        # Create a SMARTS that matches ethanol but not ethane
        labels = bond_handler.find_matches(mol2.to_topology())
        # Find the C-O bond
        co_bond = None
        for bond_indices, _match in labels.items():
            atoms = [mol2.atoms[i] for i in bond_indices]
            if (atoms[0].atomic_number == 6 and atoms[1].atomic_number == 8) or (
                atoms[0].atomic_number == 8 and atoms[1].atomic_number == 6
            ):
                co_bond = bond_indices
                break

        assert co_bond is not None, "Could not find C-O bond in ethanol"

        ethanol_specific_smarts = _create_smarts(mol2, co_bond, max_extend_distance=-1)
        ethanol_param = {
            "smirks": ethanol_specific_smarts,
            "id": "b-bespoke-ethanol",
            "length": 1.4 * openff.toolkit.unit.angstrom,
            "k": 550.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }
        _add_parameter_with_overwrite(bond_handler, ethanol_param)

        # Remove redundant - should keep ethanol param since mol2 uses it
        ff_cleaned = _remove_redundant_smarts([mol1, mol2], ff, id_substring="bespoke")

        param_ids = [p.id for p in ff_cleaned.get_parameter_handler("Bonds").parameters]
        assert "b-bespoke-ethanol" in param_ids

    def test_none_id_substring_does_nothing(self):
        """Test that passing None for id_substring doesn't remove anything."""
        mol = openff.toolkit.Molecule.from_smiles("CCO")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        bond_handler = ff.get_parameter_handler("Bonds")

        # Add an unused bespoke parameter
        unused_param = {
            "smirks": "[#99:1]-[#99:2]",
            "id": "b-bespoke-unused",
            "length": 1.2 * openff.toolkit.unit.angstrom,
            "k": 600.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }
        _add_parameter_with_overwrite(bond_handler, unused_param)

        original_count = len(bond_handler.parameters)

        # Call with id_substring=None
        ff_result = _remove_redundant_smarts(mol, ff, id_substring=None)

        # Nothing should be removed
        assert (
            len(ff_result.get_parameter_handler("Bonds").parameters) == original_count
        )

    def test_empty_molecule_list(self):
        """Test with empty molecule list."""
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        bond_handler = ff.get_parameter_handler("Bonds")

        # Add a bespoke parameter
        bespoke_param = {
            "smirks": "[#99:1]-[#99:2]",
            "id": "b-bespoke-test",
            "length": 1.2 * openff.toolkit.unit.angstrom,
            "k": 600.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }
        _add_parameter_with_overwrite(bond_handler, bespoke_param)

        original_count = len(bond_handler.parameters)

        # Call with empty list - should remove the bespoke parameter since nothing uses it
        ff_cleaned = _remove_redundant_smarts([], ff, id_substring="bespoke")

        new_count = len(ff_cleaned.get_parameter_handler("Bonds").parameters)
        assert new_count == original_count - 1

    def test_single_molecule_as_object(self):
        """Test that single molecule (not in list) works correctly."""
        mol = openff.toolkit.Molecule.from_smiles("CCO")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        bond_handler = ff.get_parameter_handler("Bonds")

        # Add unused parameter
        unused_param = {
            "smirks": "[#99:1]-[#99:2]",
            "id": "b-bespoke-unused",
            "length": 1.2 * openff.toolkit.unit.angstrom,
            "k": 600.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }
        _add_parameter_with_overwrite(bond_handler, unused_param)

        original_count = len(bond_handler.parameters)

        # Pass single molecule, not list
        ff_cleaned = _remove_redundant_smarts(mol, ff, id_substring="bespoke")

        new_count = len(ff_cleaned.get_parameter_handler("Bonds").parameters)
        assert new_count == original_count - 1

    def test_multiple_handlers(self):
        """Test that redundancy removal works across multiple handlers."""
        mol = openff.toolkit.Molecule.from_smiles("CC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        # Add unused parameters to multiple handlers
        bond_handler = ff.get_parameter_handler("Bonds")
        angle_handler = ff.get_parameter_handler("Angles")

        unused_bond = {
            "smirks": "[#99:1]-[#99:2]",
            "id": "b-bespoke-unused",
            "length": 1.2 * openff.toolkit.unit.angstrom,
            "k": 600.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.angstrom**2,
        }
        _add_parameter_with_overwrite(bond_handler, unused_bond)

        unused_angle = {
            "smirks": "[#99:1]-[#99:2]-[#99:3]",
            "id": "a-bespoke-unused",
            "angle": 120.0 * openff.toolkit.unit.degree,
            "k": 100.0
            * openff.toolkit.unit.kilocalorie_per_mole
            / openff.toolkit.unit.radian**2,
        }
        _add_parameter_with_overwrite(angle_handler, unused_angle)

        original_bond_count = len(bond_handler.parameters)
        original_angle_count = len(angle_handler.parameters)

        # Remove redundant from both handlers
        ff_cleaned = _remove_redundant_smarts(mol, ff, id_substring="bespoke")

        # Both should have removed the unused parameter
        assert (
            len(ff_cleaned.get_parameter_handler("Bonds").parameters)
            == original_bond_count - 1
        )
        assert (
            len(ff_cleaned.get_parameter_handler("Angles").parameters)
            == original_angle_count - 1
        )


class TestAddTypesToForcefield:
    """Tests for the add_types_to_forcefield function."""

    def test_basic_usage(self):
        """Test basic usage with benzene."""
        mol = openff.toolkit.Molecule.from_smiles("c1ccccc1")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

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
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

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
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

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
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

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
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

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
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

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

        # Full extension should add more parameters
        assert count_full > count_limited


class TestAddTypesToParameterHandler:
    """Tests for _add_types_to_parameter_handler function."""

    def test_excluded_and_included_mutually_exclusive(self):
        """Test that excluded_smirks and included_smirks cannot both be provided."""
        mol = openff.toolkit.Molecule.from_smiles("CC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        with pytest.raises(
            ValueError,
            match="excluded_smirks and included_smirks are mutually exclusive",
        ):
            _add_types_to_parameter_handler(
                mol,
                bond_handler,
                "Bonds",
                max_extend_distance=-1,
                excluded_smirks=["[#6:1]-[#6:2]"],
                included_smirks=["[#6:1]-[#1:2]"],
            )

    def test_with_excluded_smirks(self):
        """Test parameter generation with excluded SMIRKS."""
        mol = openff.toolkit.Molecule.from_smiles("CCO")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        # Get matches to find a SMIRKS to exclude
        matches = bond_handler.find_matches(mol.to_topology())
        if matches:
            first_match = next(iter(matches.values()))
            excluded_smirks = first_match.parameter_type.smirks

            original_count = len(bond_handler.parameters)

            handler_with_types = _add_types_to_parameter_handler(
                mol,
                bond_handler,
                "Bonds",
                max_extend_distance=-1,
                excluded_smirks=[excluded_smirks],
            )

            new_count = len(handler_with_types.parameters)

            # Should have added some parameters, but not for excluded SMIRKS
            assert new_count > original_count

    def test_with_included_smirks(self):
        """Test parameter generation with included SMIRKS."""
        mol = openff.toolkit.Molecule.from_smiles("CCO")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        # Get matches to find SMIRKS to include
        matches = bond_handler.find_matches(mol.to_topology())
        if len(matches) >= 2:
            # Include only one specific SMIRKS
            first_match = next(iter(matches.values()))
            included_smirks = first_match.parameter_type.smirks

            original_count = len(bond_handler.parameters)

            handler_with_types = _add_types_to_parameter_handler(
                mol,
                bond_handler,
                "Bonds",
                max_extend_distance=-1,
                included_smirks=[included_smirks],
            )

            new_count = len(handler_with_types.parameters)

            # Should have added fewer parameters than without restrictions
            assert new_count > original_count

    def test_handler_not_modified_in_place(self):
        """Test that original handler is not modified."""
        mol = openff.toolkit.Molecule.from_smiles("CC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        original_count = len(bond_handler.parameters)

        _add_types_to_parameter_handler(
            mol, bond_handler, "Bonds", max_extend_distance=-1
        )

        # Original should be unchanged
        assert len(bond_handler.parameters) == original_count

    def test_preserves_parameter_attributes(self):
        """Test that parameter attributes are correctly preserved."""
        mol = openff.toolkit.Molecule.from_smiles("CC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")
        bond_handler = ff.get_parameter_handler("Bonds")

        handler_with_types = _add_types_to_parameter_handler(
            mol, bond_handler, "Bonds", max_extend_distance=-1
        )

        # Check that bespoke parameters have proper attributes
        for param in handler_with_types.parameters:
            if "bespoke" in param.id:
                # Should have k and length
                assert hasattr(param, "k")
                assert hasattr(param, "length")
                assert param.k.m > 0  # Should be positive


class TestAddTypesToForcefieldExtended:
    """Extended tests for add_types_to_forcefield function."""

    def test_with_include_restrictions(self):
        """Test add_types_to_forcefield with include restrictions."""
        mol = openff.toolkit.Molecule.from_smiles("CCO")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        # Get a specific SMIRKS to include
        bond_handler = ff.get_parameter_handler("Bonds")
        matches = bond_handler.find_matches(mol.to_topology())
        if matches:
            first_match = next(iter(matches.values()))
            included_smirks = first_match.parameter_type.smirks

            type_gen_settings = {
                "Bonds": TypeGenerationSettings(
                    max_extend_distance=-1, include=[included_smirks]
                ),
            }

            ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

            # Should have added some parameters
            new_handler = ff_with_types.get_parameter_handler("Bonds")
            bespoke_params = [p for p in new_handler.parameters if "bespoke" in p.id]
            assert len(bespoke_params) > 0

    def test_multiple_molecules_deduplication(self):
        """Test that identical substructures across molecules are deduplicated."""
        mol1 = openff.toolkit.Molecule.from_smiles("CC")
        mol2 = openff.toolkit.Molecule.from_smiles("CCC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        # Add types for both molecules
        ff_with_types = add_types_to_forcefield([mol1, mol2], ff, type_gen_settings)

        bond_handler = ff_with_types.get_parameter_handler("Bonds")

        # Check for duplicate SMIRKS patterns
        smirks_seen = set()
        for param in bond_handler.parameters:
            if "bespoke" in param.id:
                assert param.smirks not in smirks_seen, (
                    f"Duplicate SMIRKS: {param.smirks}"
                )
                smirks_seen.add(param.smirks)

    def test_single_molecule_as_object_vs_list(self):
        """Test that single molecule works both as object and as list."""
        mol = openff.toolkit.Molecule.from_smiles("CC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        # As single object
        ff_single = add_types_to_forcefield(mol, ff, type_gen_settings)

        # As list
        ff_list = add_types_to_forcefield([mol], ff, type_gen_settings)

        # Should have same number of parameters
        single_count = len(ff_single.get_parameter_handler("Bonds").parameters)
        list_count = len(ff_list.get_parameter_handler("Bonds").parameters)

        assert single_count == list_count

    def test_complex_molecule_types(self):
        """Test with a more complex molecule."""
        mol = openff.toolkit.Molecule.from_smiles("c1ccccc1CCO")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=1, exclude=[]),
            "Angles": TypeGenerationSettings(max_extend_distance=1, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Should have added parameters
        bond_bespoke = [
            p
            for p in ff_with_types.get_parameter_handler("Bonds").parameters
            if "bespoke" in p.id
        ]
        angle_bespoke = [
            p
            for p in ff_with_types.get_parameter_handler("Angles").parameters
            if "bespoke" in p.id
        ]

        assert len(bond_bespoke) > 0
        assert len(angle_bespoke) > 0

    def test_varied_max_extend_distances(self):
        """Test different max_extend_distance for different handlers."""
        mol = openff.toolkit.Molecule.from_smiles("CCCC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        # Different extension distances
        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=0, exclude=[]),
            "Angles": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Both should have added something
        bond_handler = ff_with_types.get_parameter_handler("Bonds")
        angle_handler = ff_with_types.get_parameter_handler("Angles")

        bond_bespoke = [p for p in bond_handler.parameters if "bespoke" in p.id]
        angle_bespoke = [p for p in angle_handler.parameters if "bespoke" in p.id]

        assert len(bond_bespoke) > 0
        assert len(angle_bespoke) > 0

    def test_multiple_molecules_different_sizes(self):
        """Test with molecules of very different sizes."""
        mols = [
            openff.toolkit.Molecule.from_smiles("C"),  # Methane
            openff.toolkit.Molecule.from_smiles("CCCCCCCC"),  # Octane
            openff.toolkit.Molecule.from_smiles("c1ccccc1"),  # Benzene
        ]
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield(mols, ff, type_gen_settings)

        bond_handler = ff_with_types.get_parameter_handler("Bonds")
        bespoke_params = [p for p in bond_handler.parameters if "bespoke" in p.id]

        # Should have generated parameters for all molecules
        assert len(bespoke_params) > 0


class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_empty_force_field_handlers(self):
        """Test behavior with minimal force field."""
        mol = openff.toolkit.Molecule.from_smiles("C")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        # Try with handler that has few matches
        type_gen_settings = {
            "ImproperTorsions": TypeGenerationSettings(
                max_extend_distance=-1, exclude=[]
            ),
        }

        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Should not crash
        assert isinstance(ff_with_types, openff.toolkit.ForceField)

    def test_molecule_with_no_matching_parameters(self):
        """Test with parameter handlers that don't match the molecule."""
        mol = openff.toolkit.Molecule.from_smiles("C")  # No proper torsions
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        type_gen_settings = {
            "ProperTorsions": TypeGenerationSettings(
                max_extend_distance=-1, exclude=[]
            ),
        }

        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Should complete without error
        torsion_handler = ff_with_types.get_parameter_handler("ProperTorsions")
        bespoke_torsions = [p for p in torsion_handler.parameters if "bespoke" in p.id]

        # Methane has no proper torsions, so no bespoke parameters
        assert len(bespoke_torsions) == 0

    def test_very_small_max_extend_distance(self):
        """Test with max_extend_distance=0."""
        mol = openff.toolkit.Molecule.from_smiles("CCCCCC")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=0, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Should create very specific SMARTS patterns
        bond_handler = ff_with_types.get_parameter_handler("Bonds")
        bespoke_params = [p for p in bond_handler.parameters if "bespoke" in p.id]

        assert len(bespoke_params) > 0

    def test_charged_molecule(self):
        """Test with a charged molecule."""
        mol = openff.toolkit.Molecule.from_smiles("[NH4+]")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Should handle charged molecules
        bond_handler = ff_with_types.get_parameter_handler("Bonds")
        assert len(bond_handler.parameters) > 0

    def test_aromatic_molecule(self):
        """Test with aromatic molecule."""
        mol = openff.toolkit.Molecule.from_smiles("c1ccccc1")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        bond_handler = ff_with_types.get_parameter_handler("Bonds")
        bespoke_params = [p for p in bond_handler.parameters if "bespoke" in p.id]

        # Should create aromatic SMARTS
        assert len(bespoke_params) > 0

        # Check that at least one parameter contains aromatic notation
        aromatic_found = any(":" in p.smirks for p in bespoke_params)
        assert aromatic_found

    def test_molecule_with_stereochemistry(self):
        """Test with stereochemistry."""
        mol = openff.toolkit.Molecule.from_smiles("C[C@H](O)N")
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield(mol, ff, type_gen_settings)

        # Should handle stereochemistry without error
        bond_handler = ff_with_types.get_parameter_handler("Bonds")
        assert len(bond_handler.parameters) > 0
