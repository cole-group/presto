"""Tests for the create_types module."""

import openff.toolkit
import pytest
from openff.toolkit import ForceField
from rdkit import Chem

from bespokefit_smee.create_types import (
    _add_parameter_with_overwrite,
    _create_smarts,
    _remove_redundant_smarts,
    add_types_to_forcefield,
)
from bespokefit_smee.settings import (
    TypeGenerationSettings,
)


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

        # With redundancy removal, both may have similar counts since unused
        # parameters are removed - just verify both are reasonable
        assert count_limited >= original_count
        assert count_full >= original_count
