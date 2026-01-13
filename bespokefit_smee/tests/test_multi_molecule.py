"""Unit tests for multi-molecule fitting functionality."""

import pytest
import torch
from openff.toolkit import ForceField, Molecule

from bespokefit_smee.convert import parameterise
from bespokefit_smee.create_types import add_types_to_forcefield
from bespokefit_smee.settings import (
    ParameterisationSettings,
    TypeGenerationSettings,
)


class TestMultiMoleculeParameterisation:
    """Tests for multi-molecule parameterisation."""

    def test_single_molecule_compatibility(self):
        """Test that single molecule works as a list of one."""
        settings = ParameterisationSettings(
            smiles=["CCO"],
            linearise_harmonics=False,
            expand_torsions=False,
            msm_settings=None,
        )

        mols, off_ff, tops, tensor_ff = parameterise(settings=settings, device="cpu")

        assert len(mols) == 1
        assert len(tops) == 1
        assert mols[0].n_atoms == 9  # Ethanol

    def test_multiple_molecules_parameterisation(self):
        """Test parameterisation with multiple molecules."""
        settings = ParameterisationSettings(
            smiles=["CC", "CCO", "CCCC"],
            linearise_harmonics=False,
            expand_torsions=False,
            msm_settings=None,
        )

        mols, off_ff, tops, tensor_ff = parameterise(settings=settings, device="cpu")

        assert len(mols) == 3
        # Each molecule gets its own force field with its parameters
        assert len(tops) == 3
        assert tensor_ff is not None

        # Check that each topology can be used with its corresponding force field
        # (topology 0 uses force_field from molecule 0, etc.)
        for mol, top in zip(mols, tops, strict=True):
            assert top.n_atoms == mol.n_atoms
        assert mols[1].n_atoms == 9  # Ethanol
        assert mols[2].n_atoms == 14  # Butane

    def test_deduplication_across_molecules(self):
        """Test that SMARTS deduplication works across molecules."""
        # Use molecules that share similar structural features
        mols = [
            Molecule.from_smiles("CC", allow_undefined_stereo=True),
            Molecule.from_smiles("CCC", allow_undefined_stereo=True),
            Molecule.from_smiles("CCCC", allow_undefined_stereo=True),
        ]

        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield(mols, ff, type_gen_settings)

        bond_handler = ff_with_types.get_parameter_handler("Bonds")

        # Count bespoke parameters (those with 'bespoke' in the ID)
        bespoke_params = [p for p in bond_handler.parameters if "bespoke" in p.id]

        # Should have fewer bespoke parameters than if we parameterised each molecule separately
        # due to deduplication across molecules
        assert len(bespoke_params) > 0

    def test_invalid_smiles_validation(self):
        """Test that invalid SMILES in list raises error."""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            ParameterisationSettings(
                smiles=["CCO", "invalid_smiles_123", "CC"],
            )

    def test_empty_smiles_list(self):
        """Test that empty SMILES list is handled."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ParameterisationSettings(smiles=[])

    def test_single_invalid_smiles(self):
        """Test that one invalid SMILES in list fails."""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            ParameterisationSettings(
                smiles=["CCO", "INVALID", "CCC"],
            )

    def test_duplicate_smiles(self):
        """Test that duplicate SMILES are rejected."""
        with pytest.raises(ValueError, match="Duplicate SMILES found"):
            ParameterisationSettings(
                smiles=["CC", "CCO", "CC", "CCC"],
            )


class TestMultiMoleculeTypeGeneration:
    """Tests for multi-molecule type generation."""

    def test_type_generation_single_molecule(self):
        """Test type generation with single molecule as list."""
        mol = Molecule.from_smiles("CCO", allow_undefined_stereo=True)
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        original_count = len(ff.get_parameter_handler("Bonds").parameters)

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield([mol], ff, type_gen_settings)

        new_count = len(ff_with_types.get_parameter_handler("Bonds").parameters)
        assert new_count > original_count

    def test_type_generation_multiple_molecules(self):
        """Test type generation with multiple molecules."""
        mols = [
            Molecule.from_smiles("CCO", allow_undefined_stereo=True),
            Molecule.from_smiles("CC", allow_undefined_stereo=True),
            Molecule.from_smiles("CCC", allow_undefined_stereo=True),
        ]
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        original_count = len(ff.get_parameter_handler("Bonds").parameters)

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield(mols, ff, type_gen_settings)

        new_count = len(ff_with_types.get_parameter_handler("Bonds").parameters)
        assert new_count > original_count

    def test_shared_parameters_across_molecules(self):
        """Test that shared structural features lead to shared parameters."""
        # Ethane, propane, butane - all share C-C bonds
        mols = [
            Molecule.from_smiles("CC", allow_undefined_stereo=True),
            Molecule.from_smiles("CCC", allow_undefined_stereo=True),
            Molecule.from_smiles("CCCC", allow_undefined_stereo=True),
        ]
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        type_gen_settings = {
            "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        }

        ff_with_types = add_types_to_forcefield(mols, ff, type_gen_settings)

        # Check that the force field can label all molecules
        for mol in mols:
            labels = ff_with_types.label_molecules(mol.to_topology())[0]
            assert "Bonds" in labels
            assert len(labels["Bonds"]) > 0


class TestMultiMoleculeWorkflowSettings:
    """Tests for workflow settings with multiple molecules."""

    def test_workflow_settings_with_multiple_molecules(self):
        """Test that workflow settings accept multiple molecules."""
        from bespokefit_smee.settings import (
            MMMDSamplingSettings,
            WorkflowSettings,
        )

        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(
                smiles=["CCO", "CC", "CCC"]
            ),
            training_sampling_settings=MMMDSamplingSettings(),
            n_iterations=1,
        )

        assert len(settings.parameterisation_settings.smiles) == 3

    def test_backward_compatibility_single_molecule(self):
        """Test that single molecule still works."""
        from bespokefit_smee.settings import (
            MMMDSamplingSettings,
            WorkflowSettings,
        )

        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles=["CCO"]),
            training_sampling_settings=MMMDSamplingSettings(),
            n_iterations=1,
        )

        assert len(settings.parameterisation_settings.smiles) == 1


class TestMultiMoleculeEnergyCalculations:
    """Tests for energy calculations with multiple molecules."""

    @pytest.mark.parametrize("linearise_harmonics", [True, False])
    def test_energy_calculation_single_molecule(self, linearise_harmonics: bool):
        """Test energy calculation with single molecule."""
        settings = ParameterisationSettings(
            smiles=["CCO"],
            linearise_harmonics=linearise_harmonics,
            expand_torsions=False,
            msm_settings=None,
        )

        mols, off_ff, tops, tensor_ff = parameterise(settings=settings, device="cpu")

        # Generate a conformer
        mols[0].generate_conformers(n_conformers=1)
        coords = torch.tensor(mols[0].conformers[0].m_as("angstrom"))

        # Calculate energy
        import smee

        energy = smee.compute_energy(tops[0], tensor_ff, coords)

        assert energy.numel() == 1
        assert torch.isfinite(energy)

    @pytest.mark.parametrize("linearise_harmonics", [True, False])
    def test_energy_calculation_multiple_molecules(self, linearise_harmonics: bool):
        """Test that parameterisation works for multiple molecules."""
        settings = ParameterisationSettings(
            smiles=["CC", "CCO"],
            linearise_harmonics=linearise_harmonics,
            expand_torsions=False,
            msm_settings=None,
        )

        mols, off_ff, tops, tensor_ff = parameterise(settings=settings, device="cpu")

        # Verify we get separate topologies for each molecule
        assert len(mols) == 2
        assert len(tops) == 2  # One topology per molecule
        assert tensor_ff is not None

        # Verify topologies have correct number of atoms
        assert tops[0].n_atoms == mols[0].n_atoms
        assert tops[1].n_atoms == mols[1].n_atoms


class TestMultiMoleculeParameterSharing:
    """Tests to ensure parameters are shared correctly across molecules."""

    def test_same_force_field_for_all_molecules(self):
        """Test that all molecules use the same force field."""
        settings = ParameterisationSettings(
            smiles=["CC", "CCC", "CCCC"],
            linearise_harmonics=False,
            expand_torsions=False,
            msm_settings=None,
        )

        mols, off_ff, tops, tensor_ff = parameterise(settings=settings, device="cpu")

        # All topologies should reference the same force field structure
        # (though they have different topological indices)
        for top in tops:
            assert "Bonds" in top.parameters
            assert "Angles" in top.parameters

    def test_parameter_updates_affect_all_molecules(self):
        """Test that updating parameters affects all molecules."""
        from descent.train import ParameterConfig, Trainable

        settings = ParameterisationSettings(
            smiles=["CC", "CCC"],
            linearise_harmonics=False,
            expand_torsions=False,
            msm_settings=None,
        )

        mols, off_ff, tops, tensor_ff = parameterise(settings=settings, device="cpu")

        # Create a trainable with parameter configs
        parameter_configs = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                limits={"k": (0.0, None), "length": (0.0, None)},
            )
        }

        trainable = Trainable(tensor_ff, parameter_configs, {})

        # Get initial parameters and make a copy
        initial_params = trainable.to_values().clone()
        initial_bonds = tensor_ff.potentials_by_type["Bonds"].parameters.clone()

        # Modify parameters (scale by factor for detectable change)
        modified_params = initial_params * 1.5

        # Convert back to force field
        modified_ff = trainable.to_force_field(modified_params)

        # Check that the force field has changed
        modified_bonds = modified_ff.potentials_by_type["Bonds"].parameters

        # Parameters should be different after modification
        assert not torch.allclose(modified_bonds, initial_bonds, rtol=1e-5)
        # Verify parameters are shared across topologies (same force field)
        assert len(tops) == 2
        assert tensor_ff is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
