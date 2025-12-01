"""Unit tests for the parameterizer module."""

import descent.targets.energy
import numpy as np
import openff.interchange
import openff.toolkit
import pytest
import smee
import smee.converters
import torch
from openff.interchange.drivers.openmm import get_openmm_energies
from openff.toolkit import ForceField
from pint import Quantity

from ...loss import predict
from ...parameterise import (
    convert_to_smirnoff,
    parameterise,
)
from ...settings import ParameterisationSettings


@pytest.mark.parametrize("linearise_harmonics", [True, False])
@pytest.mark.parametrize(
    "smiles",
    [
        "CC",
        "ClC=O",
        "OCCO",
        "CC(=O)Nc1ccc(cc1)O",
        "C(C(Oc1nc(c(c(N([H])[H])c1C#N)[H])N(C(=O)C(c1c(c(C([H])([H])[H])c(c(c1[H])[H])[H])[H])([H])[H])[H])([H])[H])([H])([H])[H]",
    ],
)
def test_params_equivalent(linearise_harmonics: bool, smiles: str):
    """
    Check that we can convert a general force field to and from a
    molecule-specific TensorForceField while still assigning the same\
    parameters.
    """
    # base_ff = ForceField("openff_unconstrained-2.2.1.offxml")
    base_ff = ForceField("openff_unconstrained-2.3.0-rc1.offxml")
    settings = ParameterisationSettings(
        linearise_harmonics=linearise_harmonics,
        smiles=smiles,
        initial_force_field="openff_unconstrained-2.3.0-rc1.offxml",
        expand_torsions=False,
    )
    mol, _, _, tff = parameterise(settings=settings, device="cpu")

    # Convert the TensorForceField back to a SMIRNOFF force field
    recreated_ff = convert_to_smirnoff(tff, base=base_ff)

    # Label the molecule with both ffs
    base_labels = base_ff.label_molecules(mol.to_topology())[0]
    recreated_labels = recreated_ff.label_molecules(mol.to_topology())[0]

    # Check that everything matches, other than the SMIRKS
    assert base_labels.keys() == recreated_labels.keys()

    for param_type in base_labels.keys():
        base_params = base_labels[param_type]
        recreated_params = recreated_labels[param_type]

        # Continue if both are empty (e.g. constraints)
        if not base_params and not recreated_params:
            continue

        for param_key in base_params.keys():
            base_param_dict = base_params[param_key].to_dict()
            recreated_param_dict = recreated_params[param_key].to_dict()

            # Make sure we haven't lost any keys
            assert set(base_param_dict.keys()).issubset(recreated_param_dict.keys()), (
                f"Parameter {param_key} dicts do not have the same keys: "
                f"Base: {base_param_dict.keys()} "
                f"Recreated: {recreated_param_dict.keys()}"
            )

            # Filter out attributes that are not relevant for comparison
            unwanted_keys = {
                "smirks",  # SMIRKS get modified during conversion
                "id",  # IDs will differ
            }

            for attr in base_param_dict.keys():
                if attr in unwanted_keys:
                    continue

                # If this is a pint Quantity, convert to base units for comparison
                if isinstance(base_param_dict[attr], Quantity):
                    assert base_param_dict[
                        attr
                    ].to_base_units().magnitude == pytest.approx(
                        recreated_param_dict[attr].to_base_units().magnitude, rel=1e-5
                    ), (
                        f"Parameter {param_key} attribute {attr} does not match: "
                        f"Base: {base_param_dict[attr]} "
                        f"Recreated: {recreated_param_dict[attr]}"
                    )
                else:
                    assert base_param_dict[attr] == recreated_param_dict[attr], (
                        f"Parameter {param_key} attribute {attr} does not match: "
                        f"Base: {base_param_dict[attr]} "
                        f"Recreated: {recreated_param_dict[attr]}"
                    )


@pytest.mark.parametrize("linearise_harmonics", [True, False])
@pytest.mark.parametrize(
    "smiles, proper_torsion_excludes",
    [
        ("CC", []),
        ("ClC=O", []),
        ("OCCO", []),
        ("CC(=O)Nc1ccc(cc1)O", []),
        (
            "C(C(Oc1nc(c(c(N([H])[H])c1C#N)[H])N(C(=O)C(c1c(c(C([H])([H])[H])c(c(c1[H])[H])[H])[H])([H])[H])[H])([H])[H])([H])([H])[H]",
            [],
        ),
        (
            "C(C(Oc1nc(c(c(N([H])[H])c1C#N)[H])N(C(=O)C(c1c(c(C([H])([H])[H])c(c(c1[H])[H])[H])[H])([H])[H])[H])([H])[H])([H])([H])[H]",
            [
                "[*:1]-[*:2]#[*:3]-[*:4]",
                "[*:1]~[*:2]-[*:3]#[*:4]",
                "[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]",
            ],
        ),
    ],
)
# Adapted from https://github.com/SimonBoothroyd/befit/blob/1c4e5d1a5d6af6fe2386d9202606c6960a914689/befit/tests/test_ff.py#L17
def test_energies_equivalent(
    linearise_harmonics: bool, smiles: str, proper_torsion_excludes: list[str]
):
    """Check that we get the same energy after converting back and forth
    between ForceField and TensorForceField objects."""

    from ...settings import TypeGenerationSettings
    from ...utils.typing import NonLinearValenceType

    base_ff = ForceField("openff_unconstrained-2.3.0-rc1.offxml")

    # Build type_generation_settings with proper_torsion_excludes
    type_gen_settings: dict[NonLinearValenceType, TypeGenerationSettings] = {
        "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        "Angles": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
        "ProperTorsions": TypeGenerationSettings(
            max_extend_distance=-1,
            exclude=proper_torsion_excludes,
        ),
        "ImproperTorsions": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
    }

    settings = ParameterisationSettings(
        linearise_harmonics=linearise_harmonics,
        smiles=smiles,
        initial_force_field="openff_unconstrained-2.3.0-rc1.offxml",
        expand_torsions=True,
        type_generation_settings=type_gen_settings,
    )
    mol, _, tensor_top, tensor_ff = parameterise(settings=settings, device="cpu")
    mol.generate_conformers(n_conformers=1)

    coords = torch.tensor(mol.conformers[0].m_as("angstrom"))
    coords += torch.randn(len(coords) * 3).reshape(-1, 3) * 0.5

    energy_1 = smee.compute_energy(tensor_top, tensor_ff, coords)

    converted_back = convert_to_smirnoff(tensor_ff, base_ff)

    ff_2, [top_2] = smee.converters.convert_interchange(
        openff.interchange.Interchange.from_smirnoff(converted_back, mol.to_topology())
    )
    energy_2 = smee.compute_energy(top_2, ff_2, coords)

    assert torch.isclose(energy_1, energy_2)


@pytest.mark.parametrize("linearise_harmonics", [True, False])
@pytest.mark.parametrize(
    "smiles",
    [
        "CC",  # Ethane
        "CCO",  # Ethanol
        "ClC=O",  # Formyl chloride
        "OCCO",  # Ethylene glycol
        "CC(C)C",  # Isobutane
        "c1ccccc1",  # Benzene
        "CC(=O)O",  # Acetic acid
        "CC(=O)N",  # Acetamide
        "CC(=O)Nc1ccccc1",  # Acetanilide
        "CC(=O)Nc1ccc(cc1)O",  # Paracetamol
    ],
)
def test_openmm_smee_energy_equivalence(linearise_harmonics: bool, smiles: str):
    """Test that energies computed via OpenMM and SMEE are equivalent
    for the same force field, comparing the original OpenFF interchange
    with the converted SMEE TensorForceField."""

    settings = ParameterisationSettings(
        linearise_harmonics=linearise_harmonics,
        smiles=smiles,
        initial_force_field="openff_unconstrained-2.3.0-rc1.offxml",
        expand_torsions=False,
    )

    mol, off_ff, tensor_top, tensor_ff = parameterise(settings=settings, device="cpu")

    # Generate conformers for testing
    mol.generate_conformers(n_conformers=3)

    # Create OpenMM Interchange
    interchange = openff.interchange.Interchange.from_smirnoff(
        off_ff, mol.to_topology()
    )

    # Test each conformer
    for conf_idx, conformer in enumerate(mol.conformers):
        # Update positions in interchange
        interchange.positions = conformer

        # Get OpenMM energy
        openmm_result = get_openmm_energies(
            interchange, combine_nonbonded_forces=True, platform="Reference"
        )
        # Total energy in kcal/mol
        openmm_energy = openmm_result.total_energy.m_as("kilocalorie / mole")

        # Get SMEE energy (returns kcal/mol)
        coords = torch.tensor(conformer.m_as("angstrom"), dtype=torch.float32)
        smee_energy_tensor = smee.compute_energy(
            tensor_top, tensor_ff, coords.unsqueeze(0)
        )
        smee_energy = smee_energy_tensor.item()

        # Compare energies (allow 0.1 kcal/mol absolute tolerance)
        assert np.isclose(openmm_energy, smee_energy, rtol=1e-3, atol=0.1), (
            f"Energy mismatch for {smiles} conformer {conf_idx}: "
            f"OpenMM={openmm_energy:.4f} kcal/mol, "
            f"SMEE={smee_energy:.4f} kcal/mol"
        )


@pytest.mark.parametrize("linearise_harmonics", [True, False])
@pytest.mark.parametrize(
    "smiles",
    [
        "CC",  # Ethane
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(=O)Nc1ccccc1",  # Acetanilide
    ],
)
def test_openmm_smee_energy_with_predict(linearise_harmonics: bool, smiles: str):
    """Test that energies computed using the predict function match
    OpenMM energies for multiple conformers."""

    settings = ParameterisationSettings(
        linearise_harmonics=linearise_harmonics,
        smiles=smiles,
        initial_force_field="openff_unconstrained-2.3.0-rc1.offxml",
        expand_torsions=False,
    )

    mol, off_ff, tensor_top, tensor_ff = parameterise(settings=settings, device="cpu")

    # Generate conformers
    mol.generate_conformers(n_conformers=5)

    # Create dataset for predict function
    coords_list = [
        torch.tensor(conf.m_as("angstrom"), dtype=torch.float32)
        for conf in mol.conformers
    ]
    coords_tensor = torch.stack(coords_list)

    # Compute reference energies with OpenMM
    interchange = openff.interchange.Interchange.from_smirnoff(
        off_ff, mol.to_topology()
    )

    openmm_energies = []
    for conformer in mol.conformers:
        interchange.positions = conformer
        result = get_openmm_energies(
            interchange, combine_nonbonded_forces=True, platform="Reference"
        )
        # Convert to kcal/mol
        energy_kcal = result.total_energy.m_as("kilocalorie / mole")
        openmm_energies.append(energy_kcal)

    openmm_energies_tensor = torch.tensor(openmm_energies, dtype=torch.float32)

    # Create indexed SMILES for the dataset
    indexed_smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)

    # Create a dataset using the descent helper
    dataset = descent.targets.energy.create_dataset(
        [
            {
                "smiles": indexed_smiles,
                "coords": coords_tensor,
                "energy": openmm_energies_tensor,
                "forces": torch.zeros(
                    (len(mol.conformers), mol.n_atoms, 3), dtype=torch.float32
                ),
            }
        ]
    )

    # Use predict function to get SMEE energies
    energy_ref, energy_pred, _, _ = predict(
        dataset,
        tensor_ff,
        {indexed_smiles: tensor_top},
        reference="mean",
        normalize=False,
        device_type="cpu",
    )

    # The predict function returns relative energies, so we need to compare
    # the differences
    openmm_relative = openmm_energies_tensor - openmm_energies_tensor.mean()

    # Allow for small numerical differences (0.1 kcal/mol absolute tolerance)
    assert torch.allclose(energy_pred, openmm_relative, rtol=1e-3, atol=0.1), (
        f"Energy predictions differ for {smiles}: pred={energy_pred}, ref={openmm_relative}"
    )
