"""Unit tests for the parameterizer module."""

import openff.toolkit
import pytest
import smee
import smee.converters
import torch
from openff.toolkit import ForceField
from pint import Quantity

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
        msm_settings=None,
        initial_force_field="openff_unconstrained-2.3.0-rc1.offxml",
        expand_torsions=False,
    )
    mol, _, _, tff, _ = parameterise(settings=settings, device="cpu")

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
    "smiles, excluded_smirks",
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
    linearise_harmonics: bool, smiles: str, excluded_smirks: list[str]
):
    """Check that we get the same energy after converting back and forth
    between ForceField and TensorForceField objects."""

    base_ff = ForceField("openff_unconstrained-2.3.0-rc1.offxml")
    settings = ParameterisationSettings(
        linearise_harmonics=linearise_harmonics,
        smiles=smiles,
        msm_settings=None,
        initial_force_field="openff_unconstrained-2.3.0-rc1.offxml",
        expand_torsions=True,
        excluded_smirks=excluded_smirks,
    )
    mol, _, tensor_top, tensor_ff, _ = parameterise(settings=settings, device="cpu")
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
