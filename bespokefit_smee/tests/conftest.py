from pathlib import Path

import pytest
import smee
from descent.train import ParameterConfig, Trainable
from openff.toolkit import ForceField, Molecule


# From Simon Boothroyd
@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def jnk1_lig_smiles():
    return "C(C(Oc1nc(c(c(N([H])[H])c1C#N)[H])N(C(=O)C(c1c(c(C([H])([H])[H])c(c(c1[H])[H])[H])[H])([H])[H])[H])([H])[H])([H])([H])[H]"


@pytest.fixture
def ethanol_molecule():
    """Ethanol molecule for testing."""
    return Molecule.from_smiles("CCO")


@pytest.fixture
def ethanol_with_conformers():
    """Ethanol molecule with conformers for testing."""
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=2)
    return mol


@pytest.fixture
def simple_force_field():
    """Simple OpenFF force field for testing."""
    return ForceField("openff_unconstrained-2.3.0.offxml")


@pytest.fixture
def ethanol_tensor_topology_and_ff(ethanol_molecule, simple_force_field):
    """Ethanol tensor topology and force field for testing."""
    import openff.interchange

    interchange = openff.interchange.Interchange.from_smirnoff(
        simple_force_field, ethanol_molecule.to_topology()
    )
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)
    return tensor_top, tensor_ff


@pytest.fixture
def simple_trainable(ethanol_tensor_topology_and_ff):
    """Simple trainable for testing."""
    _, tensor_ff = ethanol_tensor_topology_and_ff

    parameter_configs = {
        "Bonds": ParameterConfig(
            cols=["k", "length"],
            scales={"k": 1.0, "length": 1.0},
        ),
    }

    return Trainable(tensor_ff, parameter_configs, {})
