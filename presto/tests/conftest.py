"""Shared test fixtures used across the test suite."""

import importlib
from pathlib import Path

import pytest
import smee
from descent.train import ParameterConfig, Trainable
from openff.toolkit import ForceField, Molecule

MODEL_REQUIRED_PACKAGE = {
    "aceff-2.0": "torchmdnet",
    "mace-off23-small": "mace",
    "mace-off23-medium": "mace",
    "mace-off23-large": "mace",
    "mace-omol-0-extra-large": "mace",
    "egret-1": "mace",
    "aimnet2": "aimnet",
    "orb-v3-conservative-omol": "orb_models",
}


def skip_if_model_unavailable(model_name: str) -> None:
    """Skip the current test if the package required by model_name is not installed."""
    pkg = MODEL_REQUIRED_PACKAGE.get(model_name)
    if pkg is not None and importlib.util.find_spec(pkg) is None:
        pytest.skip(f"{pkg} not installed")


@pytest.fixture
def write_multiconformer_sdf():
    """Return a helper that writes molecules' conformers as separate SDF records.

    ``Molecule.to_file(..., "SDF")`` only writes a single conformer, so tests that need a
    genuine multi-record SDF (the shape a user supplies as starting conformers) go via
    RDKit. The returned helper accepts a single ``Molecule`` or an iterable of them and
    writes every conformer of each as its own record to ``path``.
    """
    from rdkit import Chem

    def _write(molecules, path) -> None:
        if isinstance(molecules, Molecule):
            molecules = [molecules]
        with Chem.SDWriter(str(path)) as writer:
            for molecule in molecules:
                rdkit_molecule = molecule.to_rdkit()
                for conformer_id in range(rdkit_molecule.GetNumConformers()):
                    writer.write(rdkit_molecule, confId=conformer_id)

    return _write


# From Simon Boothroyd
@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> Path:
    """Change the working directory to a temporary path for the duration of the test."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def jnk1_lig_smiles():
    """SMILES string for a JNK1 ligand used in integration tests."""
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
    return ForceField("openff_unconstrained-2.3.0.offxml", load_plugins=True)


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
