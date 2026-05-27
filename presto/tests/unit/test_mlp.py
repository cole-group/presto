"""Unit tests for mlp module."""

import math
from typing import get_args
from unittest.mock import MagicMock, patch

import numpy as np
import openmm
import pytest
import torch
from openff.toolkit import Molecule
from openff.units import unit as off_unit
from openmmml import MLPotential

from presto.mlp import KnownModels, _cache, get_ml_omm_system, get_mlp
from presto.settings import MLPSettings

EXPECTED_MODEL_ENERGIES = {
    # Note that AceFF is currently wrong in OpenMM-ML https://github.com/openmm/openmm-ml/issues/137, but
    # this energy is the one after fixes and should be correct (currently failing).
    "aceff-2.0": -89.407890319824,
    "mace-off23-small": -963.051142471468,
    "mace-off23-medium": -963.073736333697,
    "mace-off23-large": -963.177635381956,
    "mace-omol-0-extra-large": -19.449931963902,
    "egret-1": -963.030539966774,
    "aimnet2": -200784.718554282794,
    "orb-v3-conservative-omol": -200671.118723808293,
}


class TestGetMlp:
    """Tests for get_mlp function."""

    @pytest.mark.parametrize(
        "model_name",
        [pytest.param(model, id=model) for model in get_args(KnownModels)],
    )
    def test_all_known_models_can_create_systems(self, model_name):
        """Test that known models can be loaded and create OpenMM systems."""
        _cache.clear()
        mol = Molecule.from_smiles("O")  # Water
        topology = mol.to_topology().to_openmm()

        potential = get_mlp(model_name)
        assert potential is not None
        assert isinstance(potential, MLPotential)

        system = potential.createSystem(topology)
        assert system.getNumParticles() == topology.getNumAtoms()
        assert system.getNumForces() > 0

    @pytest.mark.parametrize(
        "model_name",
        [pytest.param(model, id=model) for model in get_args(KnownModels)],
    )
    @pytest.mark.slow
    def test_all_known_models_can_calculate_energy(self, model_name):
        """Test that known models can calculate an energy for water."""
        import openmm

        _cache.clear()

        positions = (
            np.array(
                [
                    [-0.00081616, 0.36637843, -0.0],
                    [-0.8123162, -0.18348211, -0.0],
                    [0.81313236, -0.18289632, 0.0],
                ]
            )
            * off_unit.angstroms
        )

        mol = Molecule.from_smiles("O")
        mol.add_conformer(positions)
        topology = mol.to_topology().to_openmm()

        potential = get_mlp(model_name)
        system = potential.createSystem(topology)

        integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
        context = openmm.Context(system, integrator)
        context.setPositions(mol.conformers[0].to_openmm())

        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoule_per_mole
        )
        assert math.isfinite(energy)
        assert energy == pytest.approx(
            EXPECTED_MODEL_ENERGIES[model_name], rel=1e-6, abs=1e-3
        )

    def test_get_mlp_caches_results(self):
        """Test that repeated calls with the same arguments return the same object."""
        result1 = get_mlp("aimnet2")
        result2 = get_mlp("aimnet2")
        assert result1 is result2


class TestGetMlOmmSystem:
    """Tests for get_ml_omm_system function."""

    @pytest.fixture(autouse=True)
    def mock_get_mlp(self):
        """Mock get_mlp to avoid loading real models and pass isinstance checks."""
        with patch("presto.mlp.get_mlp") as mock:
            mock_potential = MagicMock()

            def create_mock_system(topology, **kwargs):
                system = openmm.System()
                for _ in range(topology.getNumAtoms()):
                    system.addParticle(1.0)
                return system

            mock_potential.createSystem.side_effect = create_mock_system
            mock.return_value = mock_potential
            yield mock

    def test_passes_potential_and_system_kwargs(self, mock_get_mlp):
        """Test constructor and createSystem kwargs are forwarded."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=1)

        get_ml_omm_system(
            mol,
            MLPSettings(
                ml_potential="custom-model",
                ml_potential_kwargs={"modelPath": "my.model"},
                ml_system_kwargs={"precision": "single"},
            ),
            torch.device("cpu"),
        )

        mock_get_mlp.assert_called_once_with("custom-model", modelPath="my.model")
        create_kwargs = mock_get_mlp.return_value.createSystem.call_args.kwargs
        assert create_kwargs["precision"] == "single"
        assert create_kwargs["device"] == "cpu"
        assert create_kwargs["charge"] == pytest.approx(0.0)  # CCO is neutral
        assert (
            "modelPath" not in create_kwargs
        )  # ml_potential_kwargs must not bleed through

    def test_ase_ml_system_kwargs_path(self, mock_get_mlp):
        """Test ASE calculator/info kwargs in ml_system_kwargs are passed through."""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=1)
        calculator = object()

        get_ml_omm_system(
            mol,
            MLPSettings(
                ml_potential="ase",
                ml_system_kwargs={
                    "calculator": calculator,
                    "info": {"foo": "bar", "charge": 1},
                },
            ),
            torch.device("cpu"),
        )

        mock_get_mlp.assert_called_once_with("ase")
        create_kwargs = mock_get_mlp.return_value.createSystem.call_args.kwargs
        assert create_kwargs["calculator"] is calculator
        assert create_kwargs["info"] == {"foo": "bar", "charge": 1}
        assert "charge" not in create_kwargs
        assert "device" not in create_kwargs

    def test_warns_charged_molecule_with_ase(self):
        """Test charge warning for ASE path."""
        mol = Molecule.from_smiles("[NH4+]")
        mol.generate_conformers(n_conformers=1)

        with pytest.warns(
            UserWarning, match="does not automatically pass molecular charge"
        ):
            get_ml_omm_system(
                mol,
                MLPSettings(
                    ml_potential="ase",
                    ml_system_kwargs={"calculator": object()},
                ),
                torch.device("cpu"),
            )


class TestAseIntegration:
    """End-to-end tests using a real ASE calculator through OpenMM-ML."""

    @pytest.fixture
    def lj_calculator(self):
        """Return a simple Lennard-Jones calculator for testing ASE integration."""
        from ase.calculators.lj import LennardJones

        return LennardJones()

    def test_ase_creates_system_with_correct_particles(self, lj_calculator):
        """Test that an ASE-backed system has the right number of particles."""
        _cache.clear()
        mol = Molecule.from_smiles("O")
        mol.generate_conformers(n_conformers=1)

        system = get_ml_omm_system(
            mol,
            MLPSettings(
                ml_potential="ase",
                ml_system_kwargs={"calculator": lj_calculator},
            ),
            torch.device("cpu"),
        )
        assert system.getNumParticles() == 3

    def test_ase_produces_finite_energy(self, lj_calculator):
        """Test that an ASE-backed system produces a finite energy."""
        _cache.clear()
        mol = Molecule.from_smiles("O")
        mol.generate_conformers(n_conformers=1)

        system = get_ml_omm_system(
            mol,
            MLPSettings(
                ml_potential="ase",
                ml_system_kwargs={"calculator": lj_calculator},
            ),
            torch.device("cpu"),
        )

        integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
        context = openmm.Context(system, integrator)
        context.setPositions(mol.conformers[0].to_openmm())

        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoule_per_mole
        )
        forces = state.getForces(asNumpy=True)

        assert math.isfinite(energy)
        assert forces.shape == (3, 3)
        assert np.all(np.isfinite(forces))
