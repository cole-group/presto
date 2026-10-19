"""Unit tests for the corrections module."""

import importlib.util
import sys
import types
from typing import Literal

import numpy as np
import openff.interchange
import openmm
import pytest
import torch
from loguru import logger
from openff.toolkit import ForceField, Molecule
from openmm.app.internal.customgbforces import GBSAGBn2Force
from openmm.app.internal.lcpo import addLCPOForce, getLCPOParamsTopology

from presto import corrections
from presto._exceptions import MissingOptionalDependencyError
from presto.corrections import (
    CORRECTION_FORCE_NAME_PREFIX,
    AnySolvSMDCorrection,
    EnergyCorrection,
    GBn2SASACorrection,
    add_correction_forces,
    apply_corrections_to_dataset,
    build_correction,
    correction_force_groups,
    get_smee_charges,
    get_system_charges,
    resolve_sampling_corrections,
)
from presto.data_utils import create_dataset_with_uniform_weights
from presto.settings import (
    AnySolvSMDCorrectionSettings,
    GBn2SASACorrectionSettings,
    _EnergyCorrectionSettingsBase,
)

_CPU = torch.device("cpu")
_KCAL = openmm.unit.kilocalorie_per_mole
_KCAL_ANGS = openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom


class LinearCorrectionSettings(_EnergyCorrectionSettingsBase):
    """Settings for a test correction with energy 0.5 * sum(x) kJ/mol/nm."""

    correction_type: Literal["linear_test"] = "linear_test"


class LinearCorrection(EnergyCorrection):
    """A cheap, position-dependent test correction."""

    side = "reference"

    def create_openmm_forces(self) -> list[openmm.Force]:
        """Return a force with energy 0.5 * x per particle."""
        force = openmm.CustomExternalForce("0.5*x")
        for i in range(self.mol.n_atoms):
            force.addParticle(i, [])
        return [force]


@pytest.fixture
def linear_correction(monkeypatch):
    """Register the linear test correction."""
    monkeypatch.setitem(
        corrections._CORRECTIONS_REGISTRY, LinearCorrectionSettings, LinearCorrection
    )
    return LinearCorrectionSettings()


@pytest.fixture
def ethanol():
    """Ethanol with two conformers and Gasteiger charges."""
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=2)
    mol.assign_partial_charges("gasteiger")
    return mol


def _coords(mol: Molecule) -> np.ndarray:
    return np.array([conformer.m_as("angstrom") for conformer in mol.conformers])


def _reference_energies_and_forces(
    system: openmm.System, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(1.0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    energies, forces = [], []
    for conformer in coords:
        context.setPositions(conformer / 10.0)
        state = context.getState(getEnergy=True, getForces=True)
        energies.append(state.getPotentialEnergy().value_in_unit(_KCAL))
        forces.append(state.getForces(asNumpy=True).value_in_unit(_KCAL_ANGS))
    return np.array(energies), np.array(forces)


def _empty_system(n_particles: int) -> openmm.System:
    system = openmm.System()
    for _ in range(n_particles):
        system.addParticle(1.0)
    return system


class TestRegistry:
    """Tests for the correction registry."""

    def test_every_correction_settings_type_is_registered(self):
        """Every correction in the settings unions has an implementation."""
        assert (
            corrections._CORRECTIONS_REGISTRY[AnySolvSMDCorrectionSettings]
            is AnySolvSMDCorrection
        )
        assert (
            corrections._CORRECTIONS_REGISTRY[GBn2SASACorrectionSettings]
            is GBn2SASACorrection
        )

    def test_sides(self):
        """Corrections declare which model they apply to."""
        assert AnySolvSMDCorrection.side == "reference"
        assert GBn2SASACorrection.side == "mm"

    def test_resolve_sampling_corrections(self):
        """'from_fit' is replaced by the fit corrections."""
        fit = [GBn2SASACorrectionSettings()]
        assert resolve_sampling_corrections("from_fit", fit) == fit
        assert resolve_sampling_corrections([], fit) == []


class TestGBn2SASACorrection:
    """Tests for the GBneck2 + SASA correction."""

    def test_matches_openmm_gbn2_with_lcpo(self, ethanol):
        """The default correction matches OpenMM's GBn2 force plus LCPO."""
        charges = ethanol.partial_charges.m
        topology = ethanol.to_topology().to_openmm()
        system = _empty_system(ethanol.n_atoms)
        gb_force = GBSAGBn2Force(SA=None)
        gb_force.addParticles(
            np.column_stack([charges, GBSAGBn2Force.getStandardParameters(topology)])
        )
        gb_force.finalize()
        system.addForce(gb_force)
        addLCPOForce(system, getLCPOParamsTopology(topology), False)

        correction = build_correction(
            GBn2SASACorrectionSettings(), ethanol, device=_CPU, charges=charges
        )
        energies, forces = correction.compute(_coords(ethanol))
        expected_energies, expected_forces = _reference_energies_and_forces(
            system, _coords(ethanol)
        )

        assert np.allclose(energies, expected_energies, atol=1e-5)
        assert np.allclose(forces, expected_forces, atol=1e-4)

    def test_ace_matches_openmm(self, ethanol):
        """The ACE option matches OpenMM's GBn2 force with SA='ACE'."""
        charges = ethanol.partial_charges.m
        topology = ethanol.to_topology().to_openmm()
        system = _empty_system(ethanol.n_atoms)
        gb_force = GBSAGBn2Force(SA="ACE")
        gb_force.addParticles(
            np.column_stack([charges, GBSAGBn2Force.getStandardParameters(topology)])
        )
        gb_force.finalize()
        system.addForce(gb_force)

        correction = build_correction(
            GBn2SASACorrectionSettings(sasa_model="ace"),
            ethanol,
            device=_CPU,
            charges=charges,
        )
        energies, _ = correction.compute(_coords(ethanol))
        expected, _ = _reference_energies_and_forces(system, _coords(ethanol))

        assert np.allclose(energies, expected, atol=1e-5)

    def test_surface_area_term_scales_with_surface_tension(self, ethanol):
        """The LCPO term is proportional to the surface tension."""
        coords = _coords(ethanol)
        zero_charges = np.zeros(ethanol.n_atoms)

        def _energy(**kwargs):
            correction = build_correction(
                GBn2SASACorrectionSettings(**kwargs),
                ethanol,
                device=_CPU,
                charges=zero_charges,
            )
            return correction.compute(coords)[0]

        surface_energy = _energy()
        doubled = _energy(
            surface_tension=0.01 * _KCAL / openmm.unit.angstrom**2,
        )

        assert np.all(surface_energy > 0)
        assert np.allclose(doubled, 2 * surface_energy)
        assert np.allclose(_energy(sasa_model=None), 0.0)

    def test_requires_one_charge_per_atom(self, ethanol):
        """Charges are required, and virtual sites are not supported."""
        with pytest.raises(ValueError, match="requires the MM partial charges"):
            build_correction(GBn2SASACorrectionSettings(), ethanol, device=_CPU)
        with pytest.raises(ValueError, match="Virtual sites"):
            build_correction(
                GBn2SASACorrectionSettings(),
                ethanol,
                device=_CPU,
                charges=np.zeros(ethanol.n_atoms + 1),
            )

    def test_unsupported_lcpo_element_raises_clear_error(self):
        """Elements without LCPO parameters suggest another surface area model."""
        mol = Molecule.from_smiles("CBr")
        correction = build_correction(
            GBn2SASACorrectionSettings(),
            mol,
            device=_CPU,
            charges=np.zeros(mol.n_atoms),
        )
        with pytest.raises(ValueError, match="sasa_model"):
            correction.create_openmm_forces()

        ace = build_correction(
            GBn2SASACorrectionSettings(sasa_model="ace"),
            mol,
            device=_CPU,
            charges=np.zeros(mol.n_atoms),
        )
        assert len(ace.create_openmm_forces()) == 1

    def test_warns_for_elements_without_gbn2_parameters(self):
        """Halogens fall back to OpenMM's default GBn2 parameters, with a warning."""
        mol = Molecule.from_smiles("CCl")
        messages = []
        sink = logger.add(lambda message: messages.append(str(message)))
        try:
            build_correction(
                GBn2SASACorrectionSettings(),
                mol,
                device=_CPU,
                charges=np.zeros(mol.n_atoms),
            ).create_openmm_forces()
        finally:
            logger.remove(sink)

        assert any("['Cl']" in message for message in messages)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Requires a CUDA-capable GPU"
    )
    def test_cuda_matches_cpu(self, ethanol):
        """Corrections evaluated on the GPU match the CPU."""
        charges = ethanol.partial_charges.m
        settings = GBn2SASACorrectionSettings()
        cpu = build_correction(settings, ethanol, device=_CPU, charges=charges)
        gpu = build_correction(
            settings, ethanol, device=torch.device("cuda"), charges=charges
        )

        cpu_energies, cpu_forces = cpu.compute(_coords(ethanol))
        gpu_energies, gpu_forces = gpu.compute(_coords(ethanol))

        assert np.allclose(cpu_energies, gpu_energies, atol=1e-3)
        assert np.allclose(cpu_forces, gpu_forces, atol=1e-2)


class TestCharges:
    """Tests for extracting MM charges."""

    def test_smee_charges_match_the_openmm_system(self):
        """The charges used in the fit match those of the sampled MM system."""
        mol = Molecule.from_smiles("CCO")
        force_field = ForceField("openff_unconstrained-2.3.0.offxml")
        interchange = openff.interchange.Interchange.from_smirnoff(
            force_field, mol.to_topology()
        )
        import smee.converters

        tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

        assert np.allclose(
            get_smee_charges(tensor_top, tensor_ff),
            get_system_charges(interchange.to_openmm_system()),
            atol=1e-6,
        )

    def test_system_without_nonbonded_force_raises(self):
        """A system with no NonbondedForce has no charges to take."""
        with pytest.raises(ValueError, match="NonbondedForce"):
            get_system_charges(_empty_system(2))


class TestAddCorrectionForces:
    """Tests for adding correction forces to sampling systems."""

    def test_no_corrections(self, ethanol):
        """Nothing is added when there are no corrections."""
        system = _empty_system(ethanol.n_atoms)
        assert add_correction_forces(system, ethanol, [], device=_CPU) is None
        assert system.getNumForces() == 0

    def test_forces_use_the_lowest_free_group_and_are_tagged(
        self, ethanol, linear_correction
    ):
        """Correction forces get their own, recognisable force group."""
        system = _empty_system(ethanol.n_atoms)
        existing = openmm.CustomExternalForce("0")
        existing.setForceGroup(0)
        system.addForce(existing)
        bias = openmm.CustomExternalForce("0")
        bias.setForceGroup(31)
        system.addForce(bias)

        group = add_correction_forces(system, ethanol, [linear_correction], device=_CPU)

        assert group == 1
        assert correction_force_groups(system) == {1}
        added = system.getForce(2)
        assert added.getName() == f"{CORRECTION_FORCE_NAME_PREFIX}linear_test"
        assert added.getForceGroup() == 1


class TestApplyCorrectionsToDataset:
    """Tests for setting a dataset's correction columns."""

    @pytest.fixture
    def dataset(self, ethanol):
        """An uncorrected dataset of the ethanol conformers."""
        coords = torch.tensor(_coords(ethanol))
        n_confs = len(coords)
        return create_dataset_with_uniform_weights(
            smiles=ethanol.to_smiles(mapped=True),
            coords=coords,
            energy=torch.zeros(n_confs, dtype=torch.float64),
            forces=torch.zeros(n_confs, ethanol.n_atoms, 3, dtype=torch.float64),
        )

    def test_recompute_sets_reference_and_mm_columns(
        self, ethanol, dataset, linear_correction
    ):
        """Reference and MM corrections are stored in their own columns."""
        gb = GBn2SASACorrectionSettings()
        charges = ethanol.partial_charges.m
        result = apply_corrections_to_dataset(
            dataset,
            ethanol,
            reference_corrections=[linear_correction],
            mm_corrections=[gb],
            device=_CPU,
            mm_charges=charges,
        )
        entry = result[0]
        coords = _coords(ethanol)

        expected_ref_e, expected_ref_f = build_correction(
            linear_correction, ethanol, device=_CPU
        ).compute(coords)
        expected_mm_e, expected_mm_f = build_correction(
            gb, ethanol, device=_CPU, charges=charges
        ).compute(coords)

        assert np.allclose(entry["reference_energy_correction"], expected_ref_e)
        assert np.allclose(
            entry["reference_forces_correction"], expected_ref_f.flatten(), atol=1e-5
        )
        assert np.allclose(entry["mm_energy_correction"], expected_mm_e, atol=1e-4)
        assert np.allclose(
            entry["mm_forces_correction"], expected_mm_f.flatten(), atol=1e-3
        )
        # The uncorrected reference is untouched
        assert torch.all(entry["energy"] == 0.0)

    def test_keep_and_zero_modes(self, ethanol, dataset, linear_correction):
        """Stored reference corrections can be kept or zeroed, MM ones are rewritten."""
        corrected = apply_corrections_to_dataset(
            dataset,
            ethanol,
            reference_corrections=[linear_correction],
            mm_corrections=[],
            device=_CPU,
        )
        stored = corrected[0]["reference_energy_correction"]
        assert not torch.all(stored == 0.0)

        kept = apply_corrections_to_dataset(
            corrected,
            ethanol,
            reference_corrections=[],
            mm_corrections=[],
            device=_CPU,
            reference_mode="keep",
        )
        assert torch.allclose(kept[0]["reference_energy_correction"], stored)

        zeroed = apply_corrections_to_dataset(
            corrected,
            ethanol,
            reference_corrections=[linear_correction],
            mm_corrections=[],
            device=_CPU,
            reference_mode="zero",
        )
        assert torch.all(zeroed[0]["reference_energy_correction"] == 0.0)
        assert torch.all(zeroed[0]["reference_forces_correction"] == 0.0)

    def test_stale_mm_corrections_are_overwritten(self, ethanol, dataset):
        """MM corrections are rewritten from the current MM model, even if absent."""
        gb_corrected = apply_corrections_to_dataset(
            dataset,
            ethanol,
            reference_corrections=[],
            mm_corrections=[GBn2SASACorrectionSettings()],
            device=_CPU,
            mm_charges=ethanol.partial_charges.m,
            reference_mode="keep",
        )
        assert not torch.all(gb_corrected[0]["mm_energy_correction"] == 0.0)

        vacuum = apply_corrections_to_dataset(
            gb_corrected,
            ethanol,
            reference_corrections=[],
            mm_corrections=[],
            device=_CPU,
            reference_mode="keep",
        )
        assert torch.all(vacuum[0]["mm_energy_correction"] == 0.0)
        assert torch.all(vacuum[0]["mm_forces_correction"] == 0.0)

    def test_reference_energy_coords_are_used_for_energies(
        self, ethanol, linear_correction
    ):
        """Mapped energies are corrected at the coordinates they were evaluated at."""
        coords = torch.tensor(_coords(ethanol))
        shifted = coords + 1.0
        dataset = create_dataset_with_uniform_weights(
            smiles=ethanol.to_smiles(mapped=True),
            coords=coords,
            energy=torch.zeros(len(coords), dtype=torch.float64),
            forces=torch.zeros_like(coords),
            reference_energy_coords=shifted,
        )

        result = apply_corrections_to_dataset(
            dataset,
            ethanol,
            reference_corrections=[linear_correction],
            mm_corrections=[],
            device=_CPU,
        )
        correction = build_correction(linear_correction, ethanol, device=_CPU)

        assert np.allclose(
            result[0]["reference_energy_correction"],
            correction.compute(shifted.numpy())[0],
            atol=1e-5,
        )
        assert np.allclose(
            result[0]["reference_forces_correction"],
            correction.compute(coords.numpy())[1].flatten(),
            atol=1e-5,
        )


class TestAnySolvSMDCorrection:
    """Tests for the AnySolv SMD correction, with AnySolv faked."""

    def test_compute_converts_units_and_passes_settings(self, ethanol, fake_anysolv):
        """Energies and forces are converted from eV to kcal/mol."""
        settings = AnySolvSMDCorrectionSettings(
            checkpoint="model_compact", dtype="float64", inference_settings="fast"
        )
        correction = build_correction(settings, ethanol, device=_CPU)
        coords = _coords(ethanol)

        energies, forces = correction.compute(coords)

        expected = -0.1 * coords[:, :, 0].sum(axis=1) * 23.060548
        assert np.allclose(energies, expected)
        assert np.allclose(forces[:, :, 0], 0.1 * 23.060548)
        call = fake_anysolv[0]
        assert call["checkpoint"] == "model_compact"
        assert call["charge"] == 0
        assert call["spin"] == 1
        assert call["solvent"] == "water"
        assert call["device"] == "cpu"
        assert call["dtype"] == torch.float64
        assert call["inference_settings"] == "fast"
        assert list(call["atomic_numbers"]) == [
            atom.atomic_number for atom in ethanol.atoms
        ]

    def test_total_charge_is_passed(self, fake_anysolv):
        """The molecule's total charge is passed to AnySolv."""
        mol = Molecule.from_smiles("CC(=O)[O-]")
        mol.generate_conformers(n_conformers=1)
        correction = build_correction(
            AnySolvSMDCorrectionSettings(checkpoint="model_compact"), mol, device=_CPU
        )
        correction.compute(_coords(mol))

        assert fake_anysolv[0]["charge"] == -1

    def test_python_force_matches_compute(self, ethanol, fake_anysolv):
        """The sampling force gives the same energies and forces as `compute`."""
        correction = build_correction(
            AnySolvSMDCorrectionSettings(checkpoint="model_compact"),
            ethanol,
            device=_CPU,
        )
        system = _empty_system(ethanol.n_atoms)
        for force in correction.create_openmm_forces():
            system.addForce(force)
        context = openmm.Context(
            system,
            openmm.VerletIntegrator(1.0),
            openmm.Platform.getPlatformByName("CPU"),
        )
        coords = _coords(ethanol)
        context.setPositions(coords[0] / 10.0)
        state = context.getState(getEnergy=True, getForces=True)

        energies, forces = correction.compute(coords[:1])

        assert np.isclose(state.getPotentialEnergy().value_in_unit(_KCAL), energies[0])
        assert np.allclose(
            state.getForces(asNumpy=True).value_in_unit(_KCAL_ANGS), forces[0]
        )

    def test_moe_checkpoint_is_downloaded_from_the_hub(self, monkeypatch, fake_anysolv):
        """'model_moe' is fetched from Hugging Face and passed as a path."""
        model_module = types.ModuleType("anysolv.model")
        model_module._CKPT_DIR = "/nonexistent"
        hub = types.ModuleType("huggingface_hub")
        hub.try_to_load_from_cache = lambda repo, filename: None
        hub.hf_hub_download = lambda repo, filename: f"/cache/{repo}/{filename}"
        monkeypatch.setitem(sys.modules, "anysolv.model", model_module)
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

        AnySolvSMDCorrection.prepare(AnySolvSMDCorrectionSettings())

        assert (
            corrections._resolve_anysolv_checkpoint("model_moe")
            == "/cache/antonknee/anysolv/model_moe.pt"
        )

    def test_gated_moe_download_explains_access(self, monkeypatch, fake_anysolv):
        """A failed download explains how to get access to the gated weights."""
        model_module = types.ModuleType("anysolv.model")
        model_module._CKPT_DIR = "/nonexistent"

        def hf_hub_download(repo, filename):
            raise RuntimeError("401 Client Error: gated repo")

        hub = types.ModuleType("huggingface_hub")
        hub.try_to_load_from_cache = lambda repo, filename: None
        hub.hf_hub_download = hf_hub_download
        monkeypatch.setitem(sys.modules, "anysolv.model", model_module)
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

        with pytest.raises(RuntimeError, match="FAIR Chemistry License"):
            AnySolvSMDCorrection.prepare(AnySolvSMDCorrectionSettings())

    def test_release_clears_the_model_cache(self, monkeypatch):
        """Releasing frees AnySolv's cached models."""
        predict = types.ModuleType("anysolv.predict")
        predict._MODEL_CACHE = {"key": object()}
        monkeypatch.setitem(sys.modules, "anysolv.predict", predict)

        AnySolvSMDCorrection.release()

        assert predict._MODEL_CACHE == {}


class TestMissingAnySolv:
    """Tests for running without the optional AnySolv dependency."""

    def test_prepare_raises_with_install_hint(self, monkeypatch):
        """A missing AnySolv fails fast with installation instructions."""
        monkeypatch.setitem(sys.modules, "anysolv", None)
        with pytest.raises(MissingOptionalDependencyError, match="conda-forge"):
            AnySolvSMDCorrection.prepare(
                AnySolvSMDCorrectionSettings(checkpoint="model_compact")
            )

    def test_settings_only_warn(self, monkeypatch):
        """Settings can still be created and saved without AnySolv."""
        monkeypatch.setitem(sys.modules, "anysolv", None)
        messages = []
        sink = logger.add(lambda message: messages.append(str(message)))
        try:
            AnySolvSMDCorrectionSettings()
        finally:
            logger.remove(sink)

        assert any("AnySolv is not installed" in message for message in messages)

    def test_gb_corrections_do_not_need_anysolv(self, monkeypatch, ethanol):
        """The MM GBneck2 correction only needs OpenMM."""
        monkeypatch.setitem(sys.modules, "anysolv", None)
        corrections.prepare_corrections([GBn2SASACorrectionSettings()])
        correction = build_correction(
            GBn2SASACorrectionSettings(),
            ethanol,
            device=_CPU,
            charges=ethanol.partial_charges.m,
        )
        assert np.all(np.isfinite(correction.compute(_coords(ethanol))[0]))


@pytest.mark.skipif(
    importlib.util.find_spec("anysolv") is None, reason="AnySolv is not installed"
)
class TestRealAnySolv:
    """Tests using the AnySolv package and its bundled compact model."""

    def test_compact_model_gives_a_conservative_solvation_correction(self, ethanol):
        """Forces are minus the gradient of the energy, and solvation is favourable."""
        correction = build_correction(
            AnySolvSMDCorrectionSettings(checkpoint="model_compact", dtype="float64"),
            ethanol,
            device=_CPU,
        )
        coords = _coords(ethanol)[:1]
        energies, forces = correction.compute(coords)

        assert energies[0] < 0.0
        step = 1e-4
        plus, minus = coords.copy(), coords.copy()
        plus[0, 2, 0] += step
        minus[0, 2, 0] -= step
        finite_difference = -(
            correction.compute(plus)[0][0] - correction.compute(minus)[0][0]
        ) / (2 * step)
        assert np.isclose(forces[0, 2, 0], finite_difference, rtol=1e-3, atol=1e-3)
