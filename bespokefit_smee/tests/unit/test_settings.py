"""Unit tests for settings module."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from openmm import unit as omm_unit
from pydantic import ValidationError

from bespokefit_smee import __version__
from bespokefit_smee.settings import (
    _DEFAULT_SMILES_PLACEHOLDER,
    MLMDSamplingSettings,
    MMMDMetadynamicsSamplingSettings,
    MMMDSamplingSettings,
    MSMSettings,
    ParameterisationSettings,
    RegularisationSettings,
    TrainingSettings,
    WorkflowSettings,
)


class TestSamplingSettingsBase:
    """Tests for sampling settings base class."""

    @pytest.fixture
    def valid_mm_md_settings(self):
        return MMMDSamplingSettings()

    def test_default_values(self, valid_mm_md_settings):
        """Test that default values are set correctly."""
        assert valid_mm_md_settings.ml_potential == "egret-1"
        assert valid_mm_md_settings.timestep.value_in_unit(omm_unit.femtoseconds) == 1.0
        assert valid_mm_md_settings.temperature.value_in_unit(omm_unit.kelvin) == 500.0
        assert valid_mm_md_settings.n_conformers == 10

    def test_equilibration_n_steps_per_conformer(self, valid_mm_md_settings):
        """Test calculation of equilibration steps."""
        expected = int(
            valid_mm_md_settings.equilibration_sampling_time_per_conformer
            / valid_mm_md_settings.timestep
        )
        assert valid_mm_md_settings.equilibration_n_steps_per_conformer == expected

    def test_production_n_snapshots_per_conformer(self, valid_mm_md_settings):
        """Test calculation of production snapshots."""
        expected = int(
            valid_mm_md_settings.production_sampling_time_per_conformer
            / valid_mm_md_settings.snapshot_interval
        )
        assert valid_mm_md_settings.production_n_snapshots_per_conformer == expected

    def test_production_n_steps_per_snapshot_per_conformer(self, valid_mm_md_settings):
        """Test calculation of production steps per snapshot."""
        expected = int(
            valid_mm_md_settings.snapshot_interval / valid_mm_md_settings.timestep
        )
        assert (
            valid_mm_md_settings.production_n_steps_per_snapshot_per_conformer
            == expected
        )

    def test_invalid_equilibration_time_not_divisible_by_timestep(self):
        """Test that invalid equilibration time raises error."""
        with pytest.raises(ValidationError, match="must be divisible by the timestep"):
            MMMDSamplingSettings(
                timestep=1.5 * omm_unit.femtoseconds,
                equilibration_sampling_time_per_conformer=1.0 * omm_unit.picoseconds,
            )

    def test_invalid_production_time_not_divisible_by_timestep(self):
        """Test that invalid production time raises error."""
        with pytest.raises(ValidationError, match="must be divisible by the timestep"):
            MMMDSamplingSettings(
                timestep=1.5 * omm_unit.femtoseconds,
                production_sampling_time_per_conformer=1.0 * omm_unit.picoseconds,
            )

    def test_output_types(self, valid_mm_md_settings):
        """Test that output types are defined."""
        from bespokefit_smee.outputs import OutputType

        assert OutputType.PDB_TRAJECTORY in valid_mm_md_settings.output_types


class TestMMMDSamplingSettings:
    """Tests for MM MD sampling settings."""

    def test_sampling_protocol_is_mm_md(self):
        """Test that sampling protocol is set correctly."""
        settings = MMMDSamplingSettings()
        assert settings.sampling_protocol == "mm_md"

    def test_to_yaml_and_from_yaml(self, tmp_path):
        """Test YAML serialization round-trip."""
        settings = MMMDSamplingSettings(
            n_conformers=5, temperature=600 * omm_unit.kelvin
        )
        yaml_path = tmp_path / "settings.yaml"
        settings.to_yaml(yaml_path)

        loaded = MMMDSamplingSettings.from_yaml(yaml_path)
        assert loaded.n_conformers == 5
        assert loaded.temperature.value_in_unit(omm_unit.kelvin) == 600.0


class TestMLMDSamplingSettings:
    """Tests for ML MD sampling settings."""

    def test_sampling_protocol_is_ml_md(self):
        """Test that sampling protocol is set correctly."""
        settings = MLMDSamplingSettings()
        assert settings.sampling_protocol == "ml_md"


class TestMMMDMetadynamicsSamplingSettings:
    """Tests for MM MD metadynamics sampling settings."""

    def test_sampling_protocol_is_mm_md_metadynamics(self):
        """Test that sampling protocol is set correctly."""
        settings = MMMDMetadynamicsSamplingSettings()
        assert settings.sampling_protocol == "mm_md_metadynamics"

    def test_default_metadynamics_parameters(self):
        """Test that default metadynamics parameters are set."""
        settings = MMMDMetadynamicsSamplingSettings()
        assert settings.bias_factor == 10.0
        assert settings.bias_width == np.pi / 10
        assert settings.bias_height.value_in_unit(omm_unit.kilojoules_per_mole) == 2.0

    def test_n_steps_per_bias(self):
        """Test calculation of steps per bias."""
        settings = MMMDMetadynamicsSamplingSettings(
            timestep=1.0 * omm_unit.femtoseconds,
            bias_frequency=0.5 * omm_unit.picoseconds,
        )
        assert settings.n_steps_per_bias == 500

    def test_n_steps_per_bias_save(self):
        """Test calculation of steps per bias save."""
        settings = MMMDMetadynamicsSamplingSettings(
            timestep=1.0 * omm_unit.femtoseconds,
            bias_save_frequency=1.0 * omm_unit.picoseconds,
        )
        assert settings.n_steps_per_bias_save == 1000

    def test_invalid_bias_frequency_not_divisible_by_timestep(self):
        """Test that invalid bias frequency raises error."""
        with pytest.raises(ValidationError, match="must be divisible by the timestep"):
            MMMDMetadynamicsSamplingSettings(
                timestep=1.5 * omm_unit.femtoseconds,
                bias_frequency=1.0 * omm_unit.picoseconds,
            )

    def test_invalid_bias_save_frequency_not_divisible_by_timestep(self):
        """Test that invalid bias save frequency raises error."""
        with pytest.raises(ValidationError, match="must be divisible by the timestep"):
            MMMDMetadynamicsSamplingSettings(
                timestep=1.5 * omm_unit.femtoseconds,
                bias_save_frequency=1.0 * omm_unit.picoseconds,
            )

    def test_production_time_not_divisible_by_bias_frequency(self):
        """Test that production time must be divisible by bias frequency."""
        with pytest.raises(ValidationError, match="must be divisible by"):
            MMMDMetadynamicsSamplingSettings(
                production_sampling_time_per_conformer=10.3 * omm_unit.picoseconds,
                bias_frequency=0.5 * omm_unit.picoseconds,
            )

    def test_output_types_includes_metadynamics_bias(self):
        """Test that output types include metadynamics bias."""
        from bespokefit_smee.outputs import OutputType

        settings = MMMDMetadynamicsSamplingSettings()
        assert OutputType.METADYNAMICS_BIAS in settings.output_types
        assert OutputType.PDB_TRAJECTORY in settings.output_types


class TestRegularisationSettings:
    """Tests for regularisation settings."""

    def test_default_values(self):
        """Test default values."""
        settings = RegularisationSettings()
        assert settings.regularisation_strength == 100.0
        assert settings.regularisation_value == "initial"
        assert "ProperTorsions" in settings.parameters
        assert "k" in settings.parameters["ProperTorsions"]

    def test_regularisation_value_validation(self):
        """Test that only valid regularisation values are accepted."""
        RegularisationSettings(regularisation_value="initial")
        RegularisationSettings(regularisation_value="zero")

        with pytest.raises(ValidationError):
            RegularisationSettings(regularisation_value="invalid")

    @given(
        strength=st.floats(
            min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        )
    )
    @hypothesis_settings(max_examples=10)
    def test_regularisation_strength_property(self, strength):
        """Test that regularisation strength can be set to positive values."""
        settings = RegularisationSettings(regularisation_strength=strength)
        assert settings.regularisation_strength == strength


class TestTrainingSettings:
    """Tests for training settings."""

    def test_default_values(self):
        """Test default values."""
        settings = TrainingSettings()
        assert settings.optimiser == "adam"
        assert settings.n_epochs == 1000
        assert settings.learning_rate == 0.01
        assert settings.learning_rate_decay == 1.0
        assert settings.loss_force_weight == 0.1

    def test_optimiser_validation(self):
        """Test that only valid optimisers are accepted."""
        TrainingSettings(optimiser="adam")
        TrainingSettings(optimiser="lm")

        with pytest.raises(ValidationError):
            TrainingSettings(optimiser="invalid")

    def test_output_types(self):
        """Test that output types are defined."""
        from bespokefit_smee.outputs import OutputType

        settings = TrainingSettings()
        assert OutputType.TENSORBOARD in settings.output_types
        assert OutputType.TRAINING_METRICS in settings.output_types

    @given(
        n_epochs=st.integers(min_value=1, max_value=10000),
        learning_rate=st.floats(
            min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @hypothesis_settings(max_examples=10)
    def test_training_parameters_property(self, n_epochs, learning_rate):
        """Test that training parameters can be set."""
        settings = TrainingSettings(n_epochs=n_epochs, learning_rate=learning_rate)
        assert settings.n_epochs == n_epochs
        assert settings.learning_rate == learning_rate


class TestMSMSettings:
    """Tests for MSM settings."""

    def test_default_values(self):
        """Test default values."""
        settings = MSMSettings()
        assert settings.ml_potential == "egret-1"
        assert settings.vib_scaling == 0.957

    def test_finite_step_and_tolerance(self):
        """Test that finite step and tolerance are set."""
        settings = MSMSettings()
        assert settings.finite_step.value_in_unit(omm_unit.nanometers) == pytest.approx(
            0.0005291772
        )
        assert settings.tolerance.value_in_unit(
            omm_unit.kilocalories_per_mole / omm_unit.angstrom
        ) == pytest.approx(0.005291772)


class TestParameterisationSettings:
    """Tests for parameterisation settings."""

    def test_valid_smiles(self):
        """Test that valid SMILES are accepted."""
        settings = ParameterisationSettings(smiles="CCO")
        assert settings.smiles == "CCO"

    def test_invalid_smiles_raises_error(self):
        """Test that invalid SMILES raise error."""
        with pytest.raises(ValidationError, match="Invalid SMILES"):
            ParameterisationSettings(smiles="invalid_smiles_123")

    def test_placeholder_smiles_raises_error(self):
        """Test that placeholder SMILES raise error."""
        with pytest.raises(ValidationError, match="Invalid SMILES"):
            ParameterisationSettings(smiles=_DEFAULT_SMILES_PLACEHOLDER)

    def test_default_initial_force_field(self):
        """Test default initial force field."""
        settings = ParameterisationSettings(smiles="CCO")
        assert settings.initial_force_field == "openff_unconstrained-2.2.1.offxml"

    def test_default_excluded_smirks(self):
        """Test that default excluded SMIRKS are set."""
        settings = ParameterisationSettings(smiles="CCO")
        assert len(settings.excluded_smirks) == 3
        assert "[*:1]-[*:2]#[*:3]-[*:4]" in settings.excluded_smirks

    def test_linear_harmonics_and_torsions(self):
        """Test linear harmonics and torsions settings."""
        settings = ParameterisationSettings(smiles="CCO")
        assert settings.linear_harmonics is True
        assert settings.linear_torsions is False

    def test_expand_torsions_default(self):
        """Test that expand torsions is True by default."""
        settings = ParameterisationSettings(smiles="CCO")
        assert settings.expand_torsions is True

    def test_msm_settings_optional(self):
        """Test that MSM settings are optional."""
        settings = ParameterisationSettings(smiles="CCO")
        assert settings.msm_settings is None

        settings_with_msm = ParameterisationSettings(
            smiles="CCO", msm_settings=MSMSettings()
        )
        assert settings_with_msm.msm_settings is not None

    @given(smiles=st.sampled_from(["CCO", "CC", "C", "CCCC", "c1ccccc1"]))
    @hypothesis_settings(max_examples=5)
    def test_valid_simple_smiles(self, smiles):
        """Test that simple valid SMILES are accepted."""
        settings = ParameterisationSettings(smiles=smiles)
        assert settings.smiles == smiles


class TestWorkflowSettings:
    """Tests for workflow settings."""

    @pytest.fixture
    def valid_workflow_settings(self):
        return WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            device_type="cpu",
        )

    def test_default_values(self, valid_workflow_settings):
        """Test default values."""
        assert valid_workflow_settings.version == __version__
        assert valid_workflow_settings.output_dir == Path(".")
        assert valid_workflow_settings.n_iterations == 2
        assert valid_workflow_settings.memory is False

    def test_version_validation_compatible(self):
        """Test that compatible versions pass validation."""
        # Should accept same major.minor version
        settings = WorkflowSettings(
            version=__version__,
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
        )
        assert settings.version == __version__

    def test_version_validation_incompatible_major(self):
        """Test that incompatible major version raises error."""
        from packaging.version import Version

        current = Version(__version__)
        incompatible = f"{current.major + 1}.0.0"

        with pytest.raises(ValueError, match="Incompatible settings version"):
            WorkflowSettings(
                version=incompatible,
                parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            )

    def test_version_validation_incompatible_minor(self):
        """Test that incompatible minor version raises error."""
        from packaging.version import Version

        current = Version(__version__)
        incompatible = f"{current.major}.{current.minor + 1}.0"

        with pytest.raises(ValueError, match="Incompatible settings version"):
            WorkflowSettings(
                version=incompatible,
                parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            )

    def test_device_type_cpu(self):
        """Test that CPU device type is accepted."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            device_type="cpu",
        )
        assert settings.device_type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_type_cuda(self):
        """Test that CUDA device type is accepted when available."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            device_type="cuda",
        )
        assert settings.device_type == "cuda"

    def test_device_type_cuda_unavailable(self, monkeypatch):
        """Test that CUDA device type raises error when unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(ValueError, match="CUDA is not available"):
            WorkflowSettings(
                parameterisation_settings=ParameterisationSettings(smiles="CCO"),
                device_type="cuda",
            )

    def test_device_property(self, valid_workflow_settings):
        """Test that device property returns torch.device."""
        device = valid_workflow_settings.device
        assert isinstance(device, torch.device)
        assert device.type == "cpu"

    def test_get_path_manager(self, valid_workflow_settings):
        """Test that path manager is created correctly."""
        path_manager = valid_workflow_settings.get_path_manager()
        assert path_manager.output_dir == valid_workflow_settings.output_dir
        assert path_manager.n_iterations == valid_workflow_settings.n_iterations

    def test_yaml_round_trip(self, tmp_path, valid_workflow_settings):
        """Test YAML serialization round-trip."""
        yaml_path = tmp_path / "workflow_settings.yaml"
        valid_workflow_settings.to_yaml(yaml_path)

        loaded = WorkflowSettings.from_yaml(yaml_path)
        assert loaded.n_iterations == valid_workflow_settings.n_iterations
        assert loaded.memory == valid_workflow_settings.memory
        assert (
            loaded.parameterisation_settings.smiles
            == valid_workflow_settings.parameterisation_settings.smiles
        )

    def test_discriminated_union_for_sampling_settings(self):
        """Test that discriminated union works for sampling settings."""
        # MM MD
        settings_mm = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            training_sampling_settings=MMMDSamplingSettings(),
        )
        assert settings_mm.training_sampling_settings.sampling_protocol == "mm_md"

        # ML MD
        settings_ml = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            training_sampling_settings=MLMDSamplingSettings(),
        )
        assert settings_ml.training_sampling_settings.sampling_protocol == "ml_md"

        # Metadynamics
        settings_metad = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            training_sampling_settings=MMMDMetadynamicsSamplingSettings(),
        )
        assert (
            settings_metad.training_sampling_settings.sampling_protocol
            == "mm_md_metadynamics"
        )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            WorkflowSettings(
                parameterisation_settings=ParameterisationSettings(smiles="CCO"),
                invalid_field="should_fail",
            )

    @given(
        n_iterations=st.integers(min_value=1, max_value=10),
        memory=st.booleans(),
    )
    @hypothesis_settings(max_examples=10)
    def test_workflow_configuration_property(self, n_iterations, memory):
        """Test that workflow configuration can be set."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            n_iterations=n_iterations,
            memory=memory,
        )
        assert settings.n_iterations == n_iterations
        assert settings.memory == memory
