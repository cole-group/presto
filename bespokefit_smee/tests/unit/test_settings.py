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
    MMMDMetadynamicsTorsionMinimisationSamplingSettings,
    MMMDSamplingSettings,
    OutlierFilterSettings,
    ParameterisationSettings,
    TrainingSettings,
    TypeGenerationSettings,
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


class TestMMMDMetadynamicsTorsionMinimisationSamplingSettings:
    """Tests for MM MD metadynamics with torsion minimisation sampling settings."""

    def test_sampling_protocol(self):
        """Test that sampling protocol is set correctly."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert settings.sampling_protocol == "mm_md_metadynamics_torsion_minimisation"

    def test_inherits_from_metadynamics_settings(self):
        """Test that it inherits from MMMDMetadynamicsSamplingSettings."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert isinstance(settings, MMMDMetadynamicsSamplingSettings)

    def test_default_minimisation_steps(self):
        """Test default minimisation steps."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert settings.ml_minimisation_steps == 10
        assert settings.mm_minimisation_steps == 10

    def test_default_torsion_restraint_force_constant(self):
        """Test default torsion restraint force constant."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert (
            settings.torsion_restraint_force_constant.value_in_unit(
                omm_unit.kilojoules_per_mole / omm_unit.radian**2
            )
            == 0.0
        )

    def test_default_mmmd_loss_weights(self):
        """Test default loss weights for MMMD samples."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert settings.loss_energy_weight_mmmd == 1000.0
        assert settings.loss_force_weight_mmmd == 0.1

    def test_default_torsion_min_loss_weights(self):
        """Test default loss weights for torsion-minimised samples."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert settings.loss_energy_weight_mm_torsion_min == 1000.0
        assert settings.loss_force_weight_mm_torsion_min == 0.1
        assert settings.loss_energy_weight_ml_torsion_min == 1000.0
        assert settings.loss_force_weight_ml_torsion_min == 0.1

    def test_custom_minimisation_steps(self):
        """Test custom minimisation steps."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            ml_minimisation_steps=20,
            mm_minimisation_steps=15,
        )
        assert settings.ml_minimisation_steps == 20
        assert settings.mm_minimisation_steps == 15

    def test_custom_torsion_restraint_force_constant(self):
        """Test custom torsion restraint force constant."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            torsion_restraint_force_constant=500.0
            * omm_unit.kilojoules_per_mole
            / omm_unit.radian**2,
        )
        assert (
            settings.torsion_restraint_force_constant.value_in_unit(
                omm_unit.kilojoules_per_mole / omm_unit.radian**2
            )
            == 500.0
        )

    def test_custom_loss_weights(self):
        """Test custom loss weights."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            loss_energy_weight_mmmd=500.0,
            loss_force_weight_mmmd=0.5,
            loss_energy_weight_mm_torsion_min=200.0,
            loss_force_weight_mm_torsion_min=0.0,
            loss_energy_weight_ml_torsion_min=100.0,
            loss_force_weight_ml_torsion_min=0.1,
        )
        assert settings.loss_energy_weight_mmmd == 500.0
        assert settings.loss_force_weight_mmmd == 0.5
        assert settings.loss_energy_weight_mm_torsion_min == 200.0
        assert settings.loss_force_weight_mm_torsion_min == 0.0

    def test_yaml_round_trip(self, tmp_path):
        """Test YAML serialization round-trip."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            ml_minimisation_steps=25,
            mm_minimisation_steps=30,
            torsion_restraint_force_constant=750.0
            * omm_unit.kilojoules_per_mole
            / omm_unit.radian**2,
            loss_energy_weight_mmmd=800.0,
            loss_force_weight_mm_torsion_min=0.05,
        )
        yaml_path = tmp_path / "settings.yaml"
        settings.to_yaml(yaml_path)

        loaded = MMMDMetadynamicsTorsionMinimisationSamplingSettings.from_yaml(
            yaml_path
        )
        assert loaded.ml_minimisation_steps == 25
        assert loaded.mm_minimisation_steps == 30
        assert (
            loaded.torsion_restraint_force_constant.value_in_unit(
                omm_unit.kilojoules_per_mole / omm_unit.radian**2
            )
            == 750.0
        )
        assert loaded.loss_energy_weight_mmmd == 800.0
        assert loaded.loss_force_weight_mm_torsion_min == 0.05

    def test_inherits_metadynamics_parameters(self):
        """Test that metadynamics parameters are inherited."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            bias_factor=15.0,
        )
        assert settings.bias_factor == 15.0

    def test_output_types_includes_metadynamics_bias(self):
        """Test that output types include metadynamics bias."""
        from bespokefit_smee.outputs import OutputType

        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert OutputType.METADYNAMICS_BIAS in settings.output_types
        assert OutputType.PDB_TRAJECTORY in settings.output_types


class TestTrainingSettings:
    """Tests for training settings."""

    def test_default_values(self):
        """Test default values."""
        settings = TrainingSettings()
        assert settings.optimiser == "adam"
        assert settings.n_epochs == 1000
        assert settings.learning_rate == 0.01
        assert settings.learning_rate_decay == 1.0
        # loss_energy_weight and loss_force_weight are now in sampling settings

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


class TestOutlierFilterSettings:
    """Tests for outlier filter settings."""

    def test_default_values(self):
        """Test default values."""
        settings = OutlierFilterSettings()
        assert settings.energy_outlier_threshold == 10.0
        assert settings.force_outlier_threshold == 50.0
        assert settings.min_conformations == 1

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        settings = OutlierFilterSettings(
            energy_outlier_threshold=5.0,
            force_outlier_threshold=25.0,
            min_conformations=5,
        )
        assert settings.energy_outlier_threshold == 5.0
        assert settings.force_outlier_threshold == 25.0
        assert settings.min_conformations == 5

    def test_disable_energy_filtering(self):
        """Test that energy filtering can be disabled."""
        settings = OutlierFilterSettings(energy_outlier_threshold=None)
        assert settings.energy_outlier_threshold is None
        assert settings.force_outlier_threshold == 50.0

    def test_disable_forces_filtering(self):
        """Test that forces filtering can be disabled."""
        settings = OutlierFilterSettings(force_outlier_threshold=None)
        assert settings.energy_outlier_threshold == 10.0
        assert settings.force_outlier_threshold is None

    def test_disable_all_filtering(self):
        """Test that all filtering can be disabled."""
        settings = OutlierFilterSettings(
            energy_outlier_threshold=None,
            force_outlier_threshold=None,
        )
        assert settings.energy_outlier_threshold is None
        assert settings.force_outlier_threshold is None

    def test_yaml_round_trip(self, tmp_path):
        """Test YAML serialization round-trip."""
        settings = OutlierFilterSettings(
            energy_outlier_threshold=7.5,
            force_outlier_threshold=40.0,
            min_conformations=3,
        )
        yaml_path = tmp_path / "outlier_settings.yaml"
        settings.to_yaml(yaml_path)

        loaded = OutlierFilterSettings.from_yaml(yaml_path)
        assert loaded.energy_outlier_threshold == 7.5
        assert loaded.force_outlier_threshold == 40.0
        assert loaded.min_conformations == 3


class TestTypeGenerationSettings:
    """Tests for type generation settings."""

    def test_default_values(self):
        """Test default values."""
        settings = TypeGenerationSettings()
        assert settings.max_extend_distance == -1
        assert settings.exclude == []

    def test_custom_max_extend_distance(self):
        """Test custom max_extend_distance."""
        settings = TypeGenerationSettings(max_extend_distance=3)
        assert settings.max_extend_distance == 3

    def test_custom_exclude_list(self):
        """Test custom exclude list."""
        exclude_list = ["[*:1]-[*:2]", "[*:1]~[*:2]"]
        settings = TypeGenerationSettings(exclude=exclude_list)
        assert settings.exclude == exclude_list

    def test_max_extend_distance_unlimited(self):
        """Test that -1 means unlimited extension."""
        settings = TypeGenerationSettings(max_extend_distance=-1)
        assert settings.max_extend_distance == -1

    def test_max_extend_distance_zero(self):
        """Test that 0 means no extension."""
        settings = TypeGenerationSettings(max_extend_distance=0)
        assert settings.max_extend_distance == 0

    def test_yaml_round_trip(self, tmp_path):
        """Test YAML serialization round-trip."""
        settings = TypeGenerationSettings(
            max_extend_distance=2, exclude=["[*:1]-[*:2]#[*:3]-[*:4]"]
        )
        yaml_path = tmp_path / "type_gen_settings.yaml"
        settings.to_yaml(yaml_path)

        loaded = TypeGenerationSettings.from_yaml(yaml_path)
        assert loaded.max_extend_distance == 2
        assert loaded.exclude == ["[*:1]-[*:2]#[*:3]-[*:4]"]


class TestParameterisationSettings:
    """Tests for parameterisation settings."""

    def test_valid_smiles(self):
        """Test that valid SMILES are accepted and converted to list."""
        settings = ParameterisationSettings(smiles="CCO")
        assert settings.smiles == ["CCO"]

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
        assert settings.initial_force_field == "openff_unconstrained-2.3.0-rc2.offxml"

    def test_default_type_generation_settings(self):
        """Test that default type generation settings are set."""
        settings = ParameterisationSettings(smiles="CCO")
        assert "ProperTorsions" in settings.type_generation_settings
        assert len(settings.type_generation_settings["ProperTorsions"].exclude) == 3
        assert (
            "[*:1]-[*:2]#[*:3]-[*:4]"
            in settings.type_generation_settings["ProperTorsions"].exclude
        )

    def test_linearise_harmonics(self):
        """Test linear harmonics setting."""
        settings = ParameterisationSettings(smiles="CCO")
        assert settings.linearise_harmonics is True

    def test_expand_torsions_default(self):
        """Test that expand torsions is True by default."""
        settings = ParameterisationSettings(smiles="CCO")
        assert settings.expand_torsions is True

    @given(smiles=st.sampled_from(["CCO", "CC", "C", "CCCC", "c1ccccc1"]))
    @hypothesis_settings(max_examples=5)
    def test_valid_simple_smiles(self, smiles):
        """Test that simple valid SMILES are accepted and converted to list."""
        settings = ParameterisationSettings(smiles=smiles)
        assert settings.smiles == [smiles]


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

        # Metadynamics with torsion minimisation
        settings_metad_torsion = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            training_sampling_settings=MMMDMetadynamicsTorsionMinimisationSamplingSettings(),
        )
        assert (
            settings_metad_torsion.training_sampling_settings.sampling_protocol
            == "mm_md_metadynamics_torsion_minimisation"
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

    def test_outlier_filter_settings_has_default(self, valid_workflow_settings):
        """Test that outlier_filter_settings has default values."""
        assert valid_workflow_settings.outlier_filter_settings is not None
        assert (
            valid_workflow_settings.outlier_filter_settings.energy_outlier_threshold
            == 10.0
        )
        assert (
            valid_workflow_settings.outlier_filter_settings.force_outlier_threshold
            == 50.0
        )

    def test_outlier_filter_settings_can_be_set(self):
        """Test that outlier_filter_settings can be configured."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            device_type="cpu",
            outlier_filter_settings=OutlierFilterSettings(
                energy_outlier_threshold=5.0,  # kcal/mol/atom
                force_outlier_threshold=25.0,
            ),
        )
        assert settings.outlier_filter_settings is not None
        assert settings.outlier_filter_settings.energy_outlier_threshold == 5.0
        assert settings.outlier_filter_settings.force_outlier_threshold == 25.0

    def test_outlier_filter_settings_yaml_round_trip(self, tmp_path):
        """Test that outlier_filter_settings survives YAML round-trip."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            device_type="cpu",
            outlier_filter_settings=OutlierFilterSettings(
                energy_outlier_threshold=7.5,  # kcal/mol/atom
                force_outlier_threshold=30.0,
                min_conformations=2,
            ),
        )
        yaml_path = tmp_path / "workflow_with_outlier.yaml"
        settings.to_yaml(yaml_path)

        loaded = WorkflowSettings.from_yaml(yaml_path)
        assert loaded.outlier_filter_settings is not None
        assert loaded.outlier_filter_settings.energy_outlier_threshold == 7.5
        assert loaded.outlier_filter_settings.force_outlier_threshold == 30.0
        assert loaded.outlier_filter_settings.min_conformations == 2
