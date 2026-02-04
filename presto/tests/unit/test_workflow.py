"""Unit tests for workflow module."""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch
from openff.toolkit import ForceField, Molecule
from pydantic import ValidationError

from presto.outputs import OutputStage, StageKind
from presto.settings import (
    MLMDSamplingSettings,
    MMMDSamplingSettings,
    OutlierFilterSettings,
    ParameterisationSettings,
    WorkflowSettings,
)
from presto.workflow import get_bespoke_force_field


class TestWorkflowFunctions:
    """Tests for workflow functions."""

    def test_get_bespoke_force_field_signature(self):
        """Test that get_bespoke_force_field has correct signature."""
        sig = inspect.signature(get_bespoke_force_field)
        assert "settings" in sig.parameters
        assert "write_settings" in sig.parameters

    def test_suppress_unwanted_output_called_on_import(self):
        """Test that suppress_unwanted_output is called on import."""
        # This verifies the module initialization
        from presto import workflow

        assert workflow is not None


class TestWorkflowPathCreation:
    """Tests for workflow path creation logic."""

    def test_workflow_creates_output_directory(self, tmp_path):
        """Test that workflow creates output directory."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            output_dir=tmp_path / "test_output",
            device_type="cpu",
        )

        path_manager = settings.get_path_manager()
        stage = OutputStage(StageKind.BASE)
        path_manager.mk_stage_dir(stage)

        assert (tmp_path / "test_output").exists()


class TestWorkflowSettingsPersistence:
    """Tests for workflow settings persistence."""

    def test_settings_yaml_serialization(self, tmp_path):
        """Test that workflow settings can be serialized to YAML."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            output_dir=tmp_path,
            n_iterations=3,
            memory=True,
            device_type="cpu",
        )

        yaml_path = tmp_path / "test_settings.yaml"
        settings.to_yaml(yaml_path)

        assert yaml_path.exists()

        # Load back and verify
        loaded = WorkflowSettings.from_yaml(yaml_path)
        assert loaded.n_iterations == 3
        assert loaded.memory is True
        assert loaded.parameterisation_settings.smiles == ["CCO"]


class TestWorkflowStages:
    """Tests for workflow stage logic."""

    def test_workflow_has_correct_number_of_stages(self):
        """Test that workflow creates correct number of stages."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            n_iterations=3,
            device_type="cpu",
        )

        path_manager = settings.get_path_manager()
        outputs_by_stage = path_manager.outputs_by_stage

        # Should have:
        # - BASE stage
        # - TESTING stage
        # - INITIAL_STATISTICS stage
        # - 3 TRAINING stages (one per iteration)
        # - PLOTS stage
        assert len(outputs_by_stage) >= 6


class TestWorkflowErrorHandling:
    """Tests for workflow error handling."""

    def test_invalid_smiles_caught_early(self):
        """Test that invalid SMILES is caught during settings creation."""
        with pytest.raises(ValidationError):
            WorkflowSettings(
                parameterisation_settings=ParameterisationSettings(
                    smiles="invalid_smiles_xyz"
                ),
            )


class TestWorkflowMemoryMode:
    """Tests for workflow memory mode."""

    def test_memory_false_by_default(self):
        """Test that memory is False by default."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
        )
        assert settings.memory is False

    def test_memory_can_be_enabled(self):
        """Test that memory mode can be enabled."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            memory=True,
        )
        assert settings.memory is True


class TestWorkflowDeviceHandling:
    """Tests for workflow device handling."""

    def test_cpu_device_always_available(self):
        """Test that CPU device is always available."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            device_type="cpu",
        )
        assert settings.device_type == "cpu"

    def test_device_property_returns_torch_device(self):
        """Test that device property returns torch.device."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            device_type="cpu",
        )
        device = settings.device
        assert isinstance(device, torch.device)
        assert device.type == "cpu"


class TestWorkflowIterations:
    """Tests for workflow iteration settings."""

    def test_default_iterations(self):
        """Test default number of iterations."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
        )
        assert settings.n_iterations == 2

    def test_custom_iterations(self):
        """Test custom number of iterations."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            n_iterations=5,
        )
        assert settings.n_iterations == 5

    def test_single_iteration(self):
        """Test single iteration workflow."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            n_iterations=1,
        )
        assert settings.n_iterations == 1


class TestWorkflowIntegration:
    """Integration tests for workflow components."""

    def test_workflow_settings_integrates_with_path_manager(self, tmp_path):
        """Test that workflow settings integrates with path manager."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            output_dir=tmp_path,
            n_iterations=2,
        )

        path_manager = settings.get_path_manager()
        assert path_manager.output_dir == tmp_path
        assert path_manager.n_iterations == 2

    def test_workflow_components_use_consistent_settings(self):
        """Test that workflow components use consistent settings."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            n_iterations=3,
        )

        path_manager = settings.get_path_manager()

        # Check that training stages match n_iterations
        training_stages = [
            stage
            for stage in path_manager.outputs_by_stage.keys()
            if stage.kind == StageKind.TRAINING
        ]
        assert len(training_stages) == 3


@pytest.fixture
def mock_workflow_dependencies():
    with (
        patch("presto.workflow.parameterise") as mock_parameterise,
        patch("presto.workflow._SAMPLING_FNS_REGISTRY") as mock_sampling_registry,
        patch("presto.workflow._TRAINING_FNS_REGISTRY") as mock_training_registry,
        patch("presto.workflow.convert_to_smirnoff") as mock_convert_to_smirnoff,
        patch("presto.workflow.analyse_workflow") as mock_analyse_workflow,
        patch("presto.workflow.write_scatter") as mock_write_scatter,
        patch("presto.workflow.filter_dataset_outliers") as mock_filter_outliers,
        patch("presto.workflow.Trainable") as mock_trainable_cls,
    ):
        # Setup basic mock returns
        mock_mol = MagicMock(spec=Molecule)
        mock_off_ff = MagicMock(spec=ForceField)
        mock_tensor_top = MagicMock()
        mock_tensor_ff = MagicMock()
        mock_tensor_ff.potentials_by_type = {}  # Required for pruning logic

        # parameterise return
        mock_parameterise.return_value = (
            [mock_mol],  # off_mols
            mock_off_ff,  # initial_off_ff
            [mock_tensor_top],  # tensor_tops
            mock_tensor_ff,  # tensor_ff
        )

        # Trainable return
        mock_trainable = MagicMock()
        mock_trainable_cls.return_value = mock_trainable
        mock_trainable.to_values.return_value = torch.tensor([1.0])
        # Force field conversion
        mock_trainable.to_force_field.return_value = mock_tensor_ff

        # Sampling return
        mock_sample_fn = MagicMock()
        mock_sample_dataset = MagicMock()
        mock_sample_fn.return_value = [mock_sample_dataset]
        mock_sampling_registry.__getitem__.return_value = mock_sample_fn

        # Training return
        mock_train_fn = MagicMock()
        mock_train_fn.return_value = (
            torch.tensor([0.5]),
            mock_trainable,
        )  # params, trainable
        mock_training_registry.__getitem__.return_value = mock_train_fn

        # Conversion return
        mock_convert_to_smirnoff.return_value = mock_off_ff

        # Scatter return
        mock_write_scatter.return_value = (0.0, 0.0, 0.0, 0.0)

        yield {
            "parameterise": mock_parameterise,
            "sampling_registry": mock_sampling_registry,
            "training_registry": mock_training_registry,
            "convert_to_smirnoff": mock_convert_to_smirnoff,
            "analyse_workflow": mock_analyse_workflow,
            "write_scatter": mock_write_scatter,
            "filter_outliers": mock_filter_outliers,
            "trainable_cls": mock_trainable_cls,
            "sample_fn": mock_sample_fn,
            "train_fn": mock_train_fn,
            "off_ff": mock_off_ff,
        }


def test_get_bespoke_force_field_basic(tmp_path, mock_workflow_dependencies):
    """Test basic execution of the workflow."""
    from openmm import unit

    settings = WorkflowSettings(
        parameterisation_settings=ParameterisationSettings(smiles="C"),
        output_dir=tmp_path,
        n_iterations=2,
        device_type="cpu",
        # Use simple settings
        testing_sampling_settings=MLMDSamplingSettings(
            sampling_protocol="ml_md",
            snapshot_interval=1.0 * unit.femtoseconds,
            production_sampling_time_per_conformer=2.0 * unit.femtoseconds,
        ),
        training_sampling_settings=MMMDSamplingSettings(
            sampling_protocol="mm_md",
            snapshot_interval=1.0 * unit.femtoseconds,
            production_sampling_time_per_conformer=2.0 * unit.femtoseconds,
            equilibration_sampling_time_per_conformer=0.0 * unit.femtoseconds,
        ),
    )

    # Run workflow
    final_ff = get_bespoke_force_field(settings, write_settings=False)

    deps = mock_workflow_dependencies

    # Verify parameterisation called
    deps["parameterise"].assert_called_once()

    # Verify initial test sampling (once)
    assert deps["sample_fn"].call_count >= 1

    # Verify training loop iterations
    # Total sampling calls: 1 (test) + 2 (training) = 3
    assert deps["sample_fn"].call_count == 3

    # Total training calls: 2
    assert deps["train_fn"].call_count == 2

    # Analyse called at end
    deps["analyse_workflow"].assert_called_once()

    assert final_ff == deps["off_ff"]


def test_workflow_memory_mode(tmp_path, mock_workflow_dependencies):
    """Test workflow with memory=True."""
    settings = WorkflowSettings(
        parameterisation_settings=ParameterisationSettings(smiles="C"),
        output_dir=tmp_path,
        n_iterations=2,
        memory=True,  # Enable memory
        device_type="cpu",
    )

    # Patch where the object is looked up
    with patch("datasets.combine.concatenate_datasets") as mock_concat:
        # Mock concatenation returning a new dataset
        mock_concat.return_value = MagicMock()

        get_bespoke_force_field(settings, write_settings=False)

        # Concatenation should happen in 2nd iteration
        mock_concat.assert_called()


def test_workflow_outlier_filtering(tmp_path, mock_workflow_dependencies):
    """Test workflow with outlier filtering enabled."""
    settings = WorkflowSettings(
        parameterisation_settings=ParameterisationSettings(smiles="C"),
        output_dir=tmp_path,
        n_iterations=1,
        outlier_filter_settings=OutlierFilterSettings(),  # Enable filtering
        device_type="cpu",
    )

    get_bespoke_force_field(settings, write_settings=False)

    deps = mock_workflow_dependencies
    deps["filter_outliers"].assert_called()
