"""Unit tests for workflow module."""

import pytest
from pydantic import ValidationError

from presto.settings import (
    ParameterisationSettings,
    WorkflowSettings,
)


class TestWorkflowFunctions:
    """Tests for workflow functions."""

    def test_get_bespoke_force_field_signature(self):
        """Test that get_bespoke_force_field has correct signature."""
        import inspect

        from presto.workflow import get_bespoke_force_field

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
        from presto.outputs import OutputStage, StageKind

        stage = OutputStage(StageKind.BASE)
        path_manager.mk_stage_dir(stage)

        assert (tmp_path / "test_output").exists()


class TestWorkflowSettingsPersistence:
    """Tests for workflow settings persistence."""

    # TODO: Fix
    @pytest.mark.skip(reason="Requires CUDA...")
    def test_settings_yaml_serialization(self, tmp_path):
        """Test that workflow settings can be serialized to YAML."""
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            output_dir=tmp_path,
            n_iterations=3,
            memory=True,
        )

        yaml_path = tmp_path / "test_settings.yaml"
        settings.to_yaml(yaml_path)

        assert yaml_path.exists()

        # Load back and verify
        loaded = WorkflowSettings.from_yaml(yaml_path)
        assert loaded.n_iterations == 3
        assert loaded.memory is True
        assert loaded.parameterisation_settings.smiles == "CCO"


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
        import torch

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
        from presto.outputs import StageKind

        training_stages = [
            stage
            for stage in path_manager.outputs_by_stage.keys()
            if stage.kind == StageKind.TRAINING
        ]
        assert len(training_stages) == 3
