"""Unit tests for outputs module."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bespokefit_smee.outputs import (
    OutputStage,
    OutputType,
    StageKind,
    WorkflowPathManager,
    delete_path,
)
from bespokefit_smee.settings import (
    MMMDSamplingSettings,
    TrainingSettings,
)


class TestOutputType:
    """Tests for OutputType enum."""

    def test_all_output_types_have_values(self):
        """Test that all output types have string values."""
        for output_type in OutputType:
            assert isinstance(output_type.value, str)
            assert len(output_type.value) > 0

    def test_workflow_settings_type(self):
        """Test workflow settings output type."""
        assert OutputType.WORKFLOW_SETTINGS.value == "workflow_settings.yaml"

    def test_offxml_type(self):
        """Test OFFXML output type."""
        assert OutputType.OFFXML.value == "bespoke_ff.offxml"


class TestStageKind:
    """Tests for StageKind enum."""

    def test_stage_kind_values(self):
        """Test that stage kinds have correct values."""
        assert StageKind.BASE.value == ""
        assert StageKind.TESTING.value == "test_data"
        assert StageKind.TRAINING.value == "training_iteration"
        assert StageKind.INITIAL_STATISTICS.value == "initial_statistics"
        assert StageKind.PLOTS.value == "plots"


class TestOutputStage:
    """Tests for OutputStage dataclass."""

    def test_stage_without_index(self):
        """Test stage without index."""
        stage = OutputStage(StageKind.BASE)
        assert stage.kind == StageKind.BASE
        assert stage.index is None
        assert str(stage) == ""

    def test_stage_with_index(self):
        """Test stage with index."""
        stage = OutputStage(StageKind.TRAINING, 1)
        assert stage.kind == StageKind.TRAINING
        assert stage.index == 1
        assert str(stage) == "training_iteration_1"

    def test_testing_stage(self):
        """Test testing stage."""
        stage = OutputStage(StageKind.TESTING)
        assert str(stage) == "test_data"

    def test_initial_statistics_stage(self):
        """Test initial statistics stage."""
        stage = OutputStage(StageKind.INITIAL_STATISTICS)
        assert str(stage) == "initial_statistics"

    def test_frozen_dataclass(self):
        """Test that OutputStage is frozen."""
        stage = OutputStage(StageKind.BASE)
        with pytest.raises(FrozenInstanceError):
            stage.kind = StageKind.TRAINING


class TestWorkflowPathManager:
    """Tests for WorkflowPathManager."""

    @pytest.fixture
    def path_manager(self, tmp_path):
        """Create a path manager for testing."""
        return WorkflowPathManager(
            output_dir=tmp_path,
            n_iterations=2,
            training_settings=TrainingSettings(),
            training_sampling_settings=MMMDSamplingSettings(),
            testing_sampling_settings=MMMDSamplingSettings(),
        )

    def test_init(self, tmp_path):
        """Test initialization."""
        pm = WorkflowPathManager(output_dir=tmp_path)
        assert pm.output_dir == tmp_path
        assert pm.n_iterations == 1

    def test_outputs_by_stage_structure(self, path_manager):
        """Test that outputs_by_stage has correct structure."""
        outputs = path_manager.outputs_by_stage
        assert isinstance(outputs, dict)
        assert OutputStage(StageKind.BASE) in outputs
        assert OutputStage(StageKind.TESTING) in outputs
        assert OutputStage(StageKind.INITIAL_STATISTICS) in outputs
        assert OutputStage(StageKind.TRAINING, 1) in outputs
        assert OutputStage(StageKind.TRAINING, 2) in outputs
        assert OutputStage(StageKind.PLOTS) in outputs

    def test_outputs_by_stage_content(self, path_manager):
        """Test that outputs_by_stage has correct content."""
        outputs = path_manager.outputs_by_stage
        base_outputs = outputs[OutputStage(StageKind.BASE)]
        assert OutputType.WORKFLOW_SETTINGS in base_outputs

        testing_outputs = outputs[OutputStage(StageKind.TESTING)]
        assert OutputType.ENERGIES_AND_FORCES in testing_outputs
        assert OutputType.PDB_TRAJECTORY in testing_outputs

        training_outputs = outputs[OutputStage(StageKind.TRAINING, 1)]
        assert OutputType.OFFXML in training_outputs
        assert OutputType.SCATTER in training_outputs

    def test_get_stage_path(self, path_manager):
        """Test getting stage path."""
        base_path = path_manager.get_stage_path(OutputStage(StageKind.BASE))
        assert base_path == path_manager.output_dir / ""

        training_path = path_manager.get_stage_path(OutputStage(StageKind.TRAINING, 1))
        assert training_path == path_manager.output_dir / "training_iteration_1"

    def test_get_stage_path_unknown_stage(self, path_manager):
        """Test that unknown stage raises error."""
        # Create a stage that's not in outputs_by_stage
        invalid_stage = OutputStage(StageKind.TRAINING, 999)
        with pytest.raises(ValueError, match="Unknown stage"):
            path_manager.get_stage_path(invalid_stage)

    def test_mk_stage_dir(self, path_manager):
        """Test creating stage directory."""
        stage = OutputStage(StageKind.TRAINING, 1)
        path_manager.mk_stage_dir(stage)
        expected_path = path_manager.output_dir / "training_iteration_1"
        assert expected_path.exists()
        assert expected_path.is_dir()

    def test_get_output_path(self, path_manager):
        """Test getting output path."""
        stage = OutputStage(StageKind.TRAINING, 1)
        output_path = path_manager.get_output_path(stage, OutputType.OFFXML)
        expected = (
            path_manager.output_dir / "training_iteration_1" / "bespoke_ff.offxml"
        )
        assert output_path == expected

    def test_get_output_path_unknown_stage(self, path_manager):
        """Test that unknown stage raises error."""
        invalid_stage = OutputStage(StageKind.TRAINING, 999)
        with pytest.raises(ValueError, match="Unknown stage"):
            path_manager.get_output_path(invalid_stage, OutputType.OFFXML)

    def test_get_output_path_unexpected_output_type(self, path_manager):
        """Test that unexpected output type raises error."""
        stage = OutputStage(StageKind.BASE)
        with pytest.raises(ValueError, match="not expected in stage"):
            path_manager.get_output_path(stage, OutputType.OFFXML)

    def test_get_all_output_paths(self, path_manager):
        """Test getting all output paths."""
        all_paths = path_manager.get_all_output_paths(only_if_exists=False)
        assert isinstance(all_paths, dict)
        assert len(all_paths) > 0

        # Check structure
        for stage, outputs in all_paths.items():
            assert isinstance(stage, OutputStage)
            assert isinstance(outputs, dict)
            for output_type, path in outputs.items():
                assert isinstance(output_type, OutputType)
                assert isinstance(path, Path)

    def test_get_all_output_paths_only_existing(self, path_manager, tmp_path):
        """Test getting only existing output paths."""
        # Create some output files
        stage = OutputStage(StageKind.BASE)
        path_manager.mk_stage_dir(stage)
        output_path = path_manager.get_output_path(stage, OutputType.WORKFLOW_SETTINGS)
        output_path.write_text("test")

        all_paths = path_manager.get_all_output_paths(only_if_exists=True)
        assert OutputStage(StageKind.BASE) in all_paths
        assert OutputType.WORKFLOW_SETTINGS in all_paths[OutputStage(StageKind.BASE)]

    def test_get_all_output_paths_by_output_type(self, path_manager):
        """Test getting all output paths organized by output type."""
        paths_by_type = path_manager.get_all_output_paths_by_output_type(
            only_if_exists=False
        )
        assert isinstance(paths_by_type, dict)
        assert OutputType.OFFXML in paths_by_type
        assert isinstance(paths_by_type[OutputType.OFFXML], list)

        # Should have one OFFXML for initial stats + 2 for training iterations
        assert len(paths_by_type[OutputType.OFFXML]) == 3

    def test_clean_removes_files(self, path_manager):
        """Test that clean removes output files."""
        # Create some output files
        stage = OutputStage(StageKind.TRAINING, 1)
        path_manager.mk_stage_dir(stage)

        offxml_path = path_manager.get_output_path(stage, OutputType.OFFXML)
        offxml_path.write_text("test")

        scatter_path = path_manager.get_output_path(stage, OutputType.SCATTER)
        scatter_path.write_text("test")

        assert offxml_path.exists()
        assert scatter_path.exists()

        path_manager.clean()

        assert not offxml_path.exists()
        assert not scatter_path.exists()

    def test_clean_preserves_workflow_settings(self, path_manager):
        """Test that clean preserves workflow settings."""
        stage = OutputStage(StageKind.BASE)
        path_manager.mk_stage_dir(stage)

        settings_path = path_manager.get_output_path(
            stage, OutputType.WORKFLOW_SETTINGS
        )
        settings_path.write_text("test")

        path_manager.clean()

        assert settings_path.exists()

    def test_clean_removes_empty_directories(self, path_manager):
        """Test that clean removes empty stage directories."""
        stage = OutputStage(StageKind.TRAINING, 1)
        path_manager.mk_stage_dir(stage)
        stage_path = path_manager.get_stage_path(stage)

        assert stage_path.exists()

        path_manager.clean()

        assert not stage_path.exists()


class TestDeletePath:
    """Tests for delete_path function."""

    def test_delete_file(self, tmp_path):
        """Test deleting a file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")
        assert file_path.exists()

        delete_path(file_path)
        assert not file_path.exists()

    def test_delete_empty_directory_non_recursive(self, tmp_path):
        """Test deleting an empty directory non-recursively."""
        dir_path = tmp_path / "empty_dir"
        dir_path.mkdir()
        assert dir_path.exists()

        delete_path(dir_path, recursive=False)
        assert not dir_path.exists()

    def test_delete_non_empty_directory_non_recursive(self, tmp_path):
        """Test that non-empty directory is not deleted non-recursively."""
        dir_path = tmp_path / "non_empty_dir"
        dir_path.mkdir()
        (dir_path / "file.txt").write_text("test")

        with pytest.raises(OSError, match="Directory not empty"):
            delete_path(dir_path, recursive=False)

    def test_delete_directory_recursive(self, tmp_path):
        """Test deleting a directory recursively."""
        dir_path = tmp_path / "test_dir"
        dir_path.mkdir()
        (dir_path / "file1.txt").write_text("test")
        subdir = dir_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("test")

        delete_path(dir_path, recursive=True)
        assert not dir_path.exists()

    def test_delete_nonexistent_path(self, tmp_path):
        """Test that deleting nonexistent path doesn't raise error."""
        nonexistent = tmp_path / "nonexistent.txt"
        delete_path(nonexistent)  # Should not raise

    def test_delete_nested_structure(self, tmp_path):
        """Test deleting nested directory structure."""
        root = tmp_path / "root"
        root.mkdir()
        (root / "file1.txt").write_text("test")

        level1 = root / "level1"
        level1.mkdir()
        (level1 / "file2.txt").write_text("test")

        level2 = level1 / "level2"
        level2.mkdir()
        (level2 / "file3.txt").write_text("test")

        delete_path(root, recursive=True)
        assert not root.exists()
