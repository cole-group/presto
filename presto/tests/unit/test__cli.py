"""Unit tests for CLI module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from presto._cli import (
    Analyse,
    Clean,
    TrainFromCli,
    TrainFromYAML,
    WriteDefaultYAML,
)
from presto.settings import (
    _DEFAULT_SMILES_PLACEHOLDER,
    ParameterisationSettings,
    WorkflowSettings,
)


class TestWriteDefaultYAML:
    """Tests for WriteDefaultYAML command."""

    def test_cli_cmd_creates_file(self, tmp_path, monkeypatch):
        """Test that cli_cmd creates a YAML file."""
        monkeypatch.chdir(tmp_path)
        yaml_path = tmp_path / "test_settings.yaml"

        cmd = WriteDefaultYAML(file_name=yaml_path)
        cmd.cli_cmd()

        assert yaml_path.exists()

    def test_written_yaml_contains_placeholder(self, tmp_path, monkeypatch):
        """Test that written YAML contains placeholder SMILES."""
        monkeypatch.chdir(tmp_path)
        yaml_path = tmp_path / "test_settings.yaml"

        cmd = WriteDefaultYAML(file_name=yaml_path)
        cmd.cli_cmd()

        content = yaml_path.read_text()
        assert _DEFAULT_SMILES_PLACEHOLDER in content

    def test_written_yaml_can_be_loaded(self, tmp_path, monkeypatch):
        """Test that written YAML can be loaded (after fixing SMILES)."""
        monkeypatch.chdir(tmp_path)
        yaml_path = tmp_path / "test_settings.yaml"

        cmd = WriteDefaultYAML(file_name=yaml_path)
        cmd.cli_cmd()

        # Modify the SMILES to a valid one and set device_type to cpu
        content = yaml_path.read_text()
        content = content.replace(_DEFAULT_SMILES_PLACEHOLDER, "CCO")
        content = content.replace("device_type: cuda", "device_type: cpu")
        yaml_path.write_text(content)

        # Should be able to load now
        settings = WorkflowSettings.from_yaml(yaml_path)
        assert settings.parameterisation_settings.smiles == ["CCO"]


class TestTrainFromYAML:
    """Tests for TrainFromYAML command."""

    def test_train_from_yaml_calls_workflow(self):
        """Test that TrainFromYAML loads settings and calls get_bespoke_force_field."""
        with (
            patch("presto._cli.WorkflowSettings.from_yaml") as mock_from_yaml,
            patch("presto._cli.get_bespoke_force_field") as mock_get_ff,
        ):
            mock_settings = MagicMock()
            mock_from_yaml.return_value = mock_settings

            cmd = TrainFromYAML(settings_yaml=Path("fake.yaml"))
            cmd.cli_cmd()

            mock_from_yaml.assert_called_once_with(Path("fake.yaml"))
            mock_get_ff.assert_called_once_with(mock_settings, write_settings=False)


class TestTrainFromCli:
    """Tests for TrainFromCli command."""

    def test_train_from_args_calls_workflow(self):
        """Test that TrainFromCli calls get_bespoke_force_field with settings."""
        with patch("presto._cli.get_bespoke_force_field") as mock_get_ff:
            cmd = TrainFromCli(
                parameterisation_settings=ParameterisationSettings(smiles="CCO"),
                device_type="cpu",
            )
            cmd.cli_cmd()
            mock_get_ff.assert_called_once()


class TestClean:
    """Tests for Clean command."""

    def test_clean_calls_path_manager(self):
        """Test that Clean loads settings and calls path manager clean."""
        with patch("presto._cli.WorkflowSettings.from_yaml") as mock_from_yaml:
            mock_settings = MagicMock()
            mock_from_yaml.return_value = mock_settings

            cmd = Clean(settings_yaml=Path("fake.yaml"))
            cmd.cli_cmd()

            mock_from_yaml.assert_called_once_with(Path("fake.yaml"))
            mock_settings.get_path_manager.return_value.clean.assert_called_once()

    def test_cli_cmd_cleans_output(self, tmp_path, monkeypatch):
        """Test that cli_cmd cleans output directory."""
        monkeypatch.chdir(tmp_path)

        # Create a settings file and some output
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            output_dir=tmp_path,
            device_type="cpu",
        )
        yaml_path = tmp_path / "settings.yaml"
        settings.to_yaml(yaml_path)

        # Create some fake output
        path_manager = settings.get_path_manager()
        from presto.outputs import OutputStage, OutputType, StageKind

        stage = OutputStage(StageKind.TRAINING, 1)
        path_manager.mk_stage_dir(stage)
        output_file = path_manager.get_output_path(stage, OutputType.OFFXML)
        output_file.write_text("test")

        assert output_file.exists()

        # Run clean
        cmd = Clean(settings_yaml=yaml_path)
        cmd.cli_cmd()

        # Output should be cleaned
        assert not output_file.exists()


class TestAnalyse:
    """Tests for Analyse command."""

    def test_analyse_calls_analyse_workflow(self):
        """Test that Analyse loads settings and calls analyse_workflow."""
        with (
            patch("presto._cli.WorkflowSettings.from_yaml") as mock_from_yaml,
            patch("presto._cli.analyse_workflow") as mock_analyse,
        ):
            mock_settings = MagicMock()
            mock_from_yaml.return_value = mock_settings

            cmd = Analyse(settings_yaml=Path("fake.yaml"))
            cmd.cli_cmd()

            mock_from_yaml.assert_called_once_with(Path("fake.yaml"))
            mock_analyse.assert_called_once_with(mock_settings)


class TestCLISubprocess:
    """Test the CLI via subprocess calls."""

    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            ["presto", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "presto" in result.stdout.lower()

    def test_write_default_yaml_command(self, tmp_path):
        """Test write-default-yaml command."""
        yaml_path = tmp_path / "test.yaml"
        result = subprocess.run(
            ["presto", "write-default-yaml", str(yaml_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert yaml_path.exists()

    def test_train_help(self):
        """Test that train --help works."""
        result = subprocess.run(
            ["presto", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "smiles" in result.stdout.lower()
