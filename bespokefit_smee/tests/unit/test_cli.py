"""Unit tests for CLI module."""

import subprocess

from bespokefit_smee._cli import (
    Analyse,
    Clean,
    TrainFromYAML,
    WriteDefaultYAML,
)
from bespokefit_smee.settings import (
    _DEFAULT_SMILES_PLACEHOLDER,
    ParameterisationSettings,
    WorkflowSettings,
)


class TestWriteDefaultYAML:
    """Tests for WriteDefaultYAML command."""

    def test_default_file_name(self):
        """Test that default file name is set."""
        cmd = WriteDefaultYAML()
        assert cmd.file_name is not None

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

        # Modify the SMILES to a valid one
        content = yaml_path.read_text()
        content = content.replace(_DEFAULT_SMILES_PLACEHOLDER, "CCO")
        yaml_path.write_text(content)

        # Should be able to load now
        settings = WorkflowSettings.from_yaml(yaml_path)
        assert settings.parameterisation_settings.smiles == "CCO"


class TestTrainFromYAML:
    """Tests for TrainFromYAML command."""

    def test_default_settings_yaml_path(self):
        """Test that default settings YAML path is set."""
        cmd = TrainFromYAML()
        assert cmd.settings_yaml is not None


class TestClean:
    """Tests for Clean command."""

    def test_default_settings_yaml_path(self):
        """Test that default settings YAML path is set."""
        cmd = Clean()
        assert cmd.settings_yaml is not None

    def test_cli_cmd_cleans_output(self, tmp_path, monkeypatch):
        """Test that cli_cmd cleans output directory."""
        monkeypatch.chdir(tmp_path)

        # Create a settings file and some output
        settings = WorkflowSettings(
            parameterisation_settings=ParameterisationSettings(smiles="CCO"),
            output_dir=tmp_path,
        )
        yaml_path = tmp_path / "settings.yaml"
        settings.to_yaml(yaml_path)

        # Create some fake output
        path_manager = settings.get_path_manager()
        from bespokefit_smee.outputs import OutputStage, OutputType, StageKind

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

    def test_default_settings_yaml_path(self):
        """Test that default settings YAML path is set."""
        cmd = Analyse()
        assert cmd.settings_yaml is not None


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            ["bespokefit_smee", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "bespokefit_smee" in result.stdout.lower()

    def test_write_default_yaml_command(self, tmp_path):
        """Test write-default-yaml command."""
        yaml_path = tmp_path / "test.yaml"
        result = subprocess.run(
            ["bespokefit_smee", "write-default-yaml", str(yaml_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert yaml_path.exists()

    def test_train_help(self):
        """Test that train --help works."""
        result = subprocess.run(
            ["bespokefit_smee", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "smiles" in result.stdout.lower()
