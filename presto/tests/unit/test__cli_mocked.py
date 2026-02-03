from pathlib import Path
from unittest.mock import MagicMock, patch

from presto._cli import Analyse, Clean, TrainFromYAML, WriteDefaultYAML


def test_cli_train_from_yaml_mocked():
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


def test_cli_clean_mocked():
    with patch("presto._cli.WorkflowSettings.from_yaml") as mock_from_yaml:
        mock_settings = MagicMock()
        mock_from_yaml.return_value = mock_settings

        cmd = Clean(settings_yaml=Path("fake.yaml"))
        cmd.cli_cmd()

        mock_from_yaml.assert_called_once_with(Path("fake.yaml"))
        mock_settings.get_path_manager.return_value.clean.assert_called_once()


def test_cli_analyse_mocked():
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


def test_cli_write_default_yaml_mocked(tmp_path):
    out_path = tmp_path / "default.yaml"
    cmd = WriteDefaultYAML(file_name=out_path)
    # This one actually runs logic, so we can just check if it works
    cmd.cli_cmd()
    assert out_path.exists()


def test_cli_train_from_args_mocked():
    from presto._cli import TrainFromCli
    from presto.settings import ParameterisationSettings

    # TrainFromCli inherits from WorkflowSettings
    # We can mock get_bespoke_force_field
    with patch("presto._cli.get_bespoke_force_field") as mock_get_ff:
        cmd = TrainFromCli(
            parameterisation_settings=ParameterisationSettings(smiles="CCO")
        )
        # Note: CliApp.run_subcommand calls cmd.train.cli_cmd()
        cmd.cli_cmd()
        mock_get_ff.assert_called_once()
