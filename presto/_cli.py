"""A CLI for presto."""

from pathlib import Path
from typing import Any, Self

import loguru
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import CliApp, CliPositionalArg, CliSubCommand
from rich.logging import RichHandler

from . import __version__
from .analyse import analyse_workflow
from .outputs import OutputStage, OutputType, StageKind, WorkflowPathManager
from ._exceptions import InvalidSettingsError
from .settings import (
    _DEFAULT_INPUT_PLACEHOLDER,
    ParamSettings,
    SamplingSettings,
    WorkflowSettings,
    get_default_testing_sampling_settings,
    get_default_training_sampling_settings,
)
from .utils._suppress_output import suppress_unwanted_output
from .workflow import get_bespoke_force_field

logger = loguru.logger

_DEFAULT_WORKFLOW_SETTINGS_PATH = WorkflowPathManager(Path(".")).get_output_path(
    OutputStage(StageKind.BASE), OutputType.WORKFLOW_SETTINGS
)


def setup_logging_for_cli(log_level: str = "INFO") -> None:
    """Set up logging for the CLI."""
    logger.remove()
    logger.add(
        RichHandler(show_time=True, markup=True),
        format="{message}",
        level=log_level.upper(),
    )


class Version(BaseModel):
    """Print the version of presto and exit.

    Example: ``presto version``
    """

    def cli_cmd(self) -> None:
        print(f"presto {__version__}")


class TrainFromCli(WorkflowSettings):
    """Run the training process with command line arguments.

    Example: ``presto train --param-settings.molecules "CCO"``

    All ``WorkflowSettings`` fields can be set with dotted overrides like
    ``--training-sampling-settings.mlp-settings.ml-potential aimnet2``.

    The CLI is intended for simple fits, so it takes a single training and a single
    testing sampling protocol. Use ``presto train-from-yaml`` or the Python API to
    fit to multiple datasets.
    """

    # Single protocols rather than lists, so pydantic-settings generates dotted
    # overrides for their fields.
    training_sampling_settings: SamplingSettings = Field(  # type: ignore[assignment]
        default_factory=get_default_training_sampling_settings,
        description="Settings for sampling for generating the training data "
        "(usually molecular dynamics)",
        discriminator="sampling_protocol",
    )

    testing_sampling_settings: SamplingSettings = Field(  # type: ignore[assignment]
        default_factory=get_default_testing_sampling_settings,
        description="Settings for sampling for generating the testing data "
        "(usually molecular dynamics)",
        discriminator="sampling_protocol",
    )

    @field_validator(
        "training_sampling_settings", "testing_sampling_settings", mode="before"
    )
    @classmethod
    def wrap_single_sampling_settings(cls, value: Any) -> Any:
        """Reject more than one protocol, replacing the list-wrapping validator."""
        if not isinstance(value, list):
            return value
        if len(value) != 1:
            raise InvalidSettingsError(
                "The `presto train` CLI supports one training and one testing "
                "sampling protocol. Use `presto train-from-yaml` or the Python API "
                "to fit to multiple datasets."
            )
        return value[0]

    # These validators expect lists of protocols, so they run instead on the
    # WorkflowSettings built in cli_cmd.
    @model_validator(mode="after")
    def validate_starting_conformers_match_molecules(self) -> Self:
        """Defer to the WorkflowSettings built in cli_cmd."""
        return self

    @model_validator(mode="after")
    def validate_pre_computed_dataset_counts(self) -> Self:
        """Defer to the WorkflowSettings built in cli_cmd."""
        return self

    @model_validator(mode="after")
    def validate_memory_without_pre_computed(self) -> Self:
        """Defer to the WorkflowSettings built in cli_cmd."""
        return self

    @model_validator(mode="after")
    def validate_molecule_weight_counts(self) -> Self:
        """Defer to the WorkflowSettings built in cli_cmd."""
        return self

    def to_workflow_settings(self) -> WorkflowSettings:
        """Return the equivalent WorkflowSettings, with one-protocol lists."""
        return WorkflowSettings(
            **{name: getattr(self, name) for name in WorkflowSettings.model_fields}
        )

    def cli_cmd(self) -> None:
        get_bespoke_force_field(self.to_workflow_settings(), write_settings=True)


class TrainFromYAML(BaseModel):
    """Run the training process with arguments read from a YAML file.

    Example: ``presto train-from-yaml workflow_settings.yaml``

    Generate a starting YAML with ``presto write-default-yaml``.
    """

    settings_yaml: CliPositionalArg[Path] = Field(
        _DEFAULT_WORKFLOW_SETTINGS_PATH,
        description="Path to the YAML settings file for training",
    )

    def cli_cmd(self) -> None:
        logger.info(f"Running presto with settings from {self.settings_yaml}")
        settings = WorkflowSettings.from_yaml(Path(self.settings_yaml))
        # No need to write a settings file if we're reading from one already
        get_bespoke_force_field(settings, write_settings=False)


class WriteDefaultYAML(BaseModel):
    """Write a default YAML file for the training settings.

    Example: ``presto write-default-yaml workflow_settings.yaml``

    The written file contains the placeholder ``CHANGEME`` under
    ``param_settings.molecules`` — edit it before running ``train-from-yaml``.
    """

    file_name: CliPositionalArg[Path] = Field(
        _DEFAULT_WORKFLOW_SETTINGS_PATH,
        description="The name of the default YAML settings file to write",
    )

    def cli_cmd(self) -> None:
        logger.info(f"Writing default YAML settings to {self.file_name}.")
        # Temporarily set a valid molecule input to pass validation, then overwrite it
        # with a placeholder value before writing the file
        param_settings = ParamSettings(molecule_input_type="smiles", molecules="O")
        WorkflowSettings(param_settings=param_settings).to_yaml(
            self.file_name,
            overwrite={"param_settings": {"molecules": [_DEFAULT_INPUT_PLACEHOLDER]}},
        )


class Clean(BaseModel):
    """Clean the output directory by removing generated files.

    Example: ``presto clean workflow_settings.yaml``

    Removes the stage directories the settings predict, but keeps the settings YAML
    itself. Refuses, without deleting anything, if one of those directories holds a
    path the settings do not predict.
    """

    settings_yaml: CliPositionalArg[Path] = Field(
        _DEFAULT_WORKFLOW_SETTINGS_PATH,
        description="Path to the YAML settings used for training",
    )

    def cli_cmd(self) -> None:
        logger.info(
            f"Cleaning output directory with settings from {self.settings_yaml}"
        )
        settings = WorkflowSettings.from_yaml(Path(self.settings_yaml))
        settings.get_path_manager().clean()


class Analyse(BaseModel):
    """Analyse the training data and results.

    Example: ``presto analyse workflow_settings.yaml``

    Regenerates the diagnostic plots under ``<output_dir>/plots/`` from existing
    training output without re-running sampling or training.
    """

    settings_yaml: CliPositionalArg[Path] = Field(
        _DEFAULT_WORKFLOW_SETTINGS_PATH,
        description="Path to the YAML settings used for training",
    )

    def cli_cmd(self) -> None:
        logger.info(f"Analysing training data with settings from {self.settings_yaml}")
        training_settings = WorkflowSettings.from_yaml(Path(self.settings_yaml))
        analyse_workflow(training_settings)


class CLI(BaseModel):
    """presto: parameterise a bespoke force field from high-temperature MD data."""

    version: CliSubCommand[Version] = Field(
        description="Print the version of presto and exit.",
    )

    train: CliSubCommand[TrainFromCli] = Field(
        description="Train a bespoke force field from high-temperature MD data",
    )

    train_from_yaml: CliSubCommand[TrainFromYAML] = Field(
        description="The same as 'train', but arguments are read from a YAML file",
    )

    write_default_yaml: CliSubCommand[WriteDefaultYAML] = Field(
        description="Write a default YAML settings file for the training process",
    )

    clean: CliSubCommand[Clean] = Field(
        description="Clean the output directory by removing generated files",
    )

    analyse: CliSubCommand[Analyse] = Field(
        description="Analyse the training data and results",
    )

    def cli_cmd(self) -> None:
        suppress_unwanted_output()
        setup_logging_for_cli()

        CliApp.run_subcommand(self)


def run_cli() -> None:
    suppress_unwanted_output()
    CliApp.run(CLI)
