"""Implements the overall workflow for fitting a bespoke force field."""

import copy
import functools
import pathlib
from collections.abc import Callable

import datasets
import loguru
from descent.train import Trainable
from openff.toolkit import ForceField
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from presto.convert import convert_to_smirnoff

from .analyse import analyse_workflow
from .convert import parameterise
from .data_utils import filter_dataset_outliers
from .outputs import DatasetSplit, OutputStage, OutputType, StageKind
from .sampling_coordinator import sample_ligands
from .settings import SamplingSettings, WorkflowSettings
from .train import _TRAINING_FNS_REGISTRY
from .writers import write_scatter

logger = loguru.logger
console = Console()


def get_bespoke_force_field(
    settings: WorkflowSettings, write_settings: bool = True
) -> ForceField:
    """Fit a bespoke force field.

    This involves:

    - Parameterising a base force field for the target molecule and generating
      specific tagged SMARTS parameters
    - Generating training data (e.g. from high-temperature MD simulations)
    - Optimising the parameters of the force field to reproduce the training data
    - Validating the fitted force field against test data

    Parameters
    ----------
    settings : WorkflowSettings
        The workflow settings to use for fitting the force field.

    write_settings : bool, optional
        Whether to write the settings to a YAML file in the output directory, by default True.

    Returns:
    -------
    ForceField
        The fitted bespoke force field.
    """
    path_manager = settings.get_path_manager()
    # Refuse to run if generated output from an earlier fit is still present.
    path_manager.require_clean()
    stage = OutputStage(StageKind.BASE)
    path_manager.mk_stage_dir(stage)

    if write_settings:
        settings_output_path = path_manager.get_output_path(
            stage, OutputType.WORKFLOW_SETTINGS
        )
        logger.info(f"Writing workflow settings to {settings_output_path}.")
        # Copy the settings and change the output directory to be "." as we save
        # to the output directory already
        output_settings = copy.deepcopy(settings)
        output_settings.output_dir = pathlib.Path(".")
        output_settings.to_yaml(settings_output_path)

    # Parameterise the base force field for all molecules
    off_mols, initial_off_ff, tensor_tops, tensor_ff = parameterise(
        settings.param_settings, device=settings.device_type
    )

    pruned_parameter_configs = {
        p_type: p_config
        for p_type, p_config in settings.training_settings.parameter_configs.items()
        if p_type in tensor_ff.potentials_by_type
    }

    trainable = Trainable(
        tensor_ff,
        pruned_parameter_configs,
        settings.training_settings.attribute_configs,
    )

    trainable_parameters = trainable.to_values().to(settings.device)

    # Get a copy of the initial trainable parameters for regularisation
    initial_parameters = trainable_parameters.clone().detach()

    # Persist the initial force field before sampling so spawned workers
    # reconstruct their own OpenFF/OpenMM state from disk.
    initial_stage = OutputStage(StageKind.INITIAL_STATISTICS)
    path_manager.mk_stage_dir(initial_stage)
    initial_offxml_path = path_manager.get_output_path(initial_stage, OutputType.OFFXML)
    initial_off_ff.to_file(str(initial_offxml_path))

    def sample_datasets(
        stage: OutputStage,
        split: DatasetSplit,
        protocols: list[SamplingSettings],
        offxml_path: pathlib.Path,
        process_dataset: Callable[[int, int, datasets.Dataset], datasets.Dataset]
        | None = None,
    ) -> list[list[datasets.Dataset]]:
        """Sample each dataset in turn, returning the dataset for each molecule."""
        sampled = []
        for k, protocol in enumerate(protocols):
            logger.info(f"Generating {split.value}_{k} ({protocol.sampling_protocol})")
            dataset_stage = OutputStage(stage.kind, stage.index, split, k)
            path_manager.mk_stage_dir(dataset_stage)
            sampled.append(
                sample_ligands(
                    mols=off_mols,
                    offxml_path=offxml_path,
                    device_type=settings.device_type,
                    sampling_settings=protocol,
                    output_paths={
                        output_type: path_manager.get_output_path(
                            dataset_stage, output_type
                        )
                        for output_type in protocol.output_types
                    },
                    dataset_output_paths=[
                        path_manager.get_output_path_for_mol(
                            dataset_stage, OutputType.ENERGIES_AND_FORCES, i
                        )
                        for i in range(len(off_mols))
                    ],
                    n_processes=settings.n_sampling_processes,
                    process_dataset=(
                        None
                        if process_dataset is None
                        else functools.partial(process_dataset, k)
                    ),
                )
            )
        return sampled

    def write_test_scatter(stage: OutputStage, label: str) -> None:
        """Write scatter data for every test dataset and molecule."""
        for k, datasets_for_mols in enumerate(datasets_test):
            dataset_stage = OutputStage(stage.kind, stage.index, DatasetSplit.TEST, k)
            path_manager.mk_stage_dir(dataset_stage)
            for mol_idx, (dataset_test, tensor_top) in enumerate(
                zip(datasets_for_mols, tensor_tops, strict=True)
            ):
                energy_mean, energy_sd, forces_mean, forces_sd = write_scatter(
                    dataset_test,
                    tensor_ff,
                    tensor_top,
                    settings.device,
                    str(
                        path_manager.get_output_path_for_mol(
                            dataset_stage, OutputType.SCATTER, mol_idx
                        )
                    ),
                )
                logger.info(
                    f"{label} test dataset {k} molecule {mol_idx} force field statistics: Energy (Mean/SD): {energy_mean:.3e}/{energy_sd:.3e} kcal/mol, Forces (Mean/SD): {forces_mean:.3e}/{forces_sd:.3e} kcal/mol/Å"
                )

    # Generate the test data for all molecules
    stage = OutputStage(StageKind.TESTING)
    path_manager.mk_stage_dir(stage)
    logger.info("Generating test data")
    datasets_test = sample_datasets(
        stage,
        DatasetSplit.TEST,
        settings.testing_sampling_settings,
        initial_offxml_path,
    )
    dataset_weights_test = [
        protocol.dataset_weight for protocol in settings.testing_sampling_settings
    ]

    # Write out statistics on the initial force field
    stage = OutputStage(StageKind.INITIAL_STATISTICS)
    path_manager.mk_stage_dir(stage)
    write_test_scatter(stage, "Initial")

    off_ff = convert_to_smirnoff(
        trainable.to_force_field(trainable_parameters), base=initial_off_ff
    )
    off_ff.to_file(str(path_manager.get_output_path(stage, OutputType.OFFXML)))

    train_fn = _TRAINING_FNS_REGISTRY[settings.training_settings.optimiser]
    dataset_weights = [
        protocol.dataset_weight for protocol in settings.training_sampling_settings
    ]

    # Train the force field
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )

    previous_datasets = None  # Only None for the first iteration

    def process_training_dataset(
        dataset_idx: int, mol_idx: int, dataset: datasets.Dataset
    ) -> datasets.Dataset:
        if settings.outlier_filter_settings is not None:
            logger.info(
                f"Applying outlier filtering to training dataset {dataset_idx} "
                f"molecule {mol_idx}"
            )
            dataset = filter_dataset_outliers(
                dataset=dataset,
                force_field=tensor_ff,
                topology=tensor_tops[mol_idx],
                settings=settings.outlier_filter_settings,
                device=settings.device,
            )
        if settings.memory and previous_datasets is not None:
            dataset = datasets.combine.concatenate_datasets(
                [previous_datasets[dataset_idx][mol_idx], dataset]
            )
        return dataset

    with progress:
        for iteration in progress.track(
            range(1, settings.n_iterations + 1),  # Start from 1 (0 is untrained)
            description="Iterating the Fit",
        ):
            stage = OutputStage(StageKind.TRAINING, iteration)
            path_manager.mk_stage_dir(stage)

            # Workers load the force field produced by the previous iteration.
            sampling_offxml_path = (
                initial_offxml_path
                if iteration == 1
                else path_manager.get_output_path(
                    OutputStage(StageKind.TRAINING, iteration - 1), OutputType.OFFXML
                )
            )
            datasets_train = sample_datasets(
                stage,
                DatasetSplit.TRAIN,
                settings.training_sampling_settings,
                sampling_offxml_path,
                process_dataset=process_training_dataset,
            )

            train_output_paths = {
                output_type: path_manager.get_output_path(stage, output_type)
                for output_type in settings.training_settings.output_types
            }

            trainable_parameters, trainable = train_fn(
                trainable_parameters=trainable_parameters,
                initial_parameters=initial_parameters,
                trainable=trainable,
                topologies=tensor_tops,
                datasets=datasets_train,
                datasets_test=datasets_test,
                dataset_weights=dataset_weights,
                dataset_weights_test=dataset_weights_test,
                settings=settings.training_settings,
                output_paths=train_output_paths,
                device=settings.device,
            )

            for potential_type in trainable._param_types:
                tensor_ff.potentials_by_type[potential_type].parameters = copy.copy(
                    trainable.to_force_field(trainable_parameters)
                    .potentials_by_type[potential_type]
                    .parameters
                )

            off_ff = convert_to_smirnoff(
                trainable.to_force_field(trainable_parameters), base=initial_off_ff
            )
            off_ff.to_file(str(path_manager.get_output_path(stage, OutputType.OFFXML)))

            write_test_scatter(stage, f"Iteration {iteration}")

            previous_datasets = datasets_train  # Carry into the next iteration

    # Plot
    analyse_workflow(settings)

    return off_ff
