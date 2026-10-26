"""Implements the overall workflow for fitting a bespoke force field."""

import copy
import functools
import pathlib
from collections.abc import Callable

import datasets
import loguru
import smee
import torch
from descent.train import Trainable
from openff.toolkit import ForceField, Molecule
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
from .outputs import (
    DatasetSplit,
    OutputStage,
    OutputType,
    StageKind,
    WorkflowPathManager,
)
from .sampling_coordinator import sample_ligands
from .settings import SamplingSettings, WorkflowSettings
from .train import _TRAINING_FNS_REGISTRY
from .utils.typing import TorchDevice
from .writers import write_scatter

logger = loguru.logger
console = Console()


def sample_datasets(
    *,
    stage: OutputStage,
    split: DatasetSplit,
    protocols: list[SamplingSettings],
    mols: list[Molecule],
    offxml_path: pathlib.Path,
    path_manager: WorkflowPathManager,
    device_type: TorchDevice,
    n_processes: int,
    process_dataset: Callable[[int, int, datasets.Dataset], datasets.Dataset]
    | None = None,
) -> list[list[datasets.Dataset]]:
    """Sample each dataset in turn, for every molecule.

    Protocols are sampled one after another, each into its own
    ``<stage>/<split>_<k>`` subdirectory, while the molecules for each protocol are
    sampled in parallel across ``n_processes`` workers.

    Parameters
    ----------
    stage : OutputStage
        The top-level stage to sample into, e.g. ``training_iteration_1``.

    split : DatasetSplit
        Whether these are training or testing datasets, which names their
        subdirectories within ``stage``.

    protocols : list[SamplingSettings]
        The sampling protocol for each dataset, in order. Dataset ``k`` is sampled
        with ``protocols[k]``.

    mols : list[Molecule]
        The molecules to sample.

    offxml_path : pathlib.Path
        The force field the sampling workers load.

    path_manager : WorkflowPathManager
        Gives the output paths for each dataset.

    device_type : TorchDevice
        The device to sample on.

    n_processes : int
        The number of worker processes to sample the molecules with.

    process_dataset : Callable[[int, int, datasets.Dataset], datasets.Dataset] | None, optional
        Applied to each sampled dataset with the dataset index, molecule index and
        dataset, e.g. to filter outliers or add data from the previous iteration.
        By default None, which leaves the datasets unchanged.

    Returns:
    -------
    list[list[datasets.Dataset]]
        One list per dataset, holding that dataset's data for each molecule in
        order.
    """
    sampled = []
    for k, protocol in enumerate(protocols):
        logger.info(f"Generating {split.value}_{k} ({protocol.sampling_protocol})")
        dataset_stage = OutputStage(stage.kind, stage.index, split, k)
        path_manager.mk_stage_dir(dataset_stage)
        sampled.append(
            sample_ligands(
                mols=mols,
                offxml_path=offxml_path,
                device_type=device_type,
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
                    for i in range(len(mols))
                ],
                n_processes=n_processes,
                process_dataset=(
                    None
                    if process_dataset is None
                    else functools.partial(process_dataset, k)
                ),
            )
        )
    return sampled


def write_test_scatter(
    *,
    stage: OutputStage,
    label: str,
    datasets_test: list[list[datasets.Dataset]],
    force_field: smee.TensorForceField,
    topologies: list[smee.TensorTopology],
    path_manager: WorkflowPathManager,
    device: torch.device,
) -> None:
    """Write scatter data for every test dataset and molecule.

    Parameters
    ----------
    stage : OutputStage
        The top-level stage to write into, e.g. ``training_iteration_1``. Each test
        dataset's scatter data goes in its ``test_dataset_<k>`` subdirectory.

    label : str
        Describes the force field in the logged statistics, e.g. ``"Initial"``.

    datasets_test : list[list[datasets.Dataset]]
        One list per test dataset, holding that dataset's data for each molecule in
        order.

    force_field : smee.TensorForceField
        The force field to predict energies and forces with.

    topologies : list[smee.TensorTopology]
        The topology of each molecule.

    path_manager : WorkflowPathManager
        Gives the output paths for each dataset.

    device : torch.device
        The device to predict on.
    """
    for k, datasets_for_mols in enumerate(datasets_test):
        dataset_stage = OutputStage(stage.kind, stage.index, DatasetSplit.TEST, k)
        path_manager.mk_stage_dir(dataset_stage)
        for mol_idx, (dataset_test, topology) in enumerate(
            zip(datasets_for_mols, topologies, strict=True)
        ):
            energy_mean, energy_sd, forces_mean, forces_sd = write_scatter(
                dataset_test,
                force_field,
                topology,
                device,
                str(
                    path_manager.get_output_path_for_mol(
                        dataset_stage, OutputType.SCATTER, mol_idx
                    )
                ),
            )
            logger.info(
                f"{label} test dataset {k} molecule {mol_idx} force field statistics: Energy (Mean/SD): {energy_mean:.3e}/{energy_sd:.3e} kcal/mol, Forces (Mean/SD): {forces_mean:.3e}/{forces_sd:.3e} kcal/mol/Å"
            )


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

    # Generate the test data for all molecules
    stage = OutputStage(StageKind.TESTING)
    path_manager.mk_stage_dir(stage)
    logger.info("Generating test data")

    datasets_test = sample_datasets(
        stage=stage,
        split=DatasetSplit.TEST,
        protocols=settings.testing_sampling_settings,
        mols=off_mols,
        offxml_path=initial_offxml_path,
        path_manager=path_manager,
        device_type=settings.device_type,
        n_processes=settings.n_sampling_processes,
    )
    dataset_weights_test = [
        protocol.dataset_weight for protocol in settings.testing_sampling_settings
    ]
    molecule_weights_test = [
        protocol.molecule_weights for protocol in settings.testing_sampling_settings
    ]

    # Write out statistics on the initial force field
    stage = OutputStage(StageKind.INITIAL_STATISTICS)
    path_manager.mk_stage_dir(stage)
    write_test_scatter(
        stage=stage,
        label="Initial",
        datasets_test=datasets_test,
        force_field=tensor_ff,
        topologies=tensor_tops,
        path_manager=path_manager,
        device=settings.device,
    )

    off_ff = convert_to_smirnoff(
        trainable.to_force_field(trainable_parameters), base=initial_off_ff
    )
    off_ff.to_file(str(path_manager.get_output_path(stage, OutputType.OFFXML)))

    train_fn = _TRAINING_FNS_REGISTRY[settings.training_settings.optimiser]
    dataset_weights = [
        protocol.dataset_weight for protocol in settings.training_sampling_settings
    ]
    molecule_weights = [
        protocol.molecule_weights for protocol in settings.training_sampling_settings
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
                stage=stage,
                split=DatasetSplit.TRAIN,
                protocols=settings.training_sampling_settings,
                mols=off_mols,
                offxml_path=sampling_offxml_path,
                path_manager=path_manager,
                device_type=settings.device_type,
                n_processes=settings.n_sampling_processes,
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
                molecule_weights=molecule_weights,
                molecule_weights_test=molecule_weights_test,
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

            write_test_scatter(
                stage=stage,
                label=f"Iteration {iteration}",
                datasets_test=datasets_test,
                force_field=tensor_ff,
                topologies=tensor_tops,
                path_manager=path_manager,
                device=settings.device,
            )

            previous_datasets = datasets_train  # Carry into the next iteration

    # Plot
    analyse_workflow(settings)

    return off_ff
