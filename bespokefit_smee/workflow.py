"""Implements the overall workflow for fitting a bespoke force field."""

import copy
import pathlib

import datasets
import loguru
from descent.train import Trainable
from openff.toolkit import ForceField
from tqdm import tqdm

from bespokefit_smee.convert import convert_to_smirnoff

from .analyse import analyse_workflow
from .convert import parameterise
from .outputs import OutputStage, OutputType, StageKind
from .sample import _SAMPLING_FNS_REGISTRY, SampleFn
from .settings import WorkflowSettings
from .train import _TRAINING_FNS_REGISTRY
from .utils._suppress_output import suppress_unwanted_output
from .utils.rdkit_bespoke_wrapper import use_bespoke_rdkit_toolkit_decorator
from .writers import write_scatter

logger = loguru.logger

suppress_unwanted_output()


@use_bespoke_rdkit_toolkit_decorator
def get_bespoke_force_field(
    settings: WorkflowSettings, write_settings: bool = True
) -> ForceField:
    """
    Fit a bespoke force field. This involves:

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

    Returns
    -------
    ForceField
        The fitted bespoke force field.
    """
    suppress_unwanted_output()

    path_manager = settings.get_path_manager()
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
        settings.parameterisation_settings, device=settings.device_type
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

    trainable_parameters = trainable.to_values().to((settings.device))

    # Required for LM optimiser only
    for tensor_top in tensor_tops:
        for param in tensor_top.parameters.values():
            param.assignment_matrix = param.assignment_matrix.to_dense()

    # Get a copy of the initial trainable parameters for regularisation
    initial_parameters = trainable_parameters.clone().detach()

    # Generate the test data for all molecules
    stage = OutputStage(StageKind.TESTING)
    path_manager.mk_stage_dir(stage)
    test_sample_fn: SampleFn = _SAMPLING_FNS_REGISTRY[
        type(settings.testing_sampling_settings)
    ]
    logger.info("Generating test data")
    datasets_test = test_sample_fn(
        mols=off_mols,
        off_ff=initial_off_ff,
        device=settings.device,
        settings=settings.testing_sampling_settings,
        output_paths={
            output_type: path_manager.get_output_path(stage, output_type)
            for output_type in settings.testing_sampling_settings.output_types
        },
    )
    for mol_idx, dataset_test in enumerate(datasets_test):
        dataset_path = path_manager.get_output_path(
            stage, OutputType.ENERGIES_AND_FORCES
        )
        dataset_path_mol = dataset_path.parent / f"{dataset_path.stem}_mol{mol_idx}"
        dataset_test.save_to_disk(str(dataset_path_mol))

    # Write out statistics on the initial force field
    stage = OutputStage(StageKind.INITIAL_STATISTICS)
    path_manager.mk_stage_dir(stage)

    # Write scatter plots for each molecule
    for mol_idx, (dataset_test, tensor_top) in enumerate(
        zip(datasets_test, tensor_tops, strict=True)
    ):
        scatter_path = path_manager.get_output_path(stage, OutputType.SCATTER)
        scatter_path_mol = (
            scatter_path.parent
            / f"{scatter_path.stem}_mol{mol_idx}{scatter_path.suffix}"
        )
        energy_mean, energy_sd, forces_mean, forces_sd = write_scatter(
            dataset_test,
            tensor_ff,
            tensor_top,
            str(settings.device),
            str(scatter_path_mol),
        )
        logger.info(
            f"Molecule {mol_idx} initial force field statistics: Energy (Mean/SD): {energy_mean:.3e}/{energy_sd:.3e}, Forces (Mean/SD): {forces_mean:.3e}/{forces_sd:.3e}"
        )

    off_ff = convert_to_smirnoff(
        trainable.to_force_field(trainable_parameters), base=initial_off_ff
    )
    off_ff.to_file(str(path_manager.get_output_path(stage, OutputType.OFFXML)))

    train_sample_fn = _SAMPLING_FNS_REGISTRY[type(settings.training_sampling_settings)]

    train_fn = _TRAINING_FNS_REGISTRY[settings.training_settings.optimiser]

    # Train the force field
    for iteration in tqdm(
        range(1, settings.n_iterations + 1),  # Start from 1 (0 is untrained)
        leave=False,
        colour="magenta",
        desc="Iterating the Fit",
    ):
        stage = OutputStage(StageKind.TRAINING, iteration)
        path_manager.mk_stage_dir(stage)
        datasets_train = None  # Only None for the first iteration

        datasets_train_new = train_sample_fn(
            mols=off_mols,
            off_ff=off_ff,
            device=settings.device,
            settings=settings.training_sampling_settings,
            output_paths={
                output_type: path_manager.get_output_path(stage, output_type)
                for output_type in settings.training_sampling_settings.output_types
            },
        )

        # Update training dataset: concatenate if memory is enabled and not the first iteration
        should_concatenate = settings.memory and datasets_train is not None
        if should_concatenate:
            datasets_train = [
                datasets.combine.concatenate_datasets([ds_old, ds_new])
                for ds_old, ds_new in zip(
                    datasets_train, datasets_train_new, strict=True
                )
            ]
        else:
            datasets_train = datasets_train_new

        # Save each dataset
        for mol_idx, dataset_train in enumerate(datasets_train):
            dataset_path = path_manager.get_output_path(
                stage, OutputType.ENERGIES_AND_FORCES
            )
            dataset_path_mol = dataset_path.parent / f"{dataset_path.stem}_mol{mol_idx}"
            dataset_train.save_to_disk(str(dataset_path_mol))

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

        # Write scatter plots for each molecule
        for mol_idx, (dataset_test, tensor_top) in enumerate(
            zip(datasets_test, tensor_tops, strict=True)
        ):
            scatter_path = path_manager.get_output_path(stage, OutputType.SCATTER)
            scatter_path_mol = (
                scatter_path.parent
                / f"{scatter_path.stem}_mol{mol_idx}{scatter_path.suffix}"
            )
            energy_mean_new, energy_sd_new, forces_mean_new, forces_sd_new = (
                write_scatter(
                    dataset_test,
                    tensor_ff,
                    tensor_top,
                    str(settings.device),
                    str(scatter_path_mol),
                )
            )
            logger.info(
                f"Iteration {iteration} Molecule {mol_idx} force field statistics: Energy (Mean/SD): {energy_mean_new:.3e}/{energy_sd_new:.3e}, Forces (Mean/SD): {forces_mean_new:.3e}/{forces_sd_new:.3e}"
            )

    # Plot
    analyse_workflow(settings, off_mols)

    return off_ff
