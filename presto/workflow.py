"""Implements the overall workflow for fitting a bespoke force field."""

import copy
import pathlib

import datasets
import loguru
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

from ._exceptions import InvalidSettingsError, MoleculeIssue, format_molecule_issues
from .analyse import analyse_workflow
from .convert import _parameterise_loaded
from .data_utils import filter_dataset_outliers
from .load_molecules import (
    _open_sdf_supplier,
    _summarise_error,
    find_conformer_generation_failures,
    load_conformers_for_molecule,
    molecule_description,
)
from .outputs import OutputStage, OutputType, StageKind
from .sample import _SAMPLING_FNS_REGISTRY, SampleFn, load_precomputed_dataset
from .settings import PreComputedDatasetSettings, WorkflowSettings
from .train import _TRAINING_FNS_REGISTRY
from .writers import write_scatter

logger = loguru.logger
console = Console()


def _validate_workflow_molecule_inputs(settings: WorkflowSettings) -> list[Molecule]:
    """Validate molecule geometries needed by the configured workflow.

    This deliberately runs only for training, rather than during Pydantic model
    construction, so commands such as ``clean`` and ``analyse`` do not perform
    conformer generation or parse starting-conformer files.
    """
    molecules = settings.param_settings.openff_molecules
    issues: list[MoleculeIssue] = []
    unreadable_files: list[str] = []
    etkdg_stages: list[str] = []
    supplied_stages: list[tuple[str, pathlib.Path]] = []

    sampling_stages = [
        ("testing_sampling_settings", settings.testing_sampling_settings, True),
        (
            "training_sampling_settings",
            settings.training_sampling_settings,
            settings.n_iterations > 0,
        ),
    ]
    for name, stage_settings, active in sampling_stages:
        if not active or isinstance(stage_settings, PreComputedDatasetSettings):
            continue
        starting_conformers = stage_settings.starting_conformers
        if starting_conformers is None:
            etkdg_stages.append(name)
        else:
            supplied_stages.append((name, starting_conformers))

    msm_settings = settings.param_settings.msm_settings
    if msm_settings is not None:
        if msm_settings.starting_conformers is None:
            etkdg_stages.append("param_settings.msm_settings")
        else:
            supplied_stages.append(
                ("param_settings.msm_settings", msm_settings.starting_conformers)
            )

    for stage_name, path in supplied_stages:
        # A missing or unreadable file is a problem with the setting, not with the
        # molecules, so report it once rather than once per molecule.
        try:
            _open_sdf_supplier(path)
        except ValueError as exc:
            unreadable_files.append(f"{stage_name}.starting_conformers: {exc}")
            continue

        for index, molecule in enumerate(molecules):
            try:
                load_conformers_for_molecule(molecule, path)
            except Exception as exc:
                # Isomorphism and remapping raise ``OpenFFToolkitException`` subclasses
                # rather than ``ValueError``, so catch broadly to keep every molecule
                # problem inside the aggregated report.
                issues.append(
                    MoleculeIssue(
                        index=index,
                        description=molecule_description(molecule, index),
                        phase=f"{stage_name}.starting_conformers ({path})",
                        error=f"{type(exc).__name__}: {_summarise_error(exc)}",
                    )
                )

    if etkdg_stages:
        failures = find_conformer_generation_failures(molecules)
        required_by = ", ".join(etkdg_stages)
        for index, (description, error) in failures.items():
            issues.append(
                MoleculeIssue(
                    index=index,
                    description=description,
                    phase=f"ETKDG required by {required_by}",
                    error=error,
                )
            )

    if unreadable_files or issues:
        report = []
        if unreadable_files:
            report.append("Workflow starting-conformer files could not be read:")
            report.extend(f"  - {failure}" for failure in unreadable_files)
        if issues:
            report.append(
                format_molecule_issues(
                    "Workflow molecule input validation failed", issues, len(molecules)
                )
            )
        raise InvalidSettingsError("\n".join(report))

    return molecules


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
    molecules = _validate_workflow_molecule_inputs(settings)
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
    off_mols, initial_off_ff, tensor_tops, tensor_ff = _parameterise_loaded(
        settings.param_settings, molecules, device=settings.device_type
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

    if test_sample_fn is not load_precomputed_dataset:  # type: ignore[comparison-overlap]
        for mol_idx, dataset_test in enumerate(datasets_test):
            dataset_path_mol = path_manager.get_output_path_for_mol(
                stage, OutputType.ENERGIES_AND_FORCES, mol_idx
            )
            dataset_test.save_to_disk(str(dataset_path_mol))

    # Write out statistics on the initial force field
    stage = OutputStage(StageKind.INITIAL_STATISTICS)
    path_manager.mk_stage_dir(stage)

    # Write scatter plots for each molecule
    for mol_idx, (dataset_test, tensor_top) in enumerate(
        zip(datasets_test, tensor_tops, strict=True)
    ):
        scatter_path_mol = path_manager.get_output_path_for_mol(
            stage, OutputType.SCATTER, mol_idx
        )
        energy_mean, energy_sd, forces_mean, forces_sd = write_scatter(
            dataset_test,
            tensor_ff,
            tensor_top,
            settings.device,
            str(scatter_path_mol),
        )
        logger.info(
            f"Molecule {mol_idx} initial force field statistics: Energy (Mean/SD): {energy_mean:.3e}/{energy_sd:.3e} kcal/mol, Forces (Mean/SD): {forces_mean:.3e}/{forces_sd:.3e} kcal/mol/Å"
        )

    off_ff = convert_to_smirnoff(
        trainable.to_force_field(trainable_parameters), base=initial_off_ff
    )
    off_ff.to_file(str(path_manager.get_output_path(stage, OutputType.OFFXML)))

    train_sample_fn = _SAMPLING_FNS_REGISTRY[type(settings.training_sampling_settings)]

    train_fn = _TRAINING_FNS_REGISTRY[settings.training_settings.optimiser]

    # Train the force field
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )

    datasets_train = None  # Only None for the first iteration

    with progress:
        for iteration in progress.track(
            range(1, settings.n_iterations + 1),  # Start from 1 (0 is untrained)
            description="Iterating the Fit",
        ):
            stage = OutputStage(StageKind.TRAINING, iteration)
            path_manager.mk_stage_dir(stage)

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

            # Apply outlier filtering if configured
            if settings.outlier_filter_settings is not None:
                logger.info("Applying outlier filtering to training data")
                datasets_train_new = [
                    filter_dataset_outliers(
                        dataset=ds,
                        force_field=tensor_ff,
                        topology=tensor_top,
                        settings=settings.outlier_filter_settings,
                        device=settings.device,
                    )
                    for ds, tensor_top in zip(
                        datasets_train_new, tensor_tops, strict=True
                    )
                ]

            # Update training dataset: concatenate if memory is enabled and not the first iteration
            if settings.memory and datasets_train is not None:
                datasets_train = [
                    datasets.combine.concatenate_datasets([ds_old, ds_new])
                    for ds_old, ds_new in zip(
                        datasets_train, datasets_train_new, strict=True
                    )
                ]
            else:
                datasets_train = datasets_train_new

            # Save each dataset
            if train_sample_fn is not load_precomputed_dataset:  # type: ignore[comparison-overlap]
                for mol_idx, dataset_train in enumerate(datasets_train):
                    dataset_path_mol = path_manager.get_output_path_for_mol(
                        stage, OutputType.ENERGIES_AND_FORCES, mol_idx
                    )
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
                scatter_path_mol = path_manager.get_output_path_for_mol(
                    stage, OutputType.SCATTER, mol_idx
                )
                energy_mean_new, energy_sd_new, forces_mean_new, forces_sd_new = (
                    write_scatter(
                        dataset_test,
                        tensor_ff,
                        tensor_top,
                        settings.device,
                        str(scatter_path_mol),
                    )
                )
                logger.info(
                    f"Iteration {iteration} Molecule {mol_idx} force field statistics: Energy (Mean/SD): {energy_mean_new:.3e}/{energy_sd_new:.3e} kcal/mol, Forces (Mean/SD): {forces_mean_new:.3e}/{forces_sd_new:.3e} kcal/mol/Å"
                )

    # Plot
    analyse_workflow(settings)

    return off_ff
