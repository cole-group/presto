"""Output functions for run-fit."""

import contextlib
import pathlib
import typing
from typing import TextIO

import datasets
import descent.train
import h5py
import loguru
import pandas
import smee
import tensorboardX
import torch

from .loss import (
    DatasetLoss,
    LossRecord,
    compute_overall_loss_and_grad,
    predict_with_weights,
)
from .outputs import DatasetSplit
from .utils.typing import PathLike

logger = loguru.logger


SCATTER_KEYS = (
    "energy_reference",
    "energy_predicted",
    "energy_differences",
    "forces_reference",
    "forces_predicted",
    "forces_differences",
)
"""The names of the arrays `write_scatter` saves to each HDF5 file."""


@contextlib.contextmanager
def open_writer(path: pathlib.Path) -> tensorboardX.SummaryWriter:
    """Open a TensorBoard writer."""
    path.mkdir(parents=True, exist_ok=True)
    with tensorboardX.SummaryWriter(str(path)) as writer:
        yield writer


def write_scatter(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topology_in: smee.TensorTopology,
    device: torch.device,
    filename: PathLike,
) -> tuple[float, float, float, float]:
    """Predict and save energies/forces to HDF5."""
    prediction = predict_with_weights(
        dataset,
        force_field,
        {dataset[0]["smiles"]: topology_in},
        device=device,
        normalize=False,
    )
    energy_ref_all, energy_pred_all = prediction.energy_ref, prediction.energy_pred
    forces_ref_all, forces_pred_all = prediction.forces_ref, prediction.forces_pred

    with torch.no_grad():
        energy_diffs = energy_pred_all - energy_ref_all
        force_diffs = forces_pred_all - forces_ref_all

        # Save to HDF5
        with h5py.File(filename, "w") as f:
            for key, values in zip(
                SCATTER_KEYS,
                (
                    energy_ref_all,
                    energy_pred_all,
                    energy_diffs,
                    forces_ref_all,
                    forces_pred_all,
                    force_diffs,
                ),
                strict=True,
            ):
                f.create_dataset(key, data=values.cpu().numpy())

            f.attrs["n_conformers"] = len(energy_ref_all)
            f.attrs["n_atoms"] = forces_ref_all.shape[0] // len(energy_ref_all)

        # Summary statistics
        energy_mean = torch.mean(energy_diffs).item()
        energy_std = torch.std(energy_diffs).item()
        forces_mean = torch.mean(force_diffs).item()
        forces_std = torch.std(force_diffs).item()

        return energy_mean, energy_std, forces_mean, forces_std


def report(
    step: int,
    x: torch.Tensor,
    loss: torch.Tensor,
    gradient: torch.Tensor,
    hessian: torch.Tensor,
    step_quality: float,
    accept_step: bool,
    trainable: descent.train.Trainable,
    topologies: list[smee.TensorTopology],
    datasets_train: list[list[datasets.Dataset]],
    datasets_test: list[list[datasets.Dataset]],
    dataset_weights: list[float],
    dataset_weights_test: list[float],
    initial_parameters: torch.Tensor,
    regularisation_target: typing.Literal["initial", "zero"],
    metrics_file: PathLike,
    experiment_dir: pathlib.Path,
    molecule_weights: list[list[float] | None] | None = None,
    molecule_weights_test: list[list[float] | None] | None = None,
) -> None:
    """Report training progress for Levenberg-Marquardt optimizer.

    This function computes training and test losses using the same loss
    computation as train_adam (via compute_overall_loss_and_grad) to ensure
    consistent metrics reporting.

    Args:
        step: Current optimization step.
        x: Current trainable parameters.
        loss: Loss value from the optimizer (not used, we recompute for consistency).
        gradient: Gradient tensor (not used in reporting).
        hessian: Hessian tensor (not used in reporting).
        step_quality: Quality measure of the step (not used in reporting).
        accept_step: Whether the step was accepted (not used in reporting).
        trainable: The trainable object containing the force field.
        topologies: List of topologies for all molecules.
        datasets_train: For each training dataset, the dataset for each molecule.
        datasets_test: For each test dataset, the dataset for each molecule.
        dataset_weights: The weight of each training dataset.
        dataset_weights_test: The weight of each test dataset.
        initial_parameters: Initial parameters for regularization.
        regularisation_target: Regularization target ('initial' or 'zero').
        metrics_file: Path to the metrics output file, which must already contain
            its header (see `write_metrics_header`).
        experiment_dir: Path to the TensorBoard experiment directory.
        molecule_weights: For each training dataset, the weight of each molecule.
            None, for all datasets or one, weights the molecules equally.
        molecule_weights_test: For each test dataset, the weight of each molecule.
            None, for all datasets or one, weights the molecules equally.
    """
    with torch.enable_grad():  # type: ignore[no-untyped-call]
        # Compute training loss using the same function as train_adam
        loss_train, dataset_losses_train, _ = compute_overall_loss_and_grad(
            datasets_train,
            dataset_weights,
            trainable,
            x,
            initial_parameters,
            topologies,
            regularisation_target,
            x.device,
            compute_grad=False,
            molecule_weights=molecule_weights,
        )

        # Compute test loss using the same function as train_adam
        loss_test, dataset_losses_test, _ = compute_overall_loss_and_grad(
            datasets_test,
            dataset_weights_test,
            trainable,
            x,
            initial_parameters,
            topologies,
            regularisation_target,
            x.device,
            compute_grad=False,
            molecule_weights=molecule_weights_test,
        )

        with open_writer(experiment_dir) as writer:
            with open(metrics_file, "a") as f:
                write_metrics(
                    step,
                    loss_train,
                    loss_test,
                    dataset_losses_train,
                    dataset_losses_test,
                    writer,
                    f,
                )


def combined_metrics_columns() -> list[str]:
    """Return the metrics columns of the combined (weighted) losses."""
    return [
        f"loss_{split}_{field}"
        for split in ("train", "test")
        for field in LossRecord._fields
    ]


def dataset_metrics_columns(split: DatasetSplit, dataset: int) -> list[str]:
    """Return the metrics columns of one training or test dataset's losses."""
    return [f"loss_{split.value}_{dataset}_{field}" for field in DatasetLoss._fields]


def metrics_columns(n_train: int, n_test: int) -> list[str]:
    """Return the metrics file column names, which are also the TensorBoard tags.

    The combined (weighted) losses come first, followed by the energy and force
    losses of each training and then each test dataset.
    """
    return [
        *combined_metrics_columns(),
        *(
            column
            for split, n in ((DatasetSplit.TRAIN, n_train), (DatasetSplit.TEST, n_test))
            for k in range(n)
            for column in dataset_metrics_columns(split, k)
        ),
    ]


def write_metrics_header(outfile: TextIO, n_train: int, n_test: int) -> None:
    """Write the header row naming the columns written by `write_metrics`."""
    outfile.write(" ".join(metrics_columns(n_train, n_test)) + "\n")


def write_metrics(
    i: int,
    loss_train: LossRecord,
    loss_test: LossRecord,
    dataset_losses_train: list[DatasetLoss],
    dataset_losses_test: list[DatasetLoss],
    writer: tensorboardX.SummaryWriter,
    outfile: TextIO,
) -> None:
    """Write metrics to TensorBoard and output file."""
    # Records iterate over their fields in the order of metrics_columns
    values = [
        loss.detach().item()
        for loss_record in (
            loss_train,
            loss_test,
            *dataset_losses_train,
            *dataset_losses_test,
        )
        for loss in loss_record
    ]
    names = metrics_columns(len(dataset_losses_train), len(dataset_losses_test))
    for name, value in zip(names, values, strict=True):
        outfile.write(f"{value:.10f} ")
        writer.add_scalar(name, value, i)
    outfile.write("\n")


def get_potential_summary(potential: smee.TensorPotential) -> str:
    """Get string summary of a potential's parameters."""
    output = [""]
    parameter_rows = []
    for key_id, value in enumerate(potential.parameters.detach()):
        row: dict[str, int | str] = {"ID": key_id}
        row.update(
            {
                f"{col}": (f"{value[idx].item():.4f}")
                for idx, col in enumerate(potential.parameter_cols)
            }
        )
        parameter_rows.append(row)

    output.append(
        f" {potential.type} ".center(88, "="),
    )
    output.append(f"fn={potential.fn}")

    if potential.attributes is not None:
        assert potential.attribute_units is not None, (
            "Attribute units are None even though attributes are not None"
        )
        assert potential.attribute_cols is not None, (
            "Attribute columns are None even though attributes are not None"
        )
        attribute_rows = [
            {
                f"{col}{potential.attribute_units[idx]}": (
                    f"{potential.attributes[idx].item():.4f} "
                )
                for idx, col in enumerate(potential.attribute_cols)
            }
        ]
        output.append("")
        output.append("attributes=")
        output.append("")
        output.append(pandas.DataFrame(attribute_rows).to_string(index=False))

    output.append("")
    output.append("parameters=")
    output.append("")
    output.append(pandas.DataFrame(parameter_rows).to_string(index=False))

    return "\n".join(output)


def get_potential_comparison(
    pot1: smee.TensorPotential, pot2: smee.TensorPotential
) -> str:
    """Get string comparison of two potentials."""
    output = [""]
    parameter_rows = []
    for key_id, value in enumerate(
        zip(pot1.parameters.detach(), pot2.parameters.detach(), strict=False)
    ):
        row: dict[str, int | str] = {"ID": key_id}
        row.update(
            {
                f"{col}": (f"{value[0][idx].item():.4f} --> {value[1][idx].item():.4f}")
                for idx, col in enumerate(pot1.parameter_cols)
            }
        )
        parameter_rows.append(row)

    output.append(
        f" {pot1.type} vs {pot2.type} ".center(88, "="),
    )
    output.append(f"fn={pot1.fn}")

    if pot1.attributes is not None:
        assert pot1.attribute_units is not None, (
            "Attribute units are None even though attributes are not None"
        )
        assert pot1.attribute_cols is not None, (
            "Attribute columns are None even though attributes are not None"
        )
        attribute_rows = [
            {
                f"{col}{pot1.attribute_units[idx]}": (
                    f"{pot1.attributes[idx].item():.4f} "
                )
                for idx, col in enumerate(pot1.attribute_cols)
            }
        ]
        output.append("")
        output.append("attributes=")
        output.append("")
        output.append(pandas.DataFrame(attribute_rows).to_string(index=False))

    output.append("")
    output.append("parameters=")
    output.append("")
    output.append(pandas.DataFrame(parameter_rows).to_string(index=False))

    return "\n".join(output)
