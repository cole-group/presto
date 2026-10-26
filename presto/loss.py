"""Functionality for computing the loss."""

import typing

import datasets
import descent
import descent.optim
import descent.train
import descent.utils.loss
import loguru
import smee
import smee.utils
import torch

from .data_utils import get_conformer_weights_from_entry, get_weights_from_entry
from .utils.gpu import free_gpu_memory

logger = loguru.logger
_CPU_DEVICE = torch.device("cpu")


class LossRecord(typing.NamedTuple):
    """Container for different loss components."""

    energy: torch.Tensor
    forces: torch.Tensor
    regularisation: torch.Tensor


class Prediction(typing.NamedTuple):
    """Predicted and reference energies and forces of one molecule, with weights.

    Each tensor covers the conformations of every entry of the molecule, in order.
    """

    energy_ref: torch.Tensor
    """Reference energies relative to each entry's offset, with shape (n_confs,)."""
    energy_pred: torch.Tensor
    """Predicted energies relative to each entry's offset, with shape (n_confs,)."""
    forces_ref: torch.Tensor
    """Reference forces with shape (n_confs * n_atoms, 3), 0 where not finite."""
    forces_pred: torch.Tensor
    """Predicted forces with shape (n_confs * n_atoms, 3)."""
    energy_weights: torch.Tensor
    """Absolute energy weights with shape (n_confs,)."""
    forces_weights: torch.Tensor
    """Absolute force weights with shape (n_confs,)."""
    energy_conformer_weights: torch.Tensor
    """Relative conformer weights, 0 where the reference energy is not finite."""
    forces_conformer_weights: torch.Tensor
    """Relative conformer weights, 0 where any reference force is not finite."""
    entry_sizes: list[int]
    """The number of conformations in each entry, in order."""


def _mean_over_active_entries(
    loss_per_conf: torch.Tensor,
    weights: torch.Tensor,
    conformer_weights: torch.Tensor,
    entry_sizes: list[int],
    min_n_confs: int,
    device: torch.device,
) -> torch.Tensor:
    """Average the weighted mean losses of the active entries of one molecule.

    Each entry's loss is ``sum(w r d) / sum(r)``. An entry is active when
    ``sum(w r) > 0`` and it has at least ``min_n_confs`` conformations with ``r > 0``.
    Activity depends only on the weights, so inactive entries are skipped without
    forming ``0 / 0``. Returns 0 if no entry is active.
    """
    entry_losses = []
    for entry_loss, entry_w, entry_r in zip(
        torch.split(loss_per_conf, entry_sizes),
        torch.split(weights, entry_sizes),
        torch.split(conformer_weights, entry_sizes),
        strict=True,
    ):
        if (entry_r > 0).sum() < min_n_confs or (entry_w * entry_r).sum() <= 0:
            continue
        entry_losses.append((entry_w * entry_r * entry_loss).sum() / entry_r.sum())

    if not entry_losses:
        return torch.tensor(0.0, dtype=loss_per_conf.dtype, device=device)
    return torch.stack(entry_losses).mean()


def _compute_molecule_energy_force_loss(
    prediction: Prediction,
    n_atoms: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted energy and force loss for a single molecule.

    For each entry ``e`` and term (energy or forces), the loss is the weighted mean
    ``L_e = sum_i(w_i r_i d_i) / sum_i(r_i)``, where ``w`` are the absolute weights,
    ``r`` the relative conformer weights (0 for conformations with missing reference
    data) and ``d`` the squared errors. The molecule's loss is the mean of ``L_e``
    over the active entries, so each entry gets one share regardless of its number
    of conformations. An entry is inactive for a term when ``sum(w r) == 0``, or, for
    energies, when fewer than two conformations have ``r > 0`` (the energy offset
    makes a single energy carry no information).

    Args:
        prediction: The molecule's predicted and reference energies and forces, with
            their weights.
        n_atoms: Number of atoms in the molecule.
        device: Device for tensor creation.

    Returns:
        Tuple of (energy_loss, force_loss) for this molecule.
    """
    n_confs = prediction.energy_ref.shape[0]

    energy_diff_sq = ((prediction.energy_ref - prediction.energy_pred) / n_atoms) ** 2

    forces_ref_reshaped = prediction.forces_ref.reshape(n_confs, n_atoms, 3)
    forces_pred_reshaped = prediction.forces_pred.reshape(n_confs, n_atoms, 3)
    forces_diff_sq = (forces_ref_reshaped - forces_pred_reshaped) ** 2
    # Average over atoms and dimensions for each conformation
    forces_loss_per_conf = forces_diff_sq.sum(dim=[1, 2]) / (3 * n_atoms)

    energy_loss = _mean_over_active_entries(
        energy_diff_sq,
        prediction.energy_weights,
        prediction.energy_conformer_weights,
        prediction.entry_sizes,
        min_n_confs=2,
        device=device,
    )
    force_loss = _mean_over_active_entries(
        forces_loss_per_conf,
        prediction.forces_weights,
        prediction.forces_conformer_weights,
        prediction.entry_sizes,
        min_n_confs=1,
        device=device,
    )
    return energy_loss, force_loss


def _compute_molecule_loss(
    force_field: smee.TensorForceField,
    dataset: datasets.Dataset,
    topology: smee.TensorTopology,
    device: torch.device,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the energy and force loss for a single molecule.

    The losses stay attached to the force field parameters when ``create_graph``
    is True, so they can be differentiated.
    """
    prediction = predict_with_weights(
        dataset,
        force_field,
        {dataset[0]["smiles"]: topology},
        device=device,
        normalize=False,
        create_graph=create_graph,
    )
    return _compute_molecule_energy_force_loss(prediction, topology.n_atoms, device)


def _compute_molecule_total_loss_and_grad(
    force_field: smee.TensorForceField,
    dataset: datasets.Dataset,
    topology: smee.TensorTopology,
    trainable_parameters: torch.Tensor,
    device: torch.device,
    compute_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute total loss and optionally gradient for a single molecule.

    This function computes the loss for one molecule and optionally computes
    its gradient with respect to trainable parameters. The loss is detached
    after gradient computation to free memory.

    Args:
        force_field: The force field to use for predictions.
        dataset: Dataset for this molecule.
        topology: Topology for this molecule.
        trainable_parameters: Parameters to compute gradients for.
        device: Device for computations.
        compute_grad: Whether to compute gradients (default: True).

    Returns:
        Tuple of (energy_loss_detached, force_loss_detached, gradient_or_none)
    """
    # Only keep the graph if computing gradients
    energy_loss, force_loss = _compute_molecule_loss(
        force_field, dataset, topology, device, create_graph=compute_grad
    )

    # Compute combined loss for this molecule
    combined_loss = energy_loss + force_loss

    # Compute gradient for this molecule if requested and parameters require grad
    gradient = None
    if compute_grad and trainable_parameters.requires_grad:
        (gradient,) = torch.autograd.grad(
            combined_loss,
            trainable_parameters,
            create_graph=True,  # Keep graph alive for multiple molecules
        )
        # Memory optimisation: Detach gradient immediately
        gradient = gradient.detach()

    # Memory optimisation: Detach losses after gradient computation
    # This frees the computation graph for this molecule
    energy_loss_detached = energy_loss.detach()
    force_loss_detached = force_loss.detach()

    return energy_loss_detached, force_loss_detached, gradient


class DatasetLoss(typing.NamedTuple):
    """Energy and force loss for one dataset, averaged over molecules."""

    energy: torch.Tensor
    forces: torch.Tensor


def compute_dataset_loss_and_grad(
    datasets_list: list[datasets.Dataset],
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    topologies: list[smee.TensorTopology],
    device: torch.device,
    compute_grad: bool = True,
) -> tuple[DatasetLoss, torch.Tensor | None]:
    """Compute the loss of one dataset and optionally its gradient, without regularisation.

    This function computes gradients for each molecule separately using
    torch.autograd.grad, then accumulates them. This is more memory-efficient
    than computing the full loss and calling .backward(), especially when
    training with multiple molecules.

    Args:
        datasets_list: The dataset for each molecule, in the order of ``topologies``.
        trainable: The trainable object containing the force field.
        trainable_parameters: The parameters to be optimized.
        topologies: List of topologies of the molecules in the datasets.
        device: Device for computations.
        compute_grad: Whether to compute gradients (default: True).

    Returns:
        Tuple of (DatasetLoss averaged over molecules, averaged gradient or None)
    """
    # Create force field once and reuse for all molecules
    force_field = trainable.to_force_field(trainable_parameters)

    energy_losses = []
    force_losses = []
    accumulated_gradient = None
    if compute_grad:
        accumulated_gradient = torch.zeros_like(trainable_parameters)

    # Process each molecule separately to minimize memory usage
    for dataset, topology in zip(datasets_list, topologies, strict=True):
        energy_loss, force_loss, gradient = _compute_molecule_total_loss_and_grad(
            force_field,
            dataset,
            topology,
            trainable_parameters,
            device,
            compute_grad=compute_grad,
        )

        if compute_grad and gradient is not None and accumulated_gradient is not None:
            # Accumulate gradients from each molecule
            accumulated_gradient += gradient

        energy_losses.append(energy_loss)
        force_losses.append(force_loss)

        # Free GPU memory after each molecule
        if device.type == "cuda":
            free_gpu_memory()

    # Average losses across molecules
    avg_energy_loss = torch.stack(energy_losses).mean()
    avg_force_loss = torch.stack(force_losses).mean()

    # Average accumulated gradients
    if compute_grad and accumulated_gradient is not None:
        accumulated_gradient /= len(datasets_list)

    return DatasetLoss(energy=avg_energy_loss, forces=avg_force_loss), (
        accumulated_gradient
    )


def _normalise_dataset_weights(
    per_molecule_datasets_by_dataset: list[list[datasets.Dataset]],
    dataset_weights: list[float],
) -> list[float]:
    """Check there is one weight per dataset and normalise them to sum to one."""
    if not per_molecule_datasets_by_dataset:
        raise ValueError("At least one dataset is required to compute the loss")
    if len(dataset_weights) != len(per_molecule_datasets_by_dataset):
        raise ValueError(
            f"Got {len(dataset_weights)} dataset weights for "
            f"{len(per_molecule_datasets_by_dataset)} datasets"
        )
    # Normalise first so a single dataset has a weight of exactly 1.0
    total_weight = sum(dataset_weights)
    return [weight / total_weight for weight in dataset_weights]


def compute_overall_loss_and_grad(
    per_molecule_datasets_by_dataset: list[list[datasets.Dataset]],
    dataset_weights: list[float],
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    topologies: list[smee.TensorTopology],
    regularisation_target: typing.Literal["initial", "zero"],
    device: torch.device,
    compute_grad: bool = True,
) -> tuple[LossRecord, list[DatasetLoss], torch.Tensor | None]:
    """Compute the weighted loss over several datasets, plus regularisation.

    Each dataset's loss is computed separately with `compute_dataset_loss_and_grad`
    and combined as ``sum_k w_k * L_k`` with ``w_k = weight_k / sum(weights)``, so a
    dataset's influence does not depend on its number of conformers. The
    regularisation loss is added once.

    Args:
        per_molecule_datasets_by_dataset: One list per dataset, holding
            that dataset's data for each molecule in order.
        dataset_weights: The weight of each dataset.
        trainable: The trainable object containing the force field.
        trainable_parameters: The parameters to be optimized.
        initial_parameters: The initial parameters before training.
        topologies: List of topologies of the molecules in the datasets.
        regularisation_target: The type of regularisation to apply ('initial' or 'zero').
        device: Device for computations.
        compute_grad: Whether to compute gradients (default: True).

    Returns:
        Tuple of (combined LossRecord, DatasetLoss for each dataset, gradient or None)
    """
    normalised_weights = _normalise_dataset_weights(
        per_molecule_datasets_by_dataset, dataset_weights
    )

    dataset_losses = []
    accumulated_gradient = (
        torch.zeros_like(trainable_parameters) if compute_grad else None
    )
    for datasets_list, weight in zip(
        per_molecule_datasets_by_dataset, normalised_weights, strict=True
    ):
        dataset_loss, gradient = compute_dataset_loss_and_grad(
            datasets_list,
            trainable,
            trainable_parameters,
            topologies,
            device,
            compute_grad=compute_grad,
        )
        dataset_losses.append(dataset_loss)
        if gradient is not None and accumulated_gradient is not None:
            accumulated_gradient += weight * gradient

    energy_loss = torch.stack(
        [
            w * loss.energy
            for w, loss in zip(normalised_weights, dataset_losses, strict=True)
        ]
    ).sum()
    force_loss = torch.stack(
        [
            w * loss.forces
            for w, loss in zip(normalised_weights, dataset_losses, strict=True)
        ]
    ).sum()

    # Compute regularization and its gradient
    regularisation_loss = compute_regularisation_loss(
        trainable,
        trainable_parameters,
        initial_parameters,
        regularisation_target,
    )

    # Add regularization gradient if needed
    if (
        compute_grad
        and regularisation_loss.requires_grad
        and accumulated_gradient is not None
    ):
        (reg_grad,) = torch.autograd.grad(
            regularisation_loss,
            trainable_parameters,
            create_graph=False,
            retain_graph=False,
        )
        accumulated_gradient += reg_grad.detach()
    regularisation_loss = regularisation_loss.detach()

    logger.debug(
        f"Loss: Energy={energy_loss.item():.4f} "
        f"Forces={force_loss.item():.4f} "
        f"Reg={regularisation_loss.item():.4f}"
    )

    return (
        LossRecord(
            energy=energy_loss,
            forces=force_loss,
            regularisation=regularisation_loss,
        ),
        dataset_losses,
        accumulated_gradient,
    )


def compute_regularisation_loss(
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    regularisation_target: typing.Literal["initial", "zero"],
) -> torch.Tensor:
    """Compute regularisation penalty."""
    reg_loss = torch.tensor(0.0, device=trainable_parameters.device)

    regularised_idxs = trainable.regularized_idxs
    regularisation_weights = trainable.regularization_weights

    if len(regularised_idxs) == 0:
        return reg_loss

    if all(weight == 0.0 for weight in regularisation_weights):
        return reg_loss

    if regularisation_target == "initial":
        target = initial_parameters[regularised_idxs]
    elif regularisation_target == "zero":
        target = torch.zeros_like(trainable_parameters[regularised_idxs])
    else:
        raise NotImplementedError(
            f"regularisation value {regularisation_target} not implemented"
        )

    # Weighted mean of the squared deviations, so the loss rises linearly with
    # each regularisation weight
    reg_loss += (
        ((trainable_parameters[regularised_idxs] - target) ** 2)
        * regularisation_weights
    ).sum() / len(regularised_idxs)

    return reg_loss


def compute_overall_loss_hessian(
    per_molecule_datasets_by_dataset: list[list[datasets.Dataset]],
    dataset_weights: list[float],
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    topologies: list[smee.TensorTopology],
    regularisation_target: typing.Literal["initial", "zero"],
    device: torch.device,
) -> torch.Tensor:
    """Compute the Hessian of the loss from `compute_overall_loss_and_grad`.

    As with the gradient, the Hessian is computed for each molecule separately and
    accumulated with the same weights, so only one molecule's graph is held at once.

    Args:
        per_molecule_datasets_by_dataset: One list per dataset, holding
            that dataset's data for each molecule in order.
        dataset_weights: The weight of each dataset.
        trainable: The trainable object containing the force field.
        trainable_parameters: The parameters to differentiate with respect to.
        initial_parameters: The initial parameters before training.
        topologies: List of topologies of the molecules in the datasets.
        regularisation_target: The type of regularisation to apply ('initial' or 'zero').
        device: Device for computations.

    Returns:
        The Hessian with ``shape=(n_parameters, n_parameters)``.
    """
    normalised_weights = _normalise_dataset_weights(
        per_molecule_datasets_by_dataset, dataset_weights
    )
    x = trainable_parameters.detach()
    hessian = torch.zeros(len(x), len(x), dtype=x.dtype, device=x.device)

    for datasets_list, weight in zip(
        per_molecule_datasets_by_dataset, normalised_weights, strict=True
    ):
        for dataset, topology in zip(datasets_list, topologies, strict=True):

            def molecule_loss(
                _x: torch.Tensor,
                dataset: datasets.Dataset = dataset,
                topology: smee.TensorTopology = topology,
            ) -> torch.Tensor:
                energy_loss, force_loss = _compute_molecule_loss(
                    trainable.to_force_field(_x),
                    dataset,
                    topology,
                    device,
                    create_graph=True,
                )
                return energy_loss + force_loss

            # Not vectorised: the force field's sparse tensors have no batching rule
            molecule_hessian = torch.autograd.functional.hessian(  # type: ignore[no-untyped-call]
                molecule_loss, x
            )
            hessian += weight / len(datasets_list) * molecule_hessian.detach()

            if device.type == "cuda":
                free_gpu_memory()

    def regularisation_loss(_x: torch.Tensor) -> torch.Tensor:
        return compute_regularisation_loss(
            trainable, _x, initial_parameters, regularisation_target
        )

    hessian += torch.autograd.functional.hessian(  # type: ignore[no-untyped-call]
        regularisation_loss, x
    ).detach()
    return hessian


def get_loss_closure_fn(
    per_molecule_datasets_by_dataset: list[list[datasets.Dataset]],
    dataset_weights: list[float],
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    topologies: list[smee.TensorTopology],
    regularisation_target: typing.Literal["initial", "zero"],
) -> descent.optim.ClosureFn:
    """Return a closure function for use with Levenberg-Marquardt optimizer.

    This closure uses memory-efficient gradient and Hessian computation where both
    are computed per-molecule and accumulated manually, consistent with the Adam
    optimizer implementation.

    Args:
        per_molecule_datasets_by_dataset: One list per dataset, holding
            that dataset's data for each molecule in order.
        dataset_weights: The weight of each dataset.
        trainable: The trainable object containing the force field.
        trainable_parameters: The parameters to be optimized.
        initial_parameters: The initial parameters before training.
        topologies: List of topologies of the molecules in the datasets.
        regularisation_target: Type of regularisation ('initial' or 'zero').

    Returns:
        A closure function that takes a tensor and returns the loss, gradient
        (if requested), and Hessian (if requested).
    """

    def closure_fn(
        x: torch.Tensor,
        compute_gradient: bool,
        compute_hessian: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Compute Hessian first if requested to avoid memory aliasing issues. It is
        # a new tensor, so the optimizer can modify it in place.
        hessian = None
        if compute_hessian:
            hessian = compute_overall_loss_hessian(
                per_molecule_datasets_by_dataset,
                dataset_weights,
                trainable,
                x,
                initial_parameters,
                topologies,
                regularisation_target,
                x.device,
            )

        # Compute loss and gradient
        loss_record, _, gradient = compute_overall_loss_and_grad(
            per_molecule_datasets_by_dataset,
            dataset_weights,
            trainable,
            x,
            initial_parameters,
            topologies,
            regularisation_target,
            x.device,
            compute_grad=compute_gradient,
        )

        total_loss = (
            loss_record.energy + loss_record.forces + loss_record.regularisation
        )

        # Free GPU memory explicitly
        if x.device.type == "cuda":
            free_gpu_memory()

        return total_loss, gradient, hessian

    return closure_fn


def predict_with_weights(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    reference: typing.Literal["mean", "min", "median"] = "mean",
    normalize: bool = True,
    device: torch.device = _CPU_DEVICE,
    create_graph: bool = False,
) -> Prediction:
    """Predict the relative energies and forces of a dataset, with their weights.

    With ``reference="mean"``, each entry's energies are offset by their mean weighted
    by the relative conformer weights. Non-finite reference energies and forces are
    replaced with 0, so no NaN enters the graph, and their conformations get a
    conformer weight of 0 for that term.

    Args:
        dataset: The dataset to predict the energies and forces of.
        force_field: The force field to use to predict the energies and forces.
        topologies: The topologies of the molecules in the dataset.
        reference: The reference energy to compute the relative energies with respect to.
        normalize: Whether to scale the relative energies and forces.
        device: Device for computations.
        create_graph: Whether to create a computation graph for gradients.

    Returns:
        The predicted and reference energies and forces, with their weights.
    """
    energy_ref_all, energy_pred_all = [], []
    forces_ref_all, forces_pred_all = [], []
    energy_weights_all, forces_weights_all = [], []
    energy_conformer_weights_all, forces_conformer_weights_all = [], []
    entry_sizes = []

    for entry in dataset:
        smiles = entry["smiles"]

        energy_ref = entry["energy"].to(device)
        forces_ref = entry["forces"].reshape(len(energy_ref), -1, 3).to(device)

        # Get weights from entry
        energy_weights, forces_weights = get_weights_from_entry(entry)
        energy_weights = energy_weights.to(device)
        forces_weights = forces_weights.to(device)
        conformer_weights = get_conformer_weights_from_entry(entry).to(device)

        # Mask missing reference data before it is used anywhere
        energy_valid = torch.isfinite(energy_ref)
        forces_valid = torch.isfinite(forces_ref).all(dim=(1, 2))
        energy_ref = torch.where(energy_valid, energy_ref, 0.0)
        forces_ref = torch.where(torch.isfinite(forces_ref), forces_ref, 0.0)
        energy_conformer_weights = conformer_weights * energy_valid
        forces_conformer_weights = conformer_weights * forces_valid

        coords_flat = smee.utils.tensor_like(
            entry["coords"], force_field.potentials[0].parameters
        )

        coords = (
            (coords_flat.reshape(len(energy_ref), -1, 3))
            .to(device)
            .detach()
            .requires_grad_(True)
        )
        topology = topologies[smiles]

        energy_pred = smee.compute_energy(topology, force_field, coords)
        forces_pred = -torch.autograd.grad(
            energy_pred.sum(),
            coords,
            create_graph=create_graph,
            retain_graph=True,  # Need to retain connection to FF parameters
            allow_unused=True,
        )[0]

        if reference.lower() == "mean":
            # Weighted by the relative weights only, so the loss stays linear in the
            # absolute weights
            offset_weights = energy_conformer_weights
            total_offset_weight = offset_weights.sum()
            if total_offset_weight > 0:
                offset_weights = offset_weights / total_offset_weight
            energy_ref_0 = (offset_weights * energy_ref).sum()
            energy_pred_0 = (
                smee.utils.tensor_like(offset_weights, energy_pred) * energy_pred
            ).sum()
        elif reference.lower() == "min":
            min_idx = energy_ref.argmin()
            energy_ref_0 = energy_ref[min_idx]
            energy_pred_0 = energy_pred[min_idx]
        elif reference.lower() == "median":
            energy_ref_0 = energy_ref.median()
            energy_pred_0 = energy_pred.median()
        else:
            raise NotImplementedError(f"invalid reference energy {reference}")

        scale_energy, scale_forces = 1.0, 1.0

        if normalize:
            scale_energy = 1.0 / torch.sqrt(torch.tensor(energy_pred.numel()))
            scale_forces = 1.0 / torch.sqrt(torch.tensor(forces_pred.numel()))

        energy_ref_all.append(scale_energy * (energy_ref - energy_ref_0))
        forces_ref_all.append(scale_forces * forces_ref.reshape(-1, 3))

        energy_pred_all.append(scale_energy * (energy_pred - energy_pred_0))
        forces_pred_all.append(scale_forces * forces_pred.reshape(-1, 3))

        energy_weights_all.append(energy_weights)
        forces_weights_all.append(forces_weights)
        energy_conformer_weights_all.append(energy_conformer_weights)
        forces_conformer_weights_all.append(forces_conformer_weights)
        entry_sizes.append(len(energy_ref))

    energy_pred_all_tensor = torch.cat(energy_pred_all)
    forces_pred_all_tensor = torch.cat(forces_pred_all)

    energy_ref_all_tensor = torch.cat(energy_ref_all)
    energy_ref_all_tensor = smee.utils.tensor_like(
        energy_ref_all_tensor, energy_pred_all_tensor
    )

    forces_ref_all_tensor = torch.cat(forces_ref_all)
    forces_ref_all_tensor = smee.utils.tensor_like(
        forces_ref_all_tensor, forces_pred_all_tensor
    )

    return Prediction(
        energy_ref=energy_ref_all_tensor,
        energy_pred=energy_pred_all_tensor,
        forces_ref=forces_ref_all_tensor,
        forces_pred=forces_pred_all_tensor,
        energy_weights=torch.cat(energy_weights_all),
        forces_weights=torch.cat(forces_weights_all),
        energy_conformer_weights=torch.cat(energy_conformer_weights_all),
        forces_conformer_weights=torch.cat(forces_conformer_weights_all),
        entry_sizes=entry_sizes,
    )
