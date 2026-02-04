"""
Functionality for computing the loss.
"""

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

from .data_utils import get_weights_from_entry

logger = loguru.logger


class LossRecord(typing.NamedTuple):
    """Container for different loss components"""

    energy: torch.Tensor
    forces: torch.Tensor
    regularisation: torch.Tensor


def _compute_molecule_energy_force_loss(
    energy_ref: torch.Tensor,
    energy_pred: torch.Tensor,
    forces_ref: torch.Tensor,
    forces_pred: torch.Tensor,
    energy_weights: torch.Tensor,
    forces_weights: torch.Tensor,
    n_atoms: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted energy and force loss for a single molecule.

    Args:
        energy_ref: Reference energies with shape (n_confs,).
        energy_pred: Predicted energies with shape (n_confs,).
        forces_ref: Reference forces with shape (n_confs * n_atoms, 3).
        forces_pred: Predicted forces with shape (n_confs * n_atoms, 3).
        energy_weights: Per-conformation energy weights with shape (n_confs,).
        forces_weights: Per-conformation force weights with shape (n_confs,).
        n_atoms: Number of atoms in the molecule.
        device: Device for tensor creation.

    Returns:
        Tuple of (energy_loss, force_loss) for this molecule.
    """
    n_confs = energy_ref.shape[0]

    # Energy loss (weighted per conformation, then averaged)
    energy_diff_sq = ((energy_ref - energy_pred) / n_atoms) ** 2
    weighted_energy_loss = (energy_diff_sq * energy_weights).sum()

    # Effective sample size for weighted average
    effective_energy_sample_size = torch.sum(energy_weights) ** 2 / torch.sum(
        energy_weights**2
    )
    if effective_energy_sample_size > 0:
        energy_loss = weighted_energy_loss / effective_energy_sample_size
    else:
        energy_loss = torch.tensor(0.0, dtype=torch.float64, device=device)

    # Force loss (weighted per conformation, then averaged)
    forces_ref_reshaped = forces_ref.reshape(n_confs, n_atoms, 3)
    forces_pred_reshaped = forces_pred.reshape(n_confs, n_atoms, 3)
    forces_diff_sq = (forces_ref_reshaped - forces_pred_reshaped) ** 2

    # Sum over atoms and dimensions for each conformation
    forces_loss_per_conf = forces_diff_sq.sum(dim=[1, 2]) / (3 * n_atoms)
    weighted_force_loss = (forces_loss_per_conf * forces_weights).sum()

    effective_force_sample_size = torch.sum(forces_weights) ** 2 / torch.sum(
        forces_weights**2
    )
    if effective_force_sample_size > 0:
        force_loss = weighted_force_loss / effective_force_sample_size
    else:
        force_loss = torch.tensor(0.0, dtype=torch.float64, device=device)

    return energy_loss, force_loss


def _compute_molecule_total_loss_and_grad(
    force_field: smee.TensorForceField,
    dataset: datasets.Dataset,
    topology: smee.TensorTopology,
    trainable_parameters: torch.Tensor,
    device_type: str,
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
        device_type: Device type for computations.
        compute_grad: Whether to compute gradients (default: True).

    Returns:
        Tuple of (energy_loss_detached, force_loss_detached, gradient_or_none)
    """
    (
        energy_ref,
        energy_pred,
        forces_ref,
        forces_pred,
        energy_weights,
        forces_weights,
    ) = predict_with_weights(
        dataset,
        force_field,
        {dataset[0]["smiles"]: topology},
        device_type=device_type,
        normalize=False,
        create_graph=compute_grad,  # Only keep graph if computing gradients
    )

    energy_loss, force_loss = _compute_molecule_energy_force_loss(
        energy_ref,
        energy_pred,
        forces_ref,
        forces_pred,
        energy_weights,
        forces_weights,
        topology.n_atoms,
        device_type,
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


def compute_overall_loss_and_grad(
    datasets_list: list[datasets.Dataset],
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    topologies: list[smee.TensorTopology],
    regularisation_target: typing.Literal["initial", "zero"],
    device_type: str,
    compute_grad: bool = True,
) -> tuple[LossRecord, torch.Tensor | None]:
    """Compute loss and optionally gradients for memory efficiency.

    This function computes gradients for each molecule separately using
    torch.autograd.grad, then accumulates them. This is more memory-efficient
    than computing the full loss and calling .backward(), especially when
    training with multiple molecules.

    Args:
        datasets_list: List of datasets to predict the energies and forces of.
        trainable: The trainable object containing the force field.
        trainable_parameters: The parameters to be optimized.
        initial_parameters: The initial parameters before training.
        topologies: List of topologies of the molecules in the datasets.
        regularisation_target: The type of regularisation to apply ('initial' or 'zero').
        device_type: The device type (e.g., 'cpu' or 'cuda').
        compute_grad: Whether to compute gradients (default: True).

    Returns:
        Tuple of (LossRecord, accumulated_gradient or None)
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
            device_type,
            compute_grad=compute_grad,
        )

        if compute_grad and gradient is not None and accumulated_gradient is not None:
            # Accumulate gradients from each molecule
            accumulated_gradient += gradient

        energy_losses.append(energy_loss)
        force_losses.append(force_loss)

        # Free GPU memory after each molecule
        if device_type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Average losses across molecules
    avg_energy_loss = torch.stack(energy_losses).mean()
    avg_force_loss = torch.stack(force_losses).mean()

    # Average accumulated gradients
    if compute_grad and accumulated_gradient is not None:
        accumulated_gradient /= len(datasets_list)

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
        f"Loss: Energy={avg_energy_loss.item():.4f} "
        f"Forces={avg_force_loss.item():.4f} "
        f"Reg={regularisation_loss.item():.4f}"
    )

    return (
        LossRecord(
            energy=avg_energy_loss,
            forces=avg_force_loss,
            regularisation=regularisation_loss,
        ),
        accumulated_gradient,
    )


def compute_regularisation_loss(
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    regularisation_target: typing.Literal["initial", "zero"],
) -> torch.Tensor:
    """Compute regularisation penalty"""
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

    effective_sample_size = torch.sum(regularisation_weights) ** 2 / torch.sum(
        regularisation_weights**2
    )

    # L2 regularisation on all parameters
    reg_loss += (
        ((trainable_parameters[regularised_idxs] - target) ** 2)
        * regularisation_weights
    ).sum() / effective_sample_size

    return reg_loss


def get_loss_closure_fn(
    datasets_list: list[datasets.Dataset],
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    topologies: list[smee.TensorTopology],
    regularisation_target: typing.Literal["initial", "zero"],
) -> descent.optim.ClosureFn:
    """Return a closure function for use with Levenberg-Marquardt optimizer.

    This closure uses memory-efficient gradient computation where gradients
    are computed per-molecule and accumulated manually, consistent with
    the Adam optimizer implementation.

    Args:
        datasets_list: List of datasets to predict energies and forces of.
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
        def loss_fn(_x: torch.Tensor) -> torch.Tensor:
            """Compute total weighted loss."""
            _loss_record, _ = compute_overall_loss_and_grad(
                datasets_list,
                trainable,
                _x,
                initial_parameters,
                topologies,
                regularisation_target,
                _x.device.type,
                compute_grad=False,
            )
            return (
                _loss_record.energy + _loss_record.forces + _loss_record.regularisation
            )

        # Compute Hessian first if requested to avoid memory aliasing issues
        hessian = None
        if compute_hessian:
            hessian = (
                torch.autograd.functional.hessian(  # type: ignore
                    loss_fn, x, vectorize=True, create_graph=False
                )
                .detach()
                .clone()
            )  # Clone to allow in-place modifications by the optimizer

        # Compute loss and gradient
        loss_record, gradient = compute_overall_loss_and_grad(
            datasets_list,
            trainable,
            x,
            initial_parameters,
            topologies,
            regularisation_target,
            x.device.type,
            compute_grad=compute_gradient,
        )

        total_loss = (
            loss_record.energy + loss_record.forces + loss_record.regularisation
        )

        # Free GPU memory explicitly
        if x.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return total_loss, gradient, hessian

    return closure_fn


def predict_with_weights(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    reference: typing.Literal["mean", "min", "median"] = "mean",
    normalize: bool = True,
    device_type: str = "cpu",
    create_graph: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Predict the relative energies and forces with associated weights from the dataset.

    This function is similar to `predict` but also returns the energy and force weights
    from the dataset entries.

    Args:
        dataset: The dataset to predict the energies and forces of.
        force_field: The force field to use to predict the energies and forces.
        topologies: The topologies of the molecules in the dataset.
        reference: The reference energy to compute the relative energies with respect to.
        normalize: Whether to scale the relative energies and forces.
        device_type: The device type (e.g., 'cpu' or 'cuda').
        create_graph: Whether to create a computation graph for gradients.

    Returns:
        Tuple of (energy_ref, energy_pred, forces_ref, forces_pred, energy_weights, forces_weights)
    """
    energy_ref_all, energy_pred_all = [], []
    forces_ref_all, forces_pred_all = [], []
    energy_weights_all, forces_weights_all = [], []

    for entry in dataset:
        smiles = entry["smiles"]

        energy_ref = entry["energy"].to(device_type)
        forces_ref = entry["forces"].reshape(len(energy_ref), -1, 3).to(device_type)

        # Get weights from entry
        energy_weights, forces_weights = get_weights_from_entry(entry)
        energy_weights = energy_weights.to(device_type)
        forces_weights = forces_weights.to(device_type)

        coords_flat = smee.utils.tensor_like(
            entry["coords"], force_field.potentials[0].parameters
        )

        coords = (
            (coords_flat.reshape(len(energy_ref), -1, 3))
            .to(device_type)
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
            energy_ref_0 = energy_ref.mean()
            energy_pred_0 = energy_pred.mean()
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

    energy_weights_tensor = torch.cat(energy_weights_all)
    forces_weights_tensor = torch.cat(forces_weights_all)

    return (
        energy_ref_all_tensor,
        energy_pred_all_tensor,
        forces_ref_all_tensor,
        forces_pred_all_tensor,
        energy_weights_tensor,
        forces_weights_tensor,
    )
