"""
Loss functions for tuning the forcefield
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


def _compute_total_energy_force_loss(
    force_field: smee.TensorForceField,
    datasets_list: list[datasets.Dataset],
    topologies: list[smee.TensorTopology],
    device_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute total energy and force loss across all molecules.

    Args:
        force_field: The force field to use for predictions.
        datasets_list: List of datasets for each molecule.
        topologies: List of topologies for each molecule.
        device_type: Device type for computations.

    Returns:
        Tuple of (avg_energy_loss, avg_force_loss) averaged across molecules.
    """
    energy_losses = []
    force_losses = []

    for dataset, topology in zip(datasets_list, topologies, strict=True):
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
            create_graph=False,
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

        energy_losses.append(energy_loss)
        force_losses.append(force_loss)

    total_energy_loss = torch.stack(energy_losses).mean()
    total_force_loss = torch.stack(force_losses).mean()

    return total_energy_loss, total_force_loss


def prediction_loss(
    datasets_list: list[datasets.Dataset],
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    topologies: list[smee.TensorTopology],
    regularisation_target: typing.Literal["initial", "zero"],
    device_type: str,
) -> LossRecord:
    """Predict the loss function for a guess forcefield against datasets for multiple molecules.

    Energy and force weights are read from the dataset entries (energy_weights and
    forces_weights columns). If these columns are not present, default weights of 1.0
    are used.

    Args:
        datasets_list: List of datasets to predict the energies and forces of.
        trainable: The trainable object containing the force field.
        trainable_parameters: The parameters to be optimized.
        initial_parameters: The initial parameters before training.
        topologies: List of topologies of the molecules in the datasets.
        regularisation_target: The type of regularisation to apply ('initial' or 'zero').
        device_type: The device type (e.g., 'cpu' or 'cuda').

    Returns:
        The computed loss as a LossRecord.
    """
    force_field = trainable.to_force_field(trainable_parameters)

    avg_energy_loss, avg_force_loss = _compute_total_energy_force_loss(
        force_field,
        datasets_list,
        topologies,
        device_type,
    )

    # Compute regularization once for all molecules
    regularisation_loss = compute_regularisation_loss(
        trainable,
        trainable_parameters,
        initial_parameters,
        regularisation_target,
    )

    logger.debug(
        f"Loss: Energy={avg_energy_loss.item():.4f} Forces={avg_force_loss.item():.4f} Reg={regularisation_loss.item():.4f}"
    )

    return LossRecord(
        energy=avg_energy_loss,
        forces=avg_force_loss,
        regularisation=regularisation_loss,
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
    loss_energy_weight: float,
    loss_force_weight: float,
    regularisation_target: typing.Literal["initial", "zero"],
) -> descent.optim.ClosureFn:
    """Return a closure function for use with Levenberg-Marquardt optimizer.

    This closure reuses the same loss computation as `prediction_loss` but wraps it
    in a format suitable for the LM optimizer, which requires computing gradients
    and Hessians.

    Args:
        datasets_list: List of datasets to predict the energies and forces of.
        trainable: The trainable object containing the force field.
        trainable_parameters: The parameters to be optimized.
        initial_parameters: The initial parameters before training.
        topologies: List of topologies of the molecules in the datasets.
        loss_energy_weight: Weight for the energy loss term.
        loss_force_weight: Weight for the force loss term.
        regularisation_target: The type of regularisation to apply ('initial' or 'zero').

    Returns:
        A closure function that takes a tensor and returns the loss, gradient
        (if requested), and Hessian (if requested).
    """

    def closure_fn(
        x: torch.Tensor,
        compute_gradient: bool,
        compute_hessian: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        gradient, hessian = None, None

        @torch.enable_grad()  # type: ignore[no-untyped-call]
        def loss_fn(_x: torch.Tensor) -> torch.Tensor:
            """Compute the total loss for the given trainable parameters."""
            ff = trainable.to_force_field(_x)

            avg_energy_loss, avg_force_loss = _compute_total_energy_force_loss(
                ff,
                datasets_list,
                topologies,
                _x.device.type,
            )

            # Apply loss weights
            total_loss = (
                loss_energy_weight * avg_energy_loss
                + loss_force_weight * avg_force_loss
            )

            # Add regularisation
            regularisation_penalty = compute_regularisation_loss(
                trainable,
                _x,
                initial_parameters,
                regularisation_target=regularisation_target,
            )
            total_loss = total_loss + regularisation_penalty

            return total_loss

        # Compute Hessian first (needs fresh graph each time it's called internally)
        if compute_hessian:
            hessian = torch.autograd.functional.hessian(  # type: ignore[no-untyped-call]
                loss_fn, x, vectorize=True, create_graph=False
            ).detach()

        # Compute loss and gradient together
        if compute_gradient:
            loss = loss_fn(x)
            (gradient,) = torch.autograd.grad(loss, x, create_graph=False)
            gradient = gradient.detach()
            loss = loss.detach()
        else:
            # If no gradient needed, just compute loss without graph
            with torch.no_grad():
                loss = loss_fn(x)

        # Free GPU memory explicitly
        if x.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return loss, gradient, hessian

    return closure_fn


def predict(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    reference: typing.Literal["mean", "min"] = "mean",
    normalize: bool = True,
    device_type: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict the relative energies [kcal/mol] and forces [kcal/mol/Å] of a dataset.

    Args:
        dataset: The dataset to predict the energies and forces of.
        force_field: The force field to use to predict the energies and forces.
        topologies: The topologies of the molecules in the dataset. Each key should be
            a fully indexed SMILES string.
        reference: The reference energy to compute the relative energies with respect
            to. This should be either the "mean" energy of all conformers, or the
            energy of the conformer with the lowest reference energy ("min").
        normalize: Whether to scale the relative energies by ``1/sqrt(n_confs_i)``
            and the forces by ``1/sqrt(n_confs_i * n_atoms_per_conf_i * 3)`` This
            is useful when wanting to compute the MSE per entry.

    Returns:
        The predicted and reference relative energies [kcal/mol] with
        ``shape=(n_confs,)``, and predicted and reference forces [kcal/mol/Å] with
        ``shape=(n_confs * n_atoms_per_conf, 3)``.
    """
    energy_ref_all, energy_pred_all = [], []
    forces_ref_all, forces_pred_all = [], []

    for entry in dataset:
        smiles = entry["smiles"]

        energy_ref = entry["energy"].to(device_type)
        forces_ref = entry["forces"].reshape(len(energy_ref), -1, 3).to(device_type)

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
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if reference.lower() == "mean":
            energy_ref_0 = energy_ref.mean()
            energy_pred_0 = energy_pred.mean()
        elif reference.lower() == "min":
            min_idx = energy_ref.argmin()

            energy_ref_0 = energy_ref[min_idx]
            energy_pred_0 = energy_pred[min_idx]
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

    return (
        energy_ref_all_tensor,
        energy_pred_all_tensor,
        forces_ref_all_tensor,
        forces_pred_all_tensor,
    )


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
