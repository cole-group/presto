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

logger = loguru.logger


class LossRecord(typing.NamedTuple):
    """Container for different loss components"""

    energy: torch.Tensor
    forces: torch.Tensor
    regularisation: torch.Tensor


def prediction_loss(
    dataset: datasets.Dataset,
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    topology: smee.TensorTopology,
    loss_energy_weight: float,
    loss_force_weight: float,
    regularisation_target: typing.Literal["initial", "zero"],
    device_type: str,
) -> LossRecord:
    """Predict the loss function for a guess forcefield against a dataset according
    to eqn. B2 in https://doi-org.libproxy.ncl.ac.uk/10.1063/5.0155322.

    Args:
        dataset: The dataset to predict the energies and forces of.
        trainable: The trainable object containing the force field.
        trainable_parameters: The parameters to be optimized.
        initial_parameters: The initial parameters before training.
        topologies: The topologies of the molecules in the dataset.
        loss_energy_weight: Weight for the energy loss term.
        loss_force_weight: Weight for the force loss term.
        regularisation_target: The type of regularisation to apply ('initial' or 'zero').
        device_type: The device type (e.g., 'cpu' or 'cuda').

    Returns:
        The computed loss as a LossRecord.
    """
    energy_ref_all, energy_pred_all, forces_ref_all, forces_pred_all = predict(
        dataset,
        trainable.to_force_field(trainable_parameters),
        {dataset[0]["smiles"]: topology},
        device_type=device_type,
        normalize=False,
    )

    n_confs = energy_ref_all.shape[0]
    n_atoms = topology.n_atoms

    loss_energy: torch.Tensor = (
        ((energy_ref_all - energy_pred_all) / n_atoms) ** 2
    ).sum() / n_confs

    loss_forces: torch.Tensor = ((forces_ref_all - forces_pred_all) ** 2).sum() / (
        3 * n_atoms * n_confs
    )

    # Regularisation penalty
    regularisation_loss = compute_regularisation_loss(
        trainable,
        trainable_parameters,
        initial_parameters,
        regularisation_target,
        n_atoms=n_atoms,
    )

    logger.info(
        f"Loss: Energy={loss_energy.item():.4f} Forces={loss_forces.item():.4f} Reg={regularisation_loss.item():.4f}"
    )

    return LossRecord(
        energy=loss_energy * loss_energy_weight,
        forces=loss_forces * loss_force_weight,
        regularisation=regularisation_loss,
    )


def compute_regularisation_loss(
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    regularisation_target: typing.Literal["initial", "zero"],
    n_atoms: int,
) -> torch.Tensor:
    """Compute regularisation penalty"""
    reg_loss = torch.tensor(0.0, device=trainable_parameters.device)

    regularisation_idxs, regularisation_strengths = (
        trainable.get_regularization_idxs_and_strengths()
    )

    if len(regularisation_idxs) == 0:
        return reg_loss

    if all(strength == 0.0 for strength in regularisation_strengths):
        return reg_loss

    if regularisation_target == "initial":
        target = initial_parameters[regularisation_idxs]
    elif regularisation_target == "zero":
        target = torch.zeros_like(trainable_parameters[regularisation_idxs])
    else:
        raise NotImplementedError(
            f"regularisation value " f"{regularisation_target} not implemented"
        )

    # L2 regularisation on all parameters
    reg_loss += (
        ((trainable_parameters[regularisation_idxs] - target) ** 2)
        * regularisation_strengths
    ).sum() / n_atoms

    return reg_loss


def get_loss_closure_fn(
    trainable: descent.train.Trainable,
    initial_x: torch.Tensor,
    topology: smee.TensorTopology,
    dataset: datasets.Dataset,
    # regularisation_settings: RegularisationSettings,
) -> descent.optim.ClosureFn:
    """
    Return a default closure function

    Args:
        trainable: The trainable object.
        initial_x: The initial parameters before training.
        topology: The topology of the system.
        dataset: The dataset to use for the loss function.
        regularisation_settings: Settings for regularisation.

    Returns:
        A closure function that takes a tensor and returns the loss, gradient (if requested), and hessian (if requested).
    """

    def closure_fn(
        x: torch.Tensor,
        compute_gradient: bool,
        compute_hessian: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        loss, gradient, hessian = (
            torch.zeros(size=(1,), device=x.device.type),
            None,
            None,
        )

        def loss_fn(_x: torch.Tensor) -> torch.Tensor:
            """Compute the loss function for the given trainable parameters."""
            ff = trainable.to_force_field(_x)
            y_ref, y_pred = predict(
                dataset,
                ff,
                {dataset[0]["smiles"]: topology},
                device_type=x.device.type,
                normalize=False,
            )[:2]
            loss: torch.Tensor = ((y_pred - y_ref) ** 2).mean()

            regularisation_penalty = compute_regularisation_loss(
                trainable,
                _x,
                initial_x,
                regularisation_settings,
                n_atoms=topology.n_atoms,
            )
            loss += regularisation_penalty

            return loss

        loss += loss_fn(x)

        if compute_hessian:
            hessian = torch.autograd.functional.hessian(  # type: ignore[no-untyped-call]
                loss_fn, x, vectorize=True, create_graph=False
            ).detach()
        if compute_gradient:
            (gradient,) = torch.autograd.grad(loss, x, create_graph=False)
            gradient = gradient.detach()

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
