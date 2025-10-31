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

from .settings import RegularisationSettings
from .utils.typing import ValenceType

logger = loguru.logger


def prediction_loss(
    dataset: datasets.Dataset,
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    topology: smee.TensorTopology,
    loss_force_weight: float,
    regularisation_settings: RegularisationSettings,
    device_type: str,
) -> torch.Tensor:
    """Predict the loss function for a guess forcefield against a dataset.

    Args:
        dataset: The dataset to predict the energies and forces of.
        trainable: The trainable object containing the force field.
        trainable_parameters: The parameters to be optimized.
        initial_parameters: The initial parameters before training.
        topologies: The topologies of the molecules in the dataset.
        loss_force_weight: Weight for the force loss term.
        regularisation_settings: Settings for regularisation.
        device_type: The device type (e.g., 'cpu' or 'cuda').

    Returns:
        Loss value.
    """
    energy_ref_all, energy_pred_all, forces_ref_all, forces_pred_all = predict(
        dataset,
        trainable.to_force_field(trainable_parameters),
        {dataset[0]["smiles"]: topology},
        device_type=device_type,
        normalize=False,
    )
    # Loss as the JS-divergence between the two distributions
    # beta = 1.987204259e-3 * 500  # kcal/mol/K

    # def _kl_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    #     return (p * (p / q).clamp(min=1e-10).log()).sum()

    # distribution_ref = torch.exp(-energy_ref_all / beta)
    # distribution_ref = distribution_ref / distribution_ref.sum()
    # distribution_pred = torch.exp(-energy_pred_all / beta)
    # distribution_pred = distribution_pred / distribution_pred.sum()
    # m = 0.5 * (distribution_ref + distribution_pred)
    # loss_distribution = 0.5 * (
    #     _kl_div(distribution_ref, m) + _kl_div(distribution_pred, m)
    # )
    # return loss_distribution

    loss_energy: torch.Tensor = ((energy_ref_all - energy_pred_all) ** 2).mean()
    loss_forces: torch.Tensor = ((forces_ref_all - forces_pred_all) ** 2).mean()

    # Regularisation penalty
    regularisation_pentalty = compute_regularisation_penalty(
        trainable, trainable_parameters, initial_parameters, regularisation_settings
    )

    # logger.info(
    #     f"Loss: Energy={loss_energy.item():.4f} Forces={loss_forces.item():.4f} Reg={regularisation_pentalty.item():.4f}"
    # )

    return loss_energy + loss_forces * loss_force_weight + regularisation_pentalty

    # energy_loss, forces_loss = [], []
    # for entry in dataset:
    #     energy_ref = entry["energy"].to(device_type)
    #     forces_ref = entry["forces"].reshape(len(energy_ref), -1, 3).to(device_type)
    #     coords_ref = (
    #         entry["coords"]
    #         .reshape(len(energy_ref), -1, 3)
    #         .to(device_type)
    #         .detach()
    #         .requires_grad_(True)
    #     )
    #     energy_prd = smee.compute_energy(topology, force_field, coords_ref)
    #     forces_prd = -torch.autograd.grad(
    #         energy_prd.sum(),
    #         coords_ref,
    #         create_graph=True,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[0]
    #     energy_prd_0 = energy_prd.detach()[0]
    #     energy_loss.append((energy_prd - energy_ref - energy_prd_0))
    #     forces_loss.append((forces_prd - forces_ref).reshape(-1, 3))
    # lossE: torch.Tensor = (torch.cat(energy_loss) ** 2).mean()
    # lossF: torch.Tensor = (torch.cat(forces_loss) ** 2).mean()
    # return lossE + lossF * loss_force_weight**0.5


# TODO: Move this inside Descent Trainable class
def get_regularised_parameter_idxs(
    trainable: descent.train.Trainable,
    cols: dict[ValenceType, list[str]],
) -> torch.Tensor:
    """Get the indexes of the parameters to regularise (these idxs apply to the trainable_parameters,
    rather than the full set of parameters in the force field).

    Args:
        trainable: The trainable object.
        cols: Dictionary mapping valence types to parameter columns to regularise.

    Returns:
        Tensor of indexes of parameters to regularise.
    """
    idxs: list[int] = []
    col_offset = 0

    potentials = [
        trainable._force_field.potentials_by_type[potential_type]
        for potential_type in trainable._param_types
    ]

    for potential_type, potential in zip(
        trainable._param_types, potentials, strict=True
    ):
        potential_cols = potential.parameter_cols

        potential_values = potential.parameters.detach().clone()
        potential_values_flat = potential_values.flatten()

        n_rows = len(potential_values)
        unfrozen_rows = set(range(n_rows))

        if potential_type in cols:
            assert len({*cols[potential_type]} - {*potential_cols}) == 0, (
                f"unknown columns: {potential_cols}"
            )

            idxs.extend(
                col_offset + col_idx + row_idx * potential_values.shape[-1]
                for row_idx in range(n_rows)
                if row_idx in unfrozen_rows
                for col_idx, col in enumerate(potential_cols)
                if col in cols[potential_type]
            )

        col_offset += len(potential_values_flat)

    # Get the indices of the regularised values in the unfrozen idxs of the trainable
    trained_and_regularised_idxs = set()
    for idx_in_unfrozen, unfrozen_idx in enumerate(trainable._unfrozen_idxs):
        if unfrozen_idx in idxs:
            trained_and_regularised_idxs.add(idx_in_unfrozen)

    return smee.utils.tensor_like(
        torch.tensor(
            list(trained_and_regularised_idxs),
            dtype=torch.long,
        ),
        trainable._unfrozen_idxs,
    )


def compute_regularisation_penalty(
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    regularisation_settings: RegularisationSettings,
) -> torch.Tensor:
    """Compute regularisation penalty"""
    penalty = torch.tensor(0.0, device=trainable_parameters.device)

    # Get the idxs of the parameters to regularise
    if hasattr(trainable, "regularised_parameter_idxs"):
        regularised_parameter_idxs = trainable.regularised_parameter_idxs
    else:
        regularised_parameter_idxs = get_regularised_parameter_idxs(
            trainable, regularisation_settings.parameters
        )
        trainable.regularised_parameter_idxs = regularised_parameter_idxs

    if regularisation_settings.regularisation_value == "initial":
        target = initial_parameters[regularised_parameter_idxs]
    elif regularisation_settings.regularisation_value == "zero":
        target = torch.zeros_like(trainable_parameters[regularised_parameter_idxs])
    else:
        raise NotImplementedError(
            f"regularisation value "
            f"{regularisation_settings.regularisation_value} not implemented"
        )

    # L2 regularisation on all parameters
    penalty += (
        (trainable_parameters[regularised_parameter_idxs] - target) ** 2
    ).mean() * regularisation_settings.regularisation_strength

    return penalty


def get_loss_closure_fn(
    trainable: descent.train.Trainable,
    initial_x: torch.Tensor,
    topology: smee.TensorTopology,
    dataset: datasets.Dataset,
    regularisation_settings: RegularisationSettings,
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

            regularisation_penalty = compute_regularisation_penalty(
                trainable, _x, initial_x, regularisation_settings
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
