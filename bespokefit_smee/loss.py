"""
Loss functions for tuning the forcefield
"""

import typing

import datasets
import descent
import descent.optim
import descent.train
import descent.utils.loss
import descent.targets.energy
import loguru
import smee
import smee.utils
import torch

from bespokefit_smee.utils.typing import ValenceType

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

    # L2 regularisation on all parameters
    reg_loss += (
        ((trainable_parameters[regularised_idxs] - target) ** 2)
        * regularisation_weights
    ).sum() / n_atoms

    return reg_loss


def get_loss_closure_fn(
    dataset: datasets.Dataset,
    trainable: descent.train.Trainable,
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    topology: smee.TensorTopology,
    loss_energy_weight: float,
    loss_force_weight: float,
    regularisation_target: typing.Literal["initial", "zero"],
    # regularisation_settings: RegularisationSettings,
) -> descent.optim.ClosureFn:
    """
    Return a default closure function

    Args:
        dataset: The dataset to predict the energies and forces of.
        trainable: The trainable object containing the force field.
        trainable_parameters: The parameters to be optimized.
        initial_parameters: The initial parameters before training.
        topology: The topology of the molecule in the dataset.
        loss_energy_weight: Weight for the energy loss term.
        loss_force_weight: Weight for the force loss term.
        regularisation_target: The type of regularisation to apply ('initial' or 'zero').

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
                initial_parameters,
                regularisation_target=regularisation_target,
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


def filter_dataset_by_hydrogen_force_errors(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topology: smee.TensorTopology,
    threshold_multiplier: float = 4.0,
    device_type: str = "cpu",
) -> datasets.Dataset:
    """Filter dataset to remove conformations with large hydrogen force errors.

    This function should be called once per training iteration to filter the dataset
    before training begins. It:
    a) Computes the magnitude of force differences for each atom
    b) Computes the median force difference for all carbon atoms
    c) Filters out conformations where any hydrogen has force error > threshold_multiplier
       times the median carbon force difference

    Args:
        dataset: The dataset containing energies, forces, and coordinates
        force_field: The force field to use for predictions
        topology: The topology containing atomic numbers
        threshold_multiplier: Multiplier for median carbon force error threshold (default: 4.0)
        device_type: The device type (e.g., 'cpu' or 'cuda')

    Returns:
        Filtered dataset with conformations removed that have excessive hydrogen force errors
    """
    # Get predictions for the current force field
    energy_ref_all, energy_pred_all, forces_ref_all, forces_pred_all = predict(
        dataset,
        force_field,
        {dataset[0]["smiles"]: topology},
        device_type=device_type,
        normalize=False,
    )

    # Reshape forces for filtering
    n_confs = len(dataset[0]["energy"])
    forces_ref_reshaped = forces_ref_all.reshape(n_confs, -1, 3)
    forces_pred_reshaped = forces_pred_all.reshape(n_confs, -1, 3)

    # Compute force difference magnitude for each atom
    # Shape: (n_confs, n_atoms)
    force_diff_magnitude = torch.norm(forces_ref_reshaped - forces_pred_reshaped, dim=2)

    # Get atomic numbers from topology
    atomic_numbers = topology.atomic_nums

    # Create masks for carbon (atomic number 6) and hydrogen (atomic number 1)
    carbon_mask = atomic_numbers == 6
    hydrogen_mask = atomic_numbers == 1

    # Compute median force difference for all carbon atoms across all conformations
    carbon_force_diffs = force_diff_magnitude[:, carbon_mask]
    median_carbon_force_diff = torch.median(carbon_force_diffs)

    # Compute threshold
    threshold = threshold_multiplier * median_carbon_force_diff

    # Check if any hydrogen in each conformation exceeds the threshold
    hydrogen_force_diffs = force_diff_magnitude[:, hydrogen_mask]

    # For each conformation, check if max hydrogen force error exceeds threshold
    max_hydrogen_error_per_conf = hydrogen_force_diffs.max(dim=1).values

    # Keep conformations where max hydrogen error is below threshold
    keep_mask = max_hydrogen_error_per_conf <= threshold

    # Convert to numpy for indexing
    keep_indices = keep_mask.cpu().numpy()

    n_confs_discarded = n_confs - keep_indices.sum()
    if n_confs_discarded > 0:
        logger.info(
            f"Filtered out {n_confs_discarded}/{n_confs} conformations due to large hydrogen force errors"
        )

    # Filter the arrays within the dataset entry
    return descent.targets.energy.create_dataset(
        [
            {
                "smiles": dataset[0]["smiles"],
                "energy": dataset[0]["energy"][keep_indices],
                "forces": dataset[0]["forces"]
                .reshape(n_confs, -1, 3)[keep_indices]
                .flatten(),
                "coords": dataset[0]["coords"]
                .reshape(n_confs, -1, 3)[keep_indices]
                .flatten(),
            }
        ]
    )


# TODO: Move the following two functions back into smee (they are copied with minor changes)


def compute_energy(
    system: smee.TensorSystem | smee.TensorTopology,
    force_field: smee.TensorForceField,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
    vdw_energy_threshold: float | None = None,
) -> torch.Tensor:
    """Computes the potential energy [kcal / mol] of a system / topology in a given
    conformation(s).

    Args:
        system: The system or topology to compute the potential energy of.
        force_field: The force field that defines the potential energy function.
        conformer: The conformer(s) to evaluate the potential at with
            ``shape=(n_particles, 3)`` or ``shape=(n_confs, n_particles, 3)``.
        box_vectors: The box vectors of the system with ``shape=(3, 3)`` if the
            system is periodic, or ``None`` if the system is non-periodic.
        vdw_energy_threshold: Per-atom threshold for including van der Waals energy
            contributions. If ``None``, all contributions are included. If set,
            energies will be set to NaN when the per-atom van der Waals energy exceeds
            this threshold.

    Returns:
        The potential energy of the conformer(s) [kcal / mol].
    """
    import importlib
    from smee.potentials._potentials import (
        _precompute_pairwise,
        _prepare_inputs,
        compute_energy_potential,
    )

    # register the built-in potential energy functions
    importlib.import_module("smee.potentials.nonbonded")
    importlib.import_module("smee.potentials.valence")

    system, conformer, box_vectors = _prepare_inputs(system, conformer, box_vectors)
    pairwise = _precompute_pairwise(system, force_field, conformer, box_vectors)

    total_energy = smee.utils.zeros_like(
        conformer.shape[0] if conformer.ndim == 3 else 1, conformer
    )

    for potential in force_field.potentials:

        energy = compute_energy_potential(
            system, potential, conformer, box_vectors, pairwise
        )
        # print(potential.type)
        # print(energy - min(energy).item())
        # breakpoint()

        # if potential.type == "vdW":  # and vdw_energy_threshold is not None:
        # per_atom_vdw_energy = energy / system.n_particles
        # energy = torch.where(
        #     per_atom_vdw_energy > vdw_energy_threshold,
        #     torch.tensor(float("nan"), device=energy.device),
        #     energy,
        # )
        # continue

        total_energy += energy

        # energy += compute_energy_potential(
        #     system, potential, conformer, box_vectors, pairwise
        # )

    return total_energy


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

        energy_pred = compute_energy(topology, force_field, coords)
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
