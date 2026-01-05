"""Utilities for creating and manipulating datasets with energy and force weights.

This module provides a custom dataset schema that extends the descent.targets.energy
schema with energy_weights and forces_weights fields for per-conformation loss weighting.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import datasets
import datasets.table
import loguru
import pyarrow
import smee
import smee.converters
import torch

if TYPE_CHECKING:
    from bespokefit_smee.settings import OutlierFilterSettings

logger = loguru.logger

# Schema with weights for energy and forces
WEIGHTED_DATA_SCHEMA = pyarrow.schema(
    [
        ("smiles", pyarrow.string()),
        ("coords", pyarrow.list_(pyarrow.float64())),
        ("energy", pyarrow.list_(pyarrow.float64())),
        ("forces", pyarrow.list_(pyarrow.float64())),
        ("energy_weights", pyarrow.list_(pyarrow.float64())),
        ("forces_weights", pyarrow.list_(pyarrow.float64())),
    ]
)


class WeightedEntry(typing.TypedDict):
    """Represents a set of reference energies and forces with associated weights."""

    smiles: str
    """The indexed SMILES description of the molecule the energies and forces were
    computed for."""

    coords: torch.Tensor
    """The coordinates [Å] the energies and forces were evaluated at with
    ``shape=(n_confs, n_particles, 3)``."""

    energy: torch.Tensor
    """The reference energies [kcal/mol] with ``shape=(n_confs,)``."""

    forces: torch.Tensor
    """The reference forces [kcal/mol/Å] with ``shape=(n_confs, n_particles, 3)``."""

    energy_weights: torch.Tensor
    """The weights for energy loss contributions with ``shape=(n_confs,)``.
    Values of 0 will exclude that conformation from energy loss."""

    forces_weights: torch.Tensor
    """The weights for forces loss contributions with ``shape=(n_confs,)``.
    Values of 0 will exclude that conformation from forces loss.
    NaN forces should have weight 0."""


def create_weighted_dataset(entries: list[WeightedEntry]) -> datasets.Dataset:
    """Create a weighted dataset from a list of existing entries.

    Args:
        entries: The entries to create the dataset from. Each entry must include
            energy_weights and forces_weights tensors.

    Returns:
        The created dataset with torch format.
    """
    table = pyarrow.Table.from_pylist(
        [
            {
                "smiles": entry["smiles"],
                "coords": torch.tensor(entry["coords"]).flatten().tolist(),
                "energy": torch.tensor(entry["energy"]).flatten().tolist(),
                "forces": torch.tensor(entry["forces"]).flatten().tolist(),
                "energy_weights": torch.tensor(entry["energy_weights"])
                .flatten()
                .tolist(),
                "forces_weights": torch.tensor(entry["forces_weights"])
                .flatten()
                .tolist(),
            }
            for entry in entries
        ],
        schema=WEIGHTED_DATA_SCHEMA,
    )

    dataset = datasets.Dataset(datasets.table.InMemoryTable(table))
    dataset.set_format("torch")

    return dataset


def create_dataset_with_uniform_weights(
    smiles: str,
    coords: torch.Tensor,
    energy: torch.Tensor,
    forces: torch.Tensor,
    energy_weight: float = 1.0,
    forces_weight: float = 1.0,
) -> datasets.Dataset:
    """Create a weighted dataset with uniform weights for all conformations.

    This is a convenience function for creating datasets where all conformations
    have the same weight.

    Args:
        smiles: The indexed SMILES description of the molecule.
        coords: The coordinates [Å] with ``shape=(n_confs, n_particles, 3)``.
        energy: The reference energies [kcal/mol] with ``shape=(n_confs,)``.
        forces: The reference forces [kcal/mol/Å] with ``shape=(n_confs, n_particles, 3)``.
        energy_weight: The uniform weight for energy loss contributions.
        forces_weight: The uniform weight for forces loss contributions.

    Returns:
        The created dataset with torch format.
    """
    n_confs = len(energy)

    # Check if forces contain NaN - set forces_weight to 0 for those conformations
    forces_reshaped = forces.reshape(n_confs, -1, 3) if forces.dim() == 2 else forces
    forces_weights_array = torch.full((n_confs,), forces_weight, dtype=torch.float64)

    for i in range(n_confs):
        if torch.isnan(forces_reshaped[i]).any():
            forces_weights_array[i] = 0.0

    return create_weighted_dataset(
        [
            {
                "smiles": smiles,
                "coords": coords,
                "energy": energy,
                "forces": forces,
                "energy_weights": torch.full(
                    (n_confs,), energy_weight, dtype=torch.float64
                ),
                "forces_weights": forces_weights_array,
            }
        ]
    )


def merge_weighted_datasets(datasets_list: list[datasets.Dataset]) -> datasets.Dataset:
    """Merge multiple weighted datasets into a single dataset with separate entries.

    Each dataset's entries are preserved as separate entries in the output dataset.
    This allows energies to be computed relative to the minimum within each entry
    rather than across all conformations.

    Args:
        datasets_list: List of datasets to merge.

    Returns:
        A single dataset containing all entries from all input datasets.

    Raises:
        ValueError: If datasets have different SMILES strings.
    """
    if not datasets_list:
        raise ValueError("Cannot merge empty list of datasets")

    if len(datasets_list) == 1:
        return datasets_list[0]

    # Get the SMILES from the first dataset
    first_entry = datasets_list[0][0]
    smiles = first_entry["smiles"]

    all_entries = []

    for ds in datasets_list:
        for entry in ds:
            if entry["smiles"] != smiles:
                raise ValueError(
                    f"All datasets must have the same SMILES. "
                    f"Found {smiles} and {entry['smiles']}"
                )

            n_confs = len(entry["energy"])
            coords = entry["coords"].reshape(n_confs, -1, 3)
            forces = entry["forces"].reshape(n_confs, -1, 3)

            # Make energies relative to minimum within this entry
            energy = entry["energy"] - entry["energy"].min()

            all_entries.append(
                {
                    "smiles": smiles,
                    "coords": coords,
                    "energy": energy,
                    "forces": forces,
                    "energy_weights": entry["energy_weights"],
                    "forces_weights": entry["forces_weights"],
                }
            )

    return create_weighted_dataset(typing.cast(list[WeightedEntry], all_entries))


def has_weights(dataset: datasets.Dataset) -> bool:
    """Check if a dataset has weight columns.

    Args:
        dataset: The dataset to check.

    Returns:
        True if the dataset has energy_weights and forces_weights columns.
    """
    if len(dataset) == 0:
        return False

    entry = dataset[0]
    return "energy_weights" in entry and "forces_weights" in entry


def get_weights_from_entry(
    entry: WeightedEntry,
    default_energy_weight: float = 1.0,
    default_forces_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get energy and forces weights from a dataset entry.

    If the entry doesn't have weights, returns uniform weights with the specified defaults.

    Args:
        entry: A single entry from a dataset.
        default_energy_weight: Default weight if energy_weights not present.
        default_forces_weight: Default weight if forces_weights not present.

    Returns:
        Tuple of (energy_weights, forces_weights) tensors.
    """
    n_confs = len(entry["energy"])

    if "energy_weights" in entry:
        energy_weights = entry["energy_weights"]
    else:
        energy_weights = torch.full(
            (n_confs,), default_energy_weight, dtype=torch.float64
        )

    if "forces_weights" in entry:
        forces_weights = entry["forces_weights"]
    else:
        forces_weights = torch.full(
            (n_confs,), default_forces_weight, dtype=torch.float64
        )

    return energy_weights, forces_weights


def filter_dataset_outliers(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topology: smee.TensorTopology,
    settings: OutlierFilterSettings,
    device: str = "cpu",
) -> datasets.Dataset:
    """Filter outliers from a dataset based on MM vs reference energy/force differences.

    Outliers are identified by comparing MM-predicted energies and forces with the
    reference values in the dataset. Conformations where the absolute difference
    exceeds a threshold are removed.

    Args:
        dataset: The dataset to filter. Must have energy, forces, and coords columns.
        force_field: The MM force field to use for computing predicted values.
        topology: The topology for the molecule in the dataset.
        settings: Outlier filter settings containing thresholds and min_conformations.
        device: Device to use for computation ("cpu" or "cuda").

    Returns:
        A new dataset with outliers removed.

    Raises:
        ValueError: If the dataset is empty or has no conformations after filtering.
    """
    from bespokefit_smee.loss import predict_with_weights

    if len(dataset) == 0:
        raise ValueError("Cannot filter empty dataset")

    energy_outlier_threshold = settings.energy_outlier_threshold
    force_outlier_threshold = settings.force_outlier_threshold
    min_conformations = settings.min_conformations

    if energy_outlier_threshold is None and force_outlier_threshold is None:
        logger.info("No outlier filtering enabled, returning original dataset")
        return dataset

    # Build topologies dict and track entry sizes for un-flattening
    smiles = dataset[0]["smiles"]
    topologies = {smiles: topology}
    n_atoms = topology.n_atoms

    # Get predictions using predict_with_weights (no graph needed, no normalization)
    energy_ref, energy_pred, forces_ref, forces_pred, _, _ = predict_with_weights(
        dataset,
        force_field,
        topologies,
        reference="median",
        normalize=False,
        device_type=device,
        create_graph=False,
    )

    # Track entry sizes for un-flattening
    entry_sizes = [len(entry["energy"]) for entry in dataset]

    # Un-flatten results back to per-entry lists
    energy_ref_list = torch.split(energy_ref, entry_sizes)
    energy_pred_list = torch.split(energy_pred, entry_sizes)
    # Forces are (n_total_confs * n_atoms, 3), need to split by n_confs * n_atoms
    forces_split_sizes = [n * n_atoms for n in entry_sizes]
    forces_ref_list = torch.split(forces_ref, forces_split_sizes)
    forces_pred_list = torch.split(forces_pred, forces_split_sizes)

    filtered_entries = []

    for i, entry in enumerate(dataset):
        entry_smiles = entry["smiles"]
        n_confs = entry_sizes[i]

        # Get per-entry predictions
        entry_energy_ref = energy_ref_list[i]
        entry_energy_pred = energy_pred_list[i]
        entry_forces_ref = forces_ref_list[i].reshape(n_confs, n_atoms, 3)
        entry_forces_pred = forces_pred_list[i].reshape(n_confs, n_atoms, 3)

        # Compute energy differences (already reference-subtracted)
        energy_diff = torch.abs(entry_energy_ref - entry_energy_pred).detach()

        # Per-atom energy difference
        energy_diff_per_atom = energy_diff / n_atoms

        # Compute force differences (max absolute difference per conformation)
        forces_diff = torch.abs(entry_forces_ref - entry_forces_pred).detach()
        forces_diff_max = forces_diff.reshape(n_confs, -1).max(dim=1).values

        # Identify outliers using absolute thresholds
        keep_mask = torch.ones(n_confs, dtype=torch.bool, device=device)

        if energy_outlier_threshold is not None:
            energy_outliers = energy_diff_per_atom > energy_outlier_threshold
            keep_mask &= ~energy_outliers
            n_energy_outliers = energy_outliers.sum().item()
            if n_energy_outliers > 0:
                logger.info(
                    f"Found {n_energy_outliers} energy outliers "
                    f"(threshold: {energy_outlier_threshold:.4f} kcal/mol/atom)"
                )

        if force_outlier_threshold is not None:
            forces_outliers = forces_diff_max > force_outlier_threshold
            keep_mask &= ~forces_outliers
            n_forces_outliers = forces_outliers.sum().item()
            if n_forces_outliers > 0:
                logger.info(
                    f"Found {n_forces_outliers} force outliers "
                    f"(threshold: {force_outlier_threshold:.4f} kcal/mol/Å)"
                )

        # Ensure we keep at least min_conformations
        n_kept = keep_mask.sum().item()
        if n_kept < min_conformations:
            logger.warning(
                f"Filtering would keep only {n_kept} conformations, "
                f"but min_conformations={min_conformations}. "
                f"Keeping all {n_confs} conformations for this entry."
            )
            keep_mask = torch.ones(n_confs, dtype=torch.bool, device=device)

        # Extract kept conformations
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        n_kept = len(keep_indices)

        if n_kept == 0:
            logger.warning(
                f"All conformations filtered for {entry_smiles}, keeping all"
            )
            keep_indices = torch.arange(n_confs, device=device)
            n_kept = n_confs

        logger.info(f"Keeping {n_kept}/{n_confs} conformations for {entry_smiles}")

        # Get original reference values (not reference-subtracted) for the filtered entry
        original_energy = entry["energy"].to(device)
        original_forces = entry["forces"].reshape(n_confs, n_atoms, 3).to(device)
        coords = entry["coords"].reshape(n_confs, n_atoms, 3)

        # Build filtered entry
        filtered_entry: dict[str, typing.Any] = {
            "smiles": entry_smiles,
            "coords": coords[keep_indices].cpu(),
            "energy": original_energy[keep_indices].cpu(),
            "forces": original_forces[keep_indices].cpu(),
        }

        # Preserve weights if present
        if "energy_weights" in entry:
            filtered_entry["energy_weights"] = entry["energy_weights"][keep_indices]
        if "forces_weights" in entry:
            filtered_entry["forces_weights"] = entry["forces_weights"][keep_indices]

        filtered_entries.append(filtered_entry)

    # Create new dataset
    return create_weighted_dataset(typing.cast(list[WeightedEntry], filtered_entries))
