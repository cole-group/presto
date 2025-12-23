"""
Utilities for creating and manipulating datasets with energy and force weights.

This module provides a custom dataset schema that extends the descent.targets.energy
schema with energy_weights and forces_weights fields for per-conformation loss weighting.
"""

import typing

import datasets
import datasets.table
import numpy as np
import pyarrow
import torch

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

    return create_weighted_dataset(all_entries)


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
    entry: dict, default_energy_weight: float = 1.0, default_forces_weight: float = 1.0
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
