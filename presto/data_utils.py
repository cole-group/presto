"""Utilities for creating and manipulating datasets with energy and force weights.

This module provides a custom dataset schema that extends the descent.targets.energy
schema with energy_weights and forces_weights fields for per-conformation loss
weighting, and with additive correction fields. The ``energy`` and ``forces`` fields
always hold the uncorrected reference (e.g. gas-phase MLP) values. The
``reference_*_correction`` fields are added to them to form the fitting target, and
the ``mm_*_correction`` fields are added to the MM predictions (see
`presto.loss.predict_with_weights`).
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
    from presto.settings import OutlierFilterSettings

logger = loguru.logger
_CPU_DEVICE = torch.device("cpu")

# Schema with weights for energy and forces
WEIGHTED_DATA_SCHEMA = pyarrow.schema(
    [
        ("smiles", pyarrow.string()),
        ("coords", pyarrow.list_(pyarrow.float64())),
        ("energy", pyarrow.list_(pyarrow.float64())),
        ("forces", pyarrow.list_(pyarrow.float64())),
        ("energy_weights", pyarrow.list_(pyarrow.float64())),
        ("forces_weights", pyarrow.list_(pyarrow.float64())),
        ("reference_energy_correction", pyarrow.list_(pyarrow.float64())),
        ("reference_forces_correction", pyarrow.list_(pyarrow.float64())),
        ("mm_energy_correction", pyarrow.list_(pyarrow.float64())),
        ("mm_forces_correction", pyarrow.list_(pyarrow.float64())),
        ("reference_energy_coords", pyarrow.list_(pyarrow.float64())),
    ]
)

CORRECTION_COLUMNS = (
    "reference_energy_correction",
    "reference_forces_correction",
    "mm_energy_correction",
    "mm_forces_correction",
)
"""The additive correction columns, all zero when no corrections are applied."""

_PER_CONFORMER_ENERGY_COLUMNS = (
    "energy",
    "energy_weights",
    "forces_weights",
    "reference_energy_correction",
    "mm_energy_correction",
)
_PER_CONFORMER_ATOM_COLUMNS = (
    "coords",
    "forces",
    "reference_forces_correction",
    "mm_forces_correction",
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
    Values of 0 will exclude that conformation from forces loss."""

    reference_energy_correction: typing.NotRequired[torch.Tensor]
    """Correction [kcal/mol] added to ``energy`` to form the fitting target, with
    ``shape=(n_confs,)``. Zero if absent."""

    reference_forces_correction: typing.NotRequired[torch.Tensor]
    """Correction [kcal/mol/Å] added to ``forces`` to form the fitting target, with
    ``shape=(n_confs, n_particles, 3)``. Zero if absent."""

    mm_energy_correction: typing.NotRequired[torch.Tensor]
    """Fixed correction [kcal/mol] added to the MM energies, with ``shape=(n_confs,)``.
    Zero if absent."""

    mm_forces_correction: typing.NotRequired[torch.Tensor]
    """Fixed correction [kcal/mol/Å] added to the MM forces, with
    ``shape=(n_confs, n_particles, 3)``. Zero if absent."""

    reference_energy_coords: typing.NotRequired[torch.Tensor | None]
    """The coordinates [Å] the reference energies were evaluated at, with
    ``shape=(n_confs, n_particles, 3)``, if they differ from ``coords`` (e.g. when MLP
    energies of ML-minimised structures are mapped onto MM-minimised structures).
    None if the energies were evaluated at ``coords``."""


def _flat_list(value: typing.Any) -> list[float]:
    return torch.as_tensor(value).detach().cpu().flatten().tolist()


def _entry_to_row(entry: WeightedEntry) -> dict[str, typing.Any]:
    n_confs = len(entry["energy"])
    n_coords = torch.as_tensor(entry["coords"]).numel()
    row: dict[str, typing.Any] = {
        "smiles": entry["smiles"],
        "coords": _flat_list(entry["coords"]),
        "energy": _flat_list(entry["energy"]),
        "forces": _flat_list(entry["forces"]),
        "energy_weights": _flat_list(entry["energy_weights"]),
        "forces_weights": _flat_list(entry["forces_weights"]),
    }
    for column in CORRECTION_COLUMNS:
        value = entry.get(column)
        size = n_confs if column.endswith("energy_correction") else n_coords
        row[column] = [0.0] * size if value is None else _flat_list(value)
    energy_coords = entry.get("reference_energy_coords")
    row["reference_energy_coords"] = (
        None if energy_coords is None else _flat_list(energy_coords)
    )
    return row


def create_weighted_dataset(entries: list[WeightedEntry]) -> datasets.Dataset:
    """Create a weighted dataset from a list of existing entries.

    Args:
        entries: The entries to create the dataset from. Each entry must include
            energy_weights and forces_weights tensors. Missing correction tensors are
            filled with zeros.

    Returns:
        The created dataset with torch format.
    """
    table = pyarrow.Table.from_pylist(
        [_entry_to_row(entry) for entry in entries],
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
    reference_energy_coords: torch.Tensor | None = None,
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
        reference_energy_coords: The coordinates [Å] the energies were evaluated at,
            if different from ``coords``. See `WeightedEntry.reference_energy_coords`.

    Returns:
        The created dataset with torch format. Correction columns are zero.
    """
    n_confs = len(energy)

    # Non-finite values are not handled here: zero weights cannot remove them from
    # the loss (NaN * 0 = NaN), so they are dropped by `drop_nonfinite_conformers`.
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
                "forces_weights": torch.full(
                    (n_confs,), forces_weight, dtype=torch.float64
                ),
                "reference_energy_coords": reference_energy_coords,
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

            # Make energies relative to the minimum (finite) energy within this entry
            new_entry = dict(entry)
            finite_energy = entry["energy"][torch.isfinite(entry["energy"])]
            if len(finite_energy) > 0:
                new_entry["energy"] = entry["energy"] - finite_energy.min()

            all_entries.append(new_entry)

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
    device: torch.device = _CPU_DEVICE,
    min_conformations: int = 1,
) -> datasets.Dataset:
    """Filter outliers from a dataset based on MM vs reference energy/force differences.

    Outliers are identified by comparing MM-predicted energies and forces with the
    reference values in the dataset. Conformations where the absolute difference
    exceeds a threshold are removed.

    Args:
        dataset: The dataset to filter. Must have energy, forces, and coords columns.
        force_field: The MM force field to use for computing predicted values.
        topology: The topology for the molecule in the dataset.
        settings: Outlier filter settings containing the thresholds.
        device: Device to use for computation.
        min_conformations: The minimum number of conformations to keep per entry if
            any are removed.

    Returns:
        A new dataset with outliers removed.

    Raises:
        ValueError: If the dataset is empty, or if filtering would leave an entry with
            fewer than ``min_conformations`` conformations.
    """
    from presto.loss import predict_with_weights

    if len(dataset) == 0:
        raise ValueError("Cannot filter empty dataset")

    energy_outlier_threshold = settings.energy_outlier_threshold
    force_outlier_threshold = settings.force_outlier_threshold

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
        device=device,
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

        _check_min_conformations(
            int(keep_mask.sum().item()),
            n_confs,
            min_conformations,
            f"Outlier filtering of {entry_smiles}",
        )

        # Extract kept conformations
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        n_kept = len(keep_indices)

        logger.info(f"Keeping {n_kept}/{n_confs} conformations for {entry_smiles}")

        # Keep the original (not reference-subtracted) values for the filtered entry
        filtered_entries.append(_select_conformers(entry, keep_indices.cpu()))

    # Create new dataset
    return create_weighted_dataset(typing.cast(list[WeightedEntry], filtered_entries))


def _select_conformers(entry: WeightedEntry, indices: torch.Tensor) -> WeightedEntry:
    """Return a copy of a dataset entry containing only the selected conformations.

    Args:
        entry: A single entry from a weighted dataset.
        indices: The indices of the conformations to keep.

    Returns:
        The entry with every per-conformation field indexed by ``indices``.
    """
    n_confs = len(entry["energy"])
    selected: dict[str, typing.Any] = {"smiles": entry["smiles"]}
    for column in _PER_CONFORMER_ENERGY_COLUMNS:
        value = entry.get(column)
        if value is not None:
            selected[column] = torch.as_tensor(value).cpu()[indices]
    for column in (*_PER_CONFORMER_ATOM_COLUMNS, "reference_energy_coords"):
        value = entry.get(column)
        if value is not None:
            selected[column] = (
                torch.as_tensor(value).cpu().reshape(n_confs, -1, 3)[indices]
            )
    return typing.cast(WeightedEntry, selected)


def get_corrections_from_entry(
    entry: WeightedEntry,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the additive corrections from a dataset entry, defaulting to zero.

    Args:
        entry: A single entry from a dataset.

    Returns:
        Tuple of (reference_energy_correction, reference_forces_correction,
        mm_energy_correction, mm_forces_correction). Energy corrections have
        ``shape=(n_confs,)`` and forces corrections ``shape=(n_confs, n_atoms, 3)``.
    """
    energy = torch.as_tensor(entry["energy"])
    forces = torch.as_tensor(entry["forces"]).reshape(len(energy), -1, 3)

    def _get(column: str, like: torch.Tensor) -> torch.Tensor:
        value = entry.get(column)
        if value is None:
            return torch.zeros_like(like, dtype=torch.float64)
        return torch.as_tensor(value).reshape(like.shape)

    return (
        _get("reference_energy_correction", energy),
        _get("reference_forces_correction", forces),
        _get("mm_energy_correction", energy),
        _get("mm_forces_correction", forces),
    )


def ensure_correction_columns(dataset: datasets.Dataset) -> datasets.Dataset:
    """Return a weighted dataset with every column of `WEIGHTED_DATA_SCHEMA`.

    Missing weights default to 1 and missing corrections to 0, so datasets created
    with older versions of presto or with `descent.targets.energy.create_dataset`
    can be used directly. The result is built in memory, so the source dataset (and
    any directory it was loaded from) is never modified.

    Args:
        dataset: The dataset to complete.

    Returns:
        A new in-memory dataset with torch format.
    """
    entries = []
    for entry in dataset:
        energy_weights, forces_weights = get_weights_from_entry(entry)
        ref_e, ref_f, mm_e, mm_f = get_corrections_from_entry(entry)
        n_confs = len(entry["energy"])
        entries.append(
            {
                "smiles": entry["smiles"],
                "coords": torch.as_tensor(entry["coords"]).cpu(),
                "energy": torch.as_tensor(entry["energy"]).cpu(),
                "forces": torch.as_tensor(entry["forces"]).cpu(),
                "energy_weights": torch.as_tensor(energy_weights).cpu(),
                "forces_weights": torch.as_tensor(forces_weights).cpu(),
                "reference_energy_correction": ref_e.cpu(),
                "reference_forces_correction": ref_f.cpu(),
                "mm_energy_correction": mm_e.cpu(),
                "mm_forces_correction": mm_f.cpu(),
                "reference_energy_coords": (
                    None
                    if entry.get("reference_energy_coords") is None
                    else torch.as_tensor(entry["reference_energy_coords"])
                    .cpu()
                    .reshape(n_confs, -1, 3)
                ),
            }
        )
    return create_weighted_dataset(typing.cast(list[WeightedEntry], entries))


def _check_min_conformations(
    n_kept: int, n_confs: int, min_conformations: int, action: str, hint: str = ""
) -> None:
    """Raise if removing conformations leaves fewer than ``min_conformations``.

    Entries sampled with fewer conformations are only rejected if any are removed.
    """
    if n_kept < n_confs and n_kept < min_conformations:
        raise ValueError(
            f"{action} would keep only {n_kept}/{n_confs} conformations, which is "
            f"fewer than min_conformations={min_conformations}. {hint}".rstrip()
        )


def drop_nonfinite_conformers(
    dataset: datasets.Dataset, min_conformations: int = 1
) -> datasets.Dataset:
    """Remove conformations with non-finite energies, forces or corrections.

    Zero weights cannot remove non-finite values from the loss (``NaN * 0 = NaN``),
    and a single non-finite energy also corrupts the per-entry energy referencing, so
    such conformations are removed instead.

    Args:
        dataset: The weighted dataset to clean.
        min_conformations: The minimum number of conformations to keep per entry if
            any are removed.

    Returns:
        The dataset with non-finite conformations removed. The input dataset is
        returned unchanged if every value is finite.

    Raises:
        ValueError: If removing the non-finite conformations would leave an entry with
            fewer than ``min_conformations`` conformations.
    """
    cleaned_entries = []
    n_dropped_total = 0

    for entry_idx, entry in enumerate(dataset):
        n_confs = len(entry["energy"])
        finite = torch.ones(n_confs, dtype=torch.bool)
        for column in ("energy", "reference_energy_correction", "mm_energy_correction"):
            if entry.get(column) is not None:
                finite &= torch.isfinite(torch.as_tensor(entry[column]).cpu())
        for column in (
            "forces",
            "reference_forces_correction",
            "mm_forces_correction",
        ):
            if entry.get(column) is not None:
                values = torch.as_tensor(entry[column]).cpu().reshape(n_confs, -1)
                finite &= torch.isfinite(values).all(dim=1)

        n_dropped = n_confs - int(finite.sum())
        if n_dropped > 0:
            _check_min_conformations(
                n_confs - n_dropped,
                n_confs,
                min_conformations,
                f"Removing {n_dropped} conformations with non-finite energies, forces "
                f"or corrections from entry {entry_idx} of {entry['smiles']}",
                hint="Check the reference potential and corrections.",
            )
            logger.warning(
                f"Removing {n_dropped}/{n_confs} conformations with non-finite "
                f"energies, forces or corrections from entry {entry_idx} of "
                f"{entry['smiles']}."
            )
        n_dropped_total += n_dropped
        cleaned_entries.append(
            _select_conformers(entry, finite.nonzero(as_tuple=True)[0])
        )

    if n_dropped_total == 0:
        return dataset

    return create_weighted_dataset(cleaned_entries)
