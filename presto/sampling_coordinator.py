"""Coordinate non-resumable, molecule-level sampling on one node."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import datasets
import torch
from openff.toolkit import ForceField, Molecule
from rich.progress import track

from .outputs import OutputType
from .sample import _SAMPLING_FNS_REGISTRY
from .settings import PreComputedDatasetSettings, SamplingSettings


def sampling_devices(device_type: str, n_workers: int) -> list[str]:
    """Return round-robin logical devices for the requested workers."""
    if device_type != "cuda":
        return ["cpu"] * n_workers
    # torch.cuda.device_count() already reports only CUDA_VISIBLE_DEVICES.
    n_devices = max(1, torch.cuda.device_count())
    return [f"cuda:{i % n_devices}" for i in range(n_workers)]


def _sample_worker(
    molecule_json: str,
    molecule_index: int,
    offxml_path: str,
    device: str,
    sampling_settings: SamplingSettings,
    output_paths: dict[OutputType, Path],
) -> datasets.Dataset:
    """Sample one molecule, naming its side outputs with its workflow index."""
    from . import sample as sample_module

    if multiprocessing.parent_process() is not None:
        # The parent owns progress reporting, and a spawned worker is discarded
        # after one molecule, so this is not restored.
        sample_module.track = lambda sequence, *args, **kwargs: sequence  # type: ignore[attr-defined]
    sample_module._MOL_INDEX_OFFSET = molecule_index  # type: ignore[attr-defined]
    try:
        return _SAMPLING_FNS_REGISTRY[type(sampling_settings)](
            mols=[Molecule.from_json(molecule_json)],
            off_ff=ForceField(offxml_path),
            device=torch.device(device),
            settings=sampling_settings,
            output_paths=output_paths,
        )[0]
    finally:
        sample_module._MOL_INDEX_OFFSET = 0  # type: ignore[attr-defined]


def sample_ligands(
    *,
    mols: list[Molecule],
    offxml_path: Path,
    device_type: str,
    sampling_settings: SamplingSettings,
    output_paths: dict[OutputType, Path],
    canonical_paths: list[Path],
    n_processes: int,
    process_dataset: Callable[[int, datasets.Dataset], datasets.Dataset] | None = None,
) -> list[datasets.Dataset]:
    """Sample every ligand, process it in the parent, and save it.

    Generated sampling is an internal, non-resumable workflow operation: the
    workflow rejects non-clean output before calling it, so destinations are absent.
    """
    precomputed = isinstance(sampling_settings, PreComputedDatasetSettings)
    if precomputed:
        raw = _SAMPLING_FNS_REGISTRY[type(sampling_settings)](
            mols=mols,
            off_ff=ForceField(str(offxml_path)),
            device=torch.device(device_type),
            settings=sampling_settings,
            output_paths=output_paths,
        )
    else:
        raw = _sample_every_ligand(
            mols=mols,
            offxml_path=offxml_path,
            device_type=device_type,
            sampling_settings=sampling_settings,
            output_paths=output_paths,
            n_processes=n_processes,
        )

    results = [
        dataset if process_dataset is None else process_dataset(mol_idx, dataset)
        for mol_idx, dataset in enumerate(raw)
    ]
    if not precomputed:
        for dataset, path in zip(results, canonical_paths, strict=True):
            dataset.save_to_disk(str(path))
    return results


def _sample_every_ligand(
    *,
    mols: list[Molecule],
    offxml_path: Path,
    device_type: str,
    sampling_settings: SamplingSettings,
    output_paths: dict[OutputType, Path],
    n_processes: int,
) -> list[datasets.Dataset]:
    """Sample every molecule, reporting all per-ligand failures together."""
    workers = max(1, min(n_processes, len(mols)))
    devices = sampling_devices(device_type, workers)
    worker_args = [
        (
            mol.to_json(),
            mol_idx,
            str(offxml_path),
            devices[mol_idx % workers],
            sampling_settings,
            output_paths,
        )
        for mol_idx, mol in enumerate(mols)
    ]

    sampled: dict[int, datasets.Dataset] = {}
    failures: dict[int, Exception] = {}
    if workers == 1:
        for mol_idx, args in enumerate(worker_args):
            try:
                sampled[mol_idx] = _sample_worker(*args)
            except Exception as exc:
                failures[mol_idx] = exc
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(_sample_worker, *args): mol_idx
                for mol_idx, args in enumerate(worker_args)
            }
            for future in track(
                as_completed(futures),
                total=len(futures),
                description="Ligands completed",
            ):
                try:
                    sampled[futures[future]] = future.result()
                except Exception as exc:
                    failures[futures[future]] = exc

    if failures:
        details = "; ".join(
            f"molecule {i}: {exc}" for i, exc in sorted(failures.items())
        )
        raise RuntimeError(f"Sampling failed for {len(failures)} ligand(s): {details}")
    return [sampled[mol_idx] for mol_idx in range(len(mols))]
