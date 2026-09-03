"""Coordinate non-resumable, molecule-level sampling on one node."""

from __future__ import annotations

import multiprocessing
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import datasets
import torch
from loguru import logger
from openff.toolkit import ForceField, Molecule
from rich.progress import track

from . import sample as sample_module
from .outputs import OutputType
from .sample import _SAMPLING_FNS_REGISTRY
from .settings import PreComputedDatasetSettings, SamplingSettings
from .utils._suppress_output import suppress_unwanted_output

# The device this worker process claimed at start-up; unused in the parent.
_WORKER_DEVICE = "cpu"


def sampling_devices(device_type: str, n_workers: int) -> list[str]:
    """Return round-robin logical devices for the requested workers."""
    if device_type != "cuda":
        return ["cpu"] * n_workers
    # torch.cuda.device_count() already reports only CUDA_VISIBLE_DEVICES.
    n_devices = max(1, torch.cuda.device_count())
    return [f"cuda:{i % n_devices}" for i in range(n_workers)]


def _init_worker(devices: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    """Claim one device for this worker and leave all reporting to the parent."""
    global _WORKER_DEVICE
    _WORKER_DEVICE = devices.get()
    suppress_unwanted_output()
    logger.remove()
    sample_module.track = lambda sequence, *args, **kwargs: sequence  # type: ignore[attr-defined]


def _sample_worker(
    molecule_json: str,
    molecule_index: int,
    offxml_path: str,
    device: str | None,
    sampling_settings: SamplingSettings,
    output_paths: dict[OutputType, Path],
) -> datasets.Dataset:
    """Sample one molecule, naming its side outputs with its workflow index."""
    sample_module._MOL_INDEX_OFFSET = molecule_index  # type: ignore[attr-defined]
    try:
        return _SAMPLING_FNS_REGISTRY[type(sampling_settings)](
            mols=[Molecule.from_json(molecule_json)],
            off_ff=ForceField(offxml_path),
            device=torch.device(device or _WORKER_DEVICE),
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
    # Each worker claims one device at start-up and keeps it, so oversubscribing a
    # GPU stays deliberate rather than depending on which worker picks up a molecule.
    worker_args = [
        (
            mol.to_json(),
            mol_idx,
            str(offxml_path),
            devices[0] if workers == 1 else None,
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
        try:
            pickle.dumps(sampling_settings)
        except Exception as exc:
            raise RuntimeError(
                "Parallel sampling requires picklable sampling settings. Runtime "
                "objects such as custom ASE calculators require "
                "n_sampling_processes: 1."
            ) from exc
        context = multiprocessing.get_context("spawn")
        device_queue = context.Queue()
        for device in devices:
            device_queue.put(device)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(device_queue,),
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
        raise RuntimeError(
            f"Sampling failed for {len(failures)} ligand(s): {details}"
        ) from failures[min(failures)]
    return [sampled[mol_idx] for mol_idx in range(len(mols))]
