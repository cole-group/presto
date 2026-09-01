"""Coordinate resumable, molecule-level sampling on one node."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import datasets
import torch
from openff.toolkit import ForceField, Molecule
from rich.progress import track

from .outputs import OutputType, get_mol_path
from .sample import _SAMPLING_FNS_REGISTRY, load_precomputed_dataset
from .settings import PreComputedDatasetSettings, SamplingSettings


def sampling_devices(device_type: str, n_workers: int) -> list[str]:
    """Return round-robin logical devices for the requested workers."""
    if device_type != "cuda":
        return ["cpu"] * n_workers
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    n_devices = len([item for item in visible.split(",") if item.strip()]) if visible else torch.cuda.device_count()
    n_devices = max(1, n_devices)
    return [f"cuda:{i % n_devices}" for i in range(n_workers)]


def _atomic_save(dataset: datasets.Dataset, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        dataset.save_to_disk(str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_dataset(path: Path, description: str) -> datasets.Dataset:
    try:
        return datasets.load_from_disk(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"The {description} at {path} is corrupt or incomplete. Run `presto clean` "
            "or use a new output directory."
        ) from exc


def _side_outputs_exist(output_paths: dict[OutputType, Path], mol_idx: int) -> bool:
    return all(get_mol_path(path, mol_idx).exists() for path in output_paths.values())


def _sample_worker(
    molecule_json: str,
    molecule_index: int,
    offxml_path: str,
    device: str,
    sampling_settings: SamplingSettings,
    output_paths: dict[OutputType, Path],
    cache_path: Path,
) -> int:
    # Workers own all OpenMM, CUDA, and force-field state. Rich's nested progress
    # displays are disabled because the parent owns progress reporting.
    from . import sample as sample_module

    old_track = sample_module.track  # type: ignore[attr-defined]
    old_index = sample_module._COORDINATED_MOLECULE_INDEX  # type: ignore[attr-defined]
    try:
        sample_module.track = lambda sequence, *args, **kwargs: sequence  # type: ignore[attr-defined]
        sample_module._COORDINATED_MOLECULE_INDEX = molecule_index  # type: ignore[attr-defined]
        molecule = Molecule.from_json(molecule_json)
        force_field = ForceField(offxml_path)
        sample_fn = _SAMPLING_FNS_REGISTRY[type(sampling_settings)]
        result = sample_fn(
            mols=[molecule],
            off_ff=force_field,
            device=torch.device(device),
            settings=sampling_settings,
            output_paths=output_paths,
        )[0]
        _atomic_save(result, cache_path)
        return molecule_index
    finally:
        sample_module.track = old_track  # type: ignore[attr-defined]
        sample_module._COORDINATED_MOLECULE_INDEX = old_index  # type: ignore[attr-defined]


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
    """Sample missing ligands, process them in the parent, and commit atomically."""
    sample_fn = _SAMPLING_FNS_REGISTRY[type(sampling_settings)]
    if sample_fn is load_precomputed_dataset or isinstance(sampling_settings, PreComputedDatasetSettings):
        loaded_datasets = sample_fn(
            mols=mols,
            off_ff=ForceField(str(offxml_path)),
            device=torch.device(device_type),
            settings=sampling_settings,
            output_paths=output_paths,
        )
        if process_dataset is not None:
            loaded_datasets = [
                process_dataset(mol_idx, dataset)
                for mol_idx, dataset in enumerate(loaded_datasets)
            ]
        return loaded_datasets

    cache_dir = canonical_paths[0].parent / ".sampling_cache"
    cache_paths = [cache_dir / f"raw_mol{i}" for i in range(len(mols))]
    results: list[datasets.Dataset | None] = [None] * len(mols)
    missing: list[int] = []
    for i, canonical in enumerate(canonical_paths):
        if canonical.exists():
            if not _side_outputs_exist(output_paths, i):
                raise RuntimeError(
                    f"Dataset {canonical} exists but required sampling side outputs are missing. "
                    "Run `presto clean` or use a new output directory."
                )
            results[i] = _load_dataset(canonical, "committed sampling dataset")
        else:
            missing.append(i)

    if not missing:
        return [item for item in results if item is not None]

    # A cache is reusable only after protocol side outputs were completed.
    to_sample: list[int] = []
    raw: dict[int, datasets.Dataset] = {}
    for i in missing:
        if cache_paths[i].exists() and _side_outputs_exist(output_paths, i):
            raw[i] = _load_dataset(cache_paths[i], "sampling cache")
        else:
            if cache_paths[i].exists():
                shutil.rmtree(cache_paths[i])
            to_sample.append(i)

    workers = min(n_processes, len(to_sample))
    if workers > 1:
        try:
            pickle.dumps((sampling_settings, [mols[i].to_json() for i in to_sample]))
        except Exception as exc:
            raise RuntimeError(
                "Parallel sampling requires spawn-serializable sampling settings and molecules. "
                "Runtime objects such as custom ASE calculators require n_sampling_processes: 1."
            ) from exc

    failures: dict[int, BaseException] = {}
    devices = sampling_devices(device_type, max(1, workers))
    if workers <= 1:
        for i in to_sample:
            try:
                _sample_worker(mols[i].to_json(), i, str(offxml_path), devices[0], sampling_settings, output_paths, cache_paths[i])
                raw[i] = _load_dataset(cache_paths[i], "sampling cache")
            except BaseException as exc:
                failures[i] = exc
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {
                executor.submit(
                    _sample_worker, mols[i].to_json(), i, str(offxml_path),
                    devices[position % len(devices)], sampling_settings, output_paths, cache_paths[i]
                ): i
                for position, i in enumerate(to_sample)
            }
            for future in track(as_completed(futures), total=len(futures), description="Ligands completed"):
                i = futures[future]
                try:
                    future.result()
                    raw[i] = _load_dataset(cache_paths[i], "sampling cache")
                except BaseException as exc:
                    failures[i] = exc

    if failures:
        details = "; ".join(f"molecule {i}: {exc}" for i, exc in sorted(failures.items()))
        raise RuntimeError(f"Sampling failed for {len(failures)} ligand(s): {details}")

    for i in missing:
        dataset = raw[i]
        if process_dataset is not None:
            dataset = process_dataset(i, dataset)
        _atomic_save(dataset, canonical_paths[i])
        results[i] = dataset
    return [item for item in results if item is not None]


def remove_sampling_temporary_paths(stage_path: Path) -> None:
    """Remove coordinator caches and interrupted atomic-save paths."""
    shutil.rmtree(stage_path / ".sampling_cache", ignore_errors=True)
    for path in stage_path.glob(".*.tmp-*"):
        shutil.rmtree(path, ignore_errors=True)
