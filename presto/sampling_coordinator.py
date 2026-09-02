"""Coordinate non-resumable, molecule-level sampling on one node."""

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
    n_devices = (
        len([item for item in visible.split(",") if item.strip()])
        if visible
        else torch.cuda.device_count()
    )
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


def _validate_sampling_inputs(
    *,
    mols: list[Molecule],
    sampling_settings: SamplingSettings,
    output_paths: dict[OutputType, Path],
    canonical_paths: list[Path],
    n_processes: int,
) -> None:
    """Validate the structure of generated-sampling inputs."""
    if not mols:
        raise ValueError("At least one molecule is required for generated sampling.")
    if len(canonical_paths) != len(mols):
        raise ValueError(
            "Generated sampling requires exactly one canonical dataset path per "
            f"molecule; received {len(canonical_paths)} paths for {len(mols)} molecules."
        )
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ValueError("Canonical dataset paths must be unique for each molecule.")
    if set(output_paths) != sampling_settings.output_types:
        raise ValueError(
            "Sampling output paths must contain exactly the output types configured "
            f"for the protocol: {sampling_settings.output_types}."
        )
    if n_processes < 1:
        raise ValueError("n_processes must be at least 1.")
    side_output_paths = [
        get_mol_path(base_path, mol_idx)
        for base_path in output_paths.values()
        for mol_idx in range(len(mols))
    ]
    all_output_paths = [*canonical_paths, *side_output_paths]
    if len(set(all_output_paths)) != len(all_output_paths):
        raise ValueError(
            "Canonical dataset and molecule-specific output paths must be unique."
        )


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
    """Sample every ligand, process it in the parent, and commit atomically.

    Generated sampling is an internal workflow operation. Its destinations must be
    absent because the workflow rejects non-clean output before calling it.
    """
    sample_fn = _SAMPLING_FNS_REGISTRY[type(sampling_settings)]
    if sample_fn is load_precomputed_dataset or isinstance(
        sampling_settings, PreComputedDatasetSettings
    ):
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

    _validate_sampling_inputs(
        mols=mols,
        sampling_settings=sampling_settings,
        output_paths=output_paths,
        canonical_paths=canonical_paths,
        n_processes=n_processes,
    )
    cache_dir = canonical_paths[0].parent / ".sampling_cache"
    cache_paths = [cache_dir / f"raw_mol{i}" for i in range(len(mols))]
    workers = min(n_processes, len(mols))
    if workers > 1:
        try:
            pickle.dumps((sampling_settings, [molecule.to_json() for molecule in mols]))
        except Exception as exc:
            raise RuntimeError(
                "Parallel sampling requires spawn-serializable sampling settings and molecules. "
                "Runtime objects such as custom ASE calculators require n_sampling_processes: 1."
            ) from exc

    raw: dict[int, datasets.Dataset] = {}
    cache_dir.mkdir(parents=True)
    try:
        failures: dict[int, Exception] = {}
        devices = sampling_devices(device_type, workers)
        if workers == 1:
            for i, molecule in enumerate(mols):
                try:
                    _sample_worker(
                        molecule.to_json(),
                        i,
                        str(offxml_path),
                        devices[0],
                        sampling_settings,
                        output_paths,
                        cache_paths[i],
                    )
                    raw[i] = _load_dataset(cache_paths[i], "sampling cache")
                except Exception as exc:
                    failures[i] = exc
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context
            ) as executor:
                futures = {
                    executor.submit(
                        _sample_worker,
                        molecule.to_json(),
                        i,
                        str(offxml_path),
                        devices[i % len(devices)],
                        sampling_settings,
                        output_paths,
                        cache_paths[i],
                    ): i
                    for i, molecule in enumerate(mols)
                }
                for future in track(
                    as_completed(futures),
                    total=len(futures),
                    description="Ligands completed",
                ):
                    i = futures[future]
                    try:
                        future.result()
                        raw[i] = _load_dataset(cache_paths[i], "sampling cache")
                    except Exception as exc:
                        failures[i] = exc

        if failures:
            details = "; ".join(
                f"molecule {i}: {exc}" for i, exc in sorted(failures.items())
            )
            raise RuntimeError(
                f"Sampling failed for {len(failures)} ligand(s): {details}"
            )

        results = []
        for i in range(len(mols)):
            dataset = raw[i]
            if process_dataset is not None:
                dataset = process_dataset(i, dataset)
            _atomic_save(dataset, canonical_paths[i])
            results.append(dataset)
        return results
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
