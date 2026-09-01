"""Unit tests for the sampling coordinator."""

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import datasets
import pytest
from openff.toolkit import Molecule

from presto import sampling_coordinator
from presto.outputs import OutputType, get_mol_path
from presto.sampling_coordinator import sample_ligands
from presto.settings import (
    MMMDSamplingSettings,
    PreComputedDatasetSettings,
)


def _precomputed_settings(n_datasets: int) -> PreComputedDatasetSettings:
    return PreComputedDatasetSettings(
        dataset_paths=[Path(f"dataset-{i}") for i in range(n_datasets)]
    )


def _generated_inputs(tmp_path: Path, n_mols: int = 2):
    settings = MMMDSamplingSettings()
    return {
        "mols": [Molecule.from_smiles("C") for _ in range(n_mols)],
        "offxml_path": tmp_path / "force-field.offxml",
        "device_type": "cpu",
        "sampling_settings": settings,
        "output_paths": {
            output_type: tmp_path / output_type.value
            for output_type in settings.output_types
        },
        "canonical_paths": [
            tmp_path / f"energy_and_force_data_mol{i}" for i in range(n_mols)
        ],
        "n_processes": 1,
    }


def _successful_worker(*args):
    molecule_index = args[1]
    output_paths = args[5]
    cache_path = args[6]
    for base_path in output_paths.values():
        side_path = get_mol_path(base_path, molecule_index)
        if side_path.suffix:
            side_path.parent.mkdir(parents=True, exist_ok=True)
            side_path.write_text("diagnostic output")
        else:
            side_path.mkdir(parents=True, exist_ok=True)
    sampling_coordinator._atomic_save(
        datasets.Dataset.from_dict({"molecule": [molecule_index]}), cache_path
    )
    return molecule_index


class _InlineExecutor:
    """Run submitted work inline while exposing the executor Future API."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_processes_each_precomputed_dataset(monkeypatch):
    """Precomputed training data is passed through the processing callback."""
    loaded = [
        datasets.Dataset.from_dict({"value": [10]}),
        datasets.Dataset.from_dict({"value": [20]}),
    ]
    processed = [
        datasets.Dataset.from_dict({"value": [11]}),
        datasets.Dataset.from_dict({"value": [21]}),
    ]
    sample_fn = MagicMock(return_value=loaded)
    settings = _precomputed_settings(len(loaded))
    monkeypatch.setitem(
        sampling_coordinator._SAMPLING_FNS_REGISTRY,
        PreComputedDatasetSettings,
        sample_fn,
    )
    process_dataset = MagicMock(side_effect=processed)

    with patch("presto.sampling_coordinator.ForceField"):
        result = sample_ligands(
            mols=[MagicMock(), MagicMock()],
            offxml_path=Path("force-field.offxml"),
            device_type="cpu",
            sampling_settings=settings,
            output_paths={},
            canonical_paths=[Path("canonical-0"), Path("canonical-1")],
            n_processes=1,
            process_dataset=process_dataset,
        )

    assert result == processed
    assert process_dataset.call_args_list[0].args == (0, loaded[0])
    assert process_dataset.call_args_list[1].args == (1, loaded[1])


def test_returns_precomputed_datasets_without_callback(monkeypatch):
    """Precomputed test data remains unchanged when no callback is supplied."""
    loaded = [datasets.Dataset.from_dict({"value": [10]})]
    sample_fn = MagicMock(return_value=loaded)
    settings = _precomputed_settings(len(loaded))
    monkeypatch.setitem(
        sampling_coordinator._SAMPLING_FNS_REGISTRY,
        PreComputedDatasetSettings,
        sample_fn,
    )

    with patch("presto.sampling_coordinator.ForceField"):
        result = sample_ligands(
            mols=[MagicMock()],
            offxml_path=Path("force-field.offxml"),
            device_type="cpu",
            sampling_settings=settings,
            output_paths={},
            canonical_paths=[Path("canonical-0")],
            n_processes=1,
        )

    assert result is loaded


@pytest.mark.parametrize("n_processes", [1, 2])
def test_clean_directory_samples_every_molecule(tmp_path, monkeypatch, n_processes):
    """Clean generated-sampling runs sample and atomically commit every molecule."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = n_processes
    worker = MagicMock(side_effect=_successful_worker)
    process_dataset = MagicMock(
        side_effect=lambda molecule_index, dataset: dataset.add_column(
            "processed", [molecule_index]
        )
    )
    inputs["process_dataset"] = process_dataset
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)
    if n_processes > 1:
        monkeypatch.setattr(
            sampling_coordinator, "ProcessPoolExecutor", _InlineExecutor
        )

    result = sample_ligands(**inputs)

    assert [dataset["molecule"][0] for dataset in result] == [0, 1]
    assert [dataset["processed"][0] for dataset in result] == [0, 1]
    assert [call.args[1] for call in worker.call_args_list] == [0, 1]
    assert [call.args[0] for call in process_dataset.call_args_list] == [0, 1]
    assert all(path.exists() for path in inputs["canonical_paths"])
    committed = [datasets.load_from_disk(path) for path in inputs["canonical_paths"]]
    assert [dataset["processed"][0] for dataset in committed] == [0, 1]
    assert not (tmp_path / ".sampling_cache").exists()


def test_worker_failure_keeps_diagnostics_but_removes_cache(tmp_path, monkeypatch):
    """Orderly failure retains side outputs and discards transient raw datasets."""
    inputs = _generated_inputs(tmp_path, n_mols=1)
    trajectory = get_mol_path(inputs["output_paths"][OutputType.PDB_TRAJECTORY], 0)

    def failing_worker(*args):
        trajectory.write_text("partial trajectory")
        sampling_coordinator._atomic_save(
            datasets.Dataset.from_dict({"molecule": [0]}), args[6]
        )
        raise RuntimeError("simulation failed")

    worker = MagicMock(side_effect=failing_worker)
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)

    with pytest.raises(RuntimeError, match=r"molecule 0: simulation failed"):
        sample_ligands(**inputs)

    assert trajectory.exists()
    assert not (tmp_path / ".sampling_cache").exists()
    assert worker.call_count == 1


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda inputs: inputs.update(mols=[]), "At least one molecule"),
        (
            lambda inputs: inputs.update(canonical_paths=[]),
            "one canonical dataset path",
        ),
        (lambda inputs: inputs.update(n_processes=0), "n_processes"),
        (lambda inputs: inputs.update(output_paths={}), "output types configured"),
    ],
)
def test_generated_sampling_inputs_are_validated(tmp_path, change, error):
    """Malformed generated-sampling inputs fail before cache creation."""
    inputs = _generated_inputs(tmp_path, n_mols=1)
    change(inputs)

    with pytest.raises((TypeError, ValueError), match=error):
        sample_ligands(**inputs)

    assert not (tmp_path / ".sampling_cache").exists()
