"""Unit tests for the sampling coordinator."""

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import datasets
import pytest
from openff.toolkit import Molecule

from presto import sampling_coordinator
from presto.sampling_coordinator import sample_ligands, sampling_devices
from presto.settings import MMMDSamplingSettings, PreComputedDatasetSettings


def _precomputed_settings(n_datasets: int) -> PreComputedDatasetSettings:
    return PreComputedDatasetSettings(
        dataset_paths=[Path(f"dataset-{i}") for i in range(n_datasets)]
    )


def _generated_inputs(tmp_path: Path, n_mols: int = 2) -> dict:
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


def _successful_worker(molecule_json, molecule_index, *args) -> datasets.Dataset:
    return datasets.Dataset.from_dict({"molecule": [molecule_index]})


class _InlineExecutor:
    """Run submitted work inline while exposing the executor context/Future API."""

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


def test_sampling_devices_round_robin():
    """Workers share the visible CUDA devices round-robin, or all run on the CPU."""
    assert sampling_devices("cpu", 3) == ["cpu"] * 3
    with patch("torch.cuda.device_count", return_value=2):
        assert sampling_devices("cuda", 3) == ["cuda:0", "cuda:1", "cuda:0"]
    with patch("torch.cuda.device_count", return_value=0):
        assert sampling_devices("cuda", 2) == ["cuda:0", "cuda:0"]


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
    monkeypatch.setitem(
        sampling_coordinator._SAMPLING_FNS_REGISTRY,
        PreComputedDatasetSettings,
        MagicMock(return_value=loaded),
    )
    process_dataset = MagicMock(side_effect=processed)

    with patch("presto.sampling_coordinator.ForceField"):
        result = sample_ligands(
            mols=[MagicMock(), MagicMock()],
            offxml_path=Path("force-field.offxml"),
            device_type="cpu",
            sampling_settings=_precomputed_settings(len(loaded)),
            output_paths={},
            canonical_paths=[Path("canonical-0"), Path("canonical-1")],
            n_processes=1,
            process_dataset=process_dataset,
        )

    assert result == processed
    assert [call.args for call in process_dataset.call_args_list] == [
        (0, loaded[0]),
        (1, loaded[1]),
    ]


def test_returns_precomputed_datasets_without_callback(monkeypatch):
    """Precomputed test data remains unchanged when no callback is supplied."""
    loaded = [datasets.Dataset.from_dict({"value": [10]})]
    monkeypatch.setitem(
        sampling_coordinator._SAMPLING_FNS_REGISTRY,
        PreComputedDatasetSettings,
        MagicMock(return_value=loaded),
    )

    with patch("presto.sampling_coordinator.ForceField"):
        result = sample_ligands(
            mols=[MagicMock()],
            offxml_path=Path("force-field.offxml"),
            device_type="cpu",
            sampling_settings=_precomputed_settings(len(loaded)),
            output_paths={},
            canonical_paths=[Path("canonical-0")],
            n_processes=1,
        )

    assert result == loaded


@pytest.mark.parametrize("n_processes", [1, 2])
def test_samples_processes_and_saves_every_molecule(tmp_path, monkeypatch, n_processes):
    """Generated sampling runs, processes, and saves every molecule in order."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = n_processes
    inputs["process_dataset"] = lambda mol_idx, dataset: dataset.add_column(
        "processed", [mol_idx]
    )
    worker = MagicMock(side_effect=_successful_worker)
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)
    if n_processes > 1:
        monkeypatch.setattr(
            sampling_coordinator, "ProcessPoolExecutor", _InlineExecutor
        )

    result = sample_ligands(**inputs)

    assert [dataset["molecule"][0] for dataset in result] == [0, 1]
    assert [dataset["processed"][0] for dataset in result] == [0, 1]
    assert [call.args[1] for call in worker.call_args_list] == [0, 1]
    committed = [datasets.load_from_disk(path) for path in inputs["canonical_paths"]]
    assert [dataset["processed"][0] for dataset in committed] == [0, 1]


def test_worker_failures_are_aggregated_after_all_ligands(tmp_path, monkeypatch):
    """Ordinary failures remain per-ligand errors and do not stop sampling early."""
    inputs = _generated_inputs(tmp_path)
    worker = MagicMock(
        side_effect=[RuntimeError("first failed"), ValueError("second failed")]
    )
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)

    with pytest.raises(RuntimeError) as exc_info:
        sample_ligands(**inputs)

    assert "molecule 0: first failed" in str(exc_info.value)
    assert "molecule 1: second failed" in str(exc_info.value)
    assert worker.call_count == 2
    assert not any(path.exists() for path in inputs["canonical_paths"])


@pytest.mark.parametrize("process_control_exception", [KeyboardInterrupt, SystemExit])
def test_process_control_exception_escapes_immediately(
    tmp_path, monkeypatch, process_control_exception
):
    """Process-control exceptions are not aggregated as ligand failures."""
    inputs = _generated_inputs(tmp_path)
    worker = MagicMock(side_effect=process_control_exception("stop sampling"))
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)

    with pytest.raises(process_control_exception, match="stop sampling"):
        sample_ligands(**inputs)

    assert worker.call_count == 1
