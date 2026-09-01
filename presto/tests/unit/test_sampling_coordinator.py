"""Unit tests for the sampling coordinator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import datasets

from presto import sampling_coordinator
from presto.sampling_coordinator import sample_ligands
from presto.settings import PreComputedDatasetSettings


def _precomputed_settings(n_datasets: int) -> PreComputedDatasetSettings:
    return PreComputedDatasetSettings(
        dataset_paths=[Path(f"dataset-{i}") for i in range(n_datasets)]
    )


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
