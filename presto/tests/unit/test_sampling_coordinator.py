"""Unit tests for the sampling coordinator."""

import multiprocessing
import os
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import datasets
import numpy as np
import pytest
import torch
from loguru import logger as loguru_logger
from openff.toolkit import ForceField, Molecule

from presto import sample, sampling_coordinator
from presto._exceptions import MissingOptionalDependencyError
from presto.data_utils import (
    create_dataset_with_uniform_weights,
    create_weighted_dataset,
)
from presto.sampling_coordinator import sample_ligands, sampling_devices
from presto.settings import (
    AnySolvSMDCorrectionSettings,
    GBn2SASACorrectionSettings,
    MLPSettings,
    MMMDSamplingSettings,
    PreComputedDatasetSettings,
)
from presto.tests.unit._pooled_sampler import PooledSamplingSettings


def _weighted_dataset(energy: float) -> datasets.Dataset:
    """A one-conformer weighted dataset for methane."""
    return create_dataset_with_uniform_weights(
        smiles=Molecule.from_smiles("C").to_smiles(mapped=True),
        coords=torch.zeros(1, 5, 3, dtype=torch.float64),
        energy=torch.tensor([energy], dtype=torch.float64),
        forces=torch.zeros(1, 5, 3, dtype=torch.float64),
    )


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
        "dataset_output_paths": [
            tmp_path / f"energy_and_force_data_mol{i}" for i in range(n_mols)
        ],
        "n_processes": 1,
    }


def _successful_worker(molecule_json, molecule_index, *args) -> tuple:
    return datasets.Dataset.from_dict({"molecule": [molecule_index]}), []


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


def _use_inline_pool(monkeypatch) -> None:
    """Run the parallel path in this process, without spawning any workers."""
    monkeypatch.setattr(sampling_coordinator, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(sampling_coordinator, "_warm_ml_potential", MagicMock())


def _register_fake_sampler(monkeypatch, sample_fn) -> None:
    """Point the registry at ``sample_fn`` and stub out force-field loading."""
    monkeypatch.setitem(
        sampling_coordinator._SAMPLING_FNS_REGISTRY, MMMDSamplingSettings, sample_fn
    )
    monkeypatch.setattr(sampling_coordinator, "ForceField", MagicMock())


@pytest.fixture
def isolated_worker_globals(monkeypatch):
    """Undo the process-wide state ``_init_worker`` sets when run in the test process.

    Returns the mock standing in for the coordinator's logger.
    """
    # _init_worker drops every log sink in its process, so keep this session's own.
    mock_logger = MagicMock()
    monkeypatch.setattr(sampling_coordinator, "logger", mock_logger)
    # It also replaces sample's progress bar and claims a device, both permanently.
    # Re-setting each to its current value makes monkeypatch restore it at teardown.
    monkeypatch.setattr(sample, "track", sample.track)
    monkeypatch.setattr(sampling_coordinator, "_WORKER_DEVICE", "cpu")
    return mock_logger


def test_sampling_devices_round_robin(monkeypatch):
    """Workers share the visible CUDA devices round-robin, or all run on the CPU."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert sampling_devices("cpu", 3) == ["cpu"] * 3
    with patch("torch.cuda.device_count", return_value=2):
        assert sampling_devices("cuda", 3) == ["0", "1", "0"]
    # device_type is only "cuda" once CUDA has been shown available, so a count of
    # 0 means torch under-reported: fall back to one device rather than to the CPU.
    with patch("torch.cuda.device_count", return_value=0):
        assert sampling_devices("cuda", 2) == ["0", "0"]


def test_sampling_devices_names_the_parents_gpus(monkeypatch):
    """Workers are handed the parent's device ids, not indices into its visible list."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    with patch("torch.cuda.device_count", return_value=2):
        assert sampling_devices("cuda", 3) == ["4", "5", "4"]
    # CUDA truncates the visible list at the first entry it rejects, so a worker is
    # never handed an id past the count torch reports.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,99")
    with patch("torch.cuda.device_count", return_value=1):
        assert sampling_devices("cuda", 2) == ["4", "4"]


def test_processes_each_precomputed_dataset(monkeypatch):
    """Precomputed training data is passed through the processing callback."""
    loaded = [_weighted_dataset(10.0), _weighted_dataset(20.0)]
    processed = [_weighted_dataset(11.0), _weighted_dataset(21.0)]
    monkeypatch.setitem(
        sampling_coordinator._SAMPLING_FNS_REGISTRY,
        PreComputedDatasetSettings,
        MagicMock(return_value=loaded),
    )
    process_dataset = MagicMock(side_effect=processed)

    with patch("presto.sampling_coordinator.ForceField"):
        result = sample_ligands(
            mols=[Molecule.from_smiles("C"), Molecule.from_smiles("C")],
            offxml_path=Path("force-field.offxml"),
            device_type="cpu",
            sampling_settings=_precomputed_settings(len(loaded)),
            output_paths={},
            dataset_output_paths=[Path("dataset-out-0"), Path("dataset-out-1")],
            n_processes=1,
            process_dataset=process_dataset,
        )

    assert result == processed
    assert [call.args[0] for call in process_dataset.call_args_list] == [0, 1]
    assert [
        call.args[1]["energy"][0].item() for call in process_dataset.call_args_list
    ] == [10.0, 20.0]


def test_returns_precomputed_datasets_without_callback(monkeypatch):
    """Precomputed test data remains unchanged when no callback is supplied."""
    loaded = [_weighted_dataset(10.0)]
    monkeypatch.setitem(
        sampling_coordinator._SAMPLING_FNS_REGISTRY,
        PreComputedDatasetSettings,
        MagicMock(return_value=loaded),
    )

    with patch("presto.sampling_coordinator.ForceField"):
        result = sample_ligands(
            mols=[Molecule.from_smiles("C")],
            offxml_path=Path("force-field.offxml"),
            device_type="cpu",
            sampling_settings=_precomputed_settings(len(loaded)),
            output_paths={},
            dataset_output_paths=[Path("dataset-out-0")],
            n_processes=1,
        )

    assert len(result) == 1
    assert result[0]["energy"][0].item() == 10.0
    assert result[0]["mm_energy_correction"][0].item() == 0.0


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
    warm = MagicMock()
    monkeypatch.setattr(sampling_coordinator, "_warm_ml_potential", warm)
    if n_processes > 1:
        monkeypatch.setattr(
            sampling_coordinator, "ProcessPoolExecutor", _InlineExecutor
        )

    result = sample_ligands(**inputs)

    # The weight cache is warmed once in the parent, never on the serial path.
    assert warm.call_count == (1 if n_processes > 1 else 0)

    assert [dataset["molecule"][0] for dataset in result] == [0, 1]
    assert [dataset["processed"][0] for dataset in result] == [0, 1]
    assert [call.args[1] for call in worker.call_args_list] == [0, 1]
    # A lone worker samples on the parent's device; a pooled one uses what it claimed.
    expected_device = inputs["device_type"] if n_processes == 1 else None
    assert [call.args[3] for call in worker.call_args_list] == [expected_device] * 2
    committed = [
        datasets.load_from_disk(path) for path in inputs["dataset_output_paths"]
    ]
    assert [dataset["processed"][0] for dataset in committed] == [0, 1]


@pytest.mark.parametrize("n_processes", [1, 2])
def test_worker_failures_are_aggregated_after_all_ligands(
    tmp_path, monkeypatch, n_processes
):
    """Ordinary failures remain per-ligand errors and do not stop sampling early."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = n_processes
    worker = MagicMock(
        side_effect=[RuntimeError("first failed"), ValueError("second failed")]
    )
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)
    if n_processes > 1:
        _use_inline_pool(monkeypatch)

    with pytest.raises(RuntimeError) as exc_info:
        sample_ligands(**inputs)

    assert "molecule 0: first failed" in str(exc_info.value)
    assert "molecule 1: second failed" in str(exc_info.value)
    assert worker.call_count == 2
    assert not any(path.exists() for path in inputs["dataset_output_paths"])


@pytest.mark.parametrize("process_control_exception", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("n_processes", [1, 2])
def test_process_control_exception_escapes_immediately(
    tmp_path, monkeypatch, n_processes, process_control_exception
):
    """Process-control exceptions are not aggregated as ligand failures."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = n_processes
    worker = MagicMock(side_effect=process_control_exception("stop sampling"))
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)
    if n_processes > 1:
        _use_inline_pool(monkeypatch)

    with pytest.raises(process_control_exception, match="stop sampling"):
        sample_ligands(**inputs)

    if n_processes == 1:
        # Serially, nothing queued after the first failure runs at all.
        assert worker.call_count == 1


def test_parallel_sampling_rejects_unpicklable_settings(tmp_path):
    """Runtime objects which cannot cross to a spawned worker fail up front.

    In practice these are ASE calculators, which can only be injected from Python
    and so never round-trip through a spawned worker's pickled settings.
    """
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = 2
    inputs["sampling_settings"].mlp_settings.ml_system_kwargs = {
        "calculator": threading.Lock()
    }

    with pytest.raises(RuntimeError, match="n_sampling_processes: 1"):
        sample_ligands(**inputs)


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        # A CUDA worker hides every other GPU, so its own is always index 0.
        (["0", "1"], ["cuda:0", "cuda:0"]),
        (["0", "0"], ["cuda:0", "cuda:0"]),
        (["cpu", "cpu"], ["cpu", "cpu"]),
    ],
)
def test_each_worker_claims_one_device(
    isolated_worker_globals, monkeypatch, devices, expected
):
    """Workers claim a device each at start-up, oversubscribed GPUs included."""
    # _init_worker narrows this process's own CUDA_VISIBLE_DEVICES; setenv restores it.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    queue = multiprocessing.get_context("spawn").Queue()
    for device in devices:
        queue.put(device)

    claimed = []
    for _ in devices:
        sampling_coordinator._init_worker(queue)
        claimed.append(sampling_coordinator._WORKER_DEVICE)

    assert claimed == expected


def test_warms_the_weight_cache_before_spawning_workers(tmp_path, monkeypatch):
    """One system is built in the parent so workers never race a first download."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = 2
    monkeypatch.setattr(sampling_coordinator, "_WARMED_POTENTIALS", set())
    monkeypatch.setattr(
        sampling_coordinator,
        "_sample_worker",
        MagicMock(side_effect=_successful_worker),
    )
    monkeypatch.setattr(sampling_coordinator, "ProcessPoolExecutor", _InlineExecutor)
    get_system = MagicMock()
    monkeypatch.setattr(sampling_coordinator.mlp, "get_ml_omm_system", get_system)

    sample_ligands(**inputs)
    sample_ligands(**inputs)

    # Warmed once per process, not once per sampling call.
    assert get_system.call_count == 1
    assert get_system.call_args.args[1] is inputs["sampling_settings"].mlp_settings


def test_worker_logs_are_replayed_by_the_parent(tmp_path, monkeypatch):
    """Warnings raised inside a worker reach the parent, tagged by molecule."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = 2

    def worker(molecule_json, molecule_index, *args):
        return (
            datasets.Dataset.from_dict({"molecule": [molecule_index]}),
            [("WARNING", "No rotatable bonds found")],
        )

    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)
    monkeypatch.setattr(sampling_coordinator, "_warm_ml_potential", MagicMock())
    monkeypatch.setattr(sampling_coordinator, "ProcessPoolExecutor", _InlineExecutor)
    replayed = []
    monkeypatch.setattr(
        sampling_coordinator,
        "logger",
        MagicMock(log=lambda level, message: replayed.append((level, message))),
    )

    sample_ligands(**inputs)

    assert replayed == [
        ("WARNING", "[molecule 0] No rotatable bonds found"),
        ("WARNING", "[molecule 1] No rotatable bonds found"),
    ]


def test_worker_claims_one_visible_gpu(isolated_worker_globals, monkeypatch):
    """A CUDA worker sees only the one device it claimed."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    queue = multiprocessing.get_context("spawn").Queue()
    queue.put("5")

    sampling_coordinator._init_worker(queue)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    # The claimed device is now the only one visible, so it is index 0 everywhere.
    assert sampling_coordinator._WORKER_DEVICE == "cuda:0"
    # Workers share one terminal, so their records go to the parent to be replayed.
    isolated_worker_globals.add.assert_called_once_with(
        sampling_coordinator._capture_log
    )


def test_worker_indexes_its_side_outputs_by_molecule(tmp_path, monkeypatch):
    """A worker's side outputs are named for the molecule's index in the workflow."""
    seen = {}

    def fake_sample_fn(**kwargs):
        seen["offset"] = sample._MOL_INDEX_OFFSET
        seen["device"] = kwargs["device"]
        seen["smiles"] = kwargs["mols"][0].to_smiles()
        return [_weighted_dataset(1.0)]

    _register_fake_sampler(monkeypatch, fake_sample_fn)
    monkeypatch.setattr(sample, "_MOL_INDEX_OFFSET", 0)
    monkeypatch.setattr(sampling_coordinator, "_WORKER_DEVICE", "cpu")
    # A worker samples one molecule after another, so the last one's records must go.
    monkeypatch.setattr(
        sampling_coordinator, "_WORKER_LOGS", [("INFO", "from the previous molecule")]
    )

    dataset, logs = sampling_coordinator._sample_worker(
        Molecule.from_smiles("C").to_json(),
        3,
        str(tmp_path / "force-field.offxml"),
        None,
        MMMDSamplingSettings(),
        {},
    )

    assert seen["offset"] == 3
    # No device was passed, so the worker samples on the one it claimed at start-up.
    assert seen["device"] == torch.device("cpu")
    assert seen["smiles"] == Molecule.from_smiles("C").to_smiles()
    assert dataset["energy"][0].item() == 1.0
    assert logs == []
    # The offset is process-wide, so it must not leak into the worker's next molecule.
    assert sample._MOL_INDEX_OFFSET == 0


def test_worker_clears_the_molecule_index_after_a_failure(tmp_path, monkeypatch):
    """A failed molecule leaves no index behind to mislabel the worker's next one."""

    def fake_sample_fn(**kwargs):
        raise RuntimeError("sampling failed")

    _register_fake_sampler(monkeypatch, fake_sample_fn)
    monkeypatch.setattr(sample, "_MOL_INDEX_OFFSET", 0)

    with pytest.raises(RuntimeError, match="sampling failed"):
        sampling_coordinator._sample_worker(
            Molecule.from_smiles("C").to_json(),
            3,
            str(tmp_path / "force-field.offxml"),
            "cpu",
            MMMDSamplingSettings(),
            {},
        )

    assert sample._MOL_INDEX_OFFSET == 0


def test_worker_log_records_are_buffered_for_the_parent(monkeypatch):
    """A record logged inside a worker is buffered with its level, not written out."""
    monkeypatch.setattr(sampling_coordinator, "_WORKER_LOGS", [])

    sink_id = loguru_logger.add(sampling_coordinator._capture_log)
    try:
        loguru_logger.warning("No rotatable bonds found")
    finally:
        loguru_logger.remove(sink_id)

    assert sampling_coordinator._WORKER_LOGS == [
        ("WARNING", "No rotatable bonds found")
    ]


def test_results_are_keyed_by_molecule_not_completion_order(tmp_path, monkeypatch):
    """Pooled ligands finish in any order, so results and logs are keyed by index."""
    inputs = _generated_inputs(tmp_path, n_mols=3)
    inputs["n_processes"] = 3

    def worker(molecule_json, molecule_index, *args):
        return (
            datasets.Dataset.from_dict({"molecule": [molecule_index]}),
            [("WARNING", f"ligand {molecule_index} warned")],
        )

    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)
    _use_inline_pool(monkeypatch)
    # A real pool yields whichever ligand finished first; the reverse is the worst case.
    monkeypatch.setattr(
        sampling_coordinator, "as_completed", lambda futures: reversed(list(futures))
    )
    replayed = []
    monkeypatch.setattr(
        sampling_coordinator,
        "logger",
        MagicMock(log=lambda level, message: replayed.append(message)),
    )

    result = sample_ligands(**inputs)

    assert [dataset["molecule"][0] for dataset in result] == [0, 1, 2]
    assert replayed == [
        "[molecule 0] ligand 0 warned",
        "[molecule 1] ligand 1 warned",
        "[molecule 2] ligand 2 warned",
    ]


def test_a_single_ligand_never_starts_a_pool(tmp_path, monkeypatch):
    """One ligand is sampled in the parent however many processes were asked for."""
    inputs = _generated_inputs(tmp_path, n_mols=1)
    inputs["n_processes"] = 4
    # A pool would reject these settings up front, so sampling at all proves it is serial.
    inputs["sampling_settings"].mlp_settings.ml_system_kwargs = {
        "calculator": threading.Lock()
    }
    worker = MagicMock(side_effect=_successful_worker)
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)
    warm = MagicMock()
    monkeypatch.setattr(sampling_coordinator, "_warm_ml_potential", warm)

    result = sample_ligands(**inputs)

    assert len(result) == 1
    assert warm.call_count == 0
    # The parent's own device, not one claimed from the queue.
    assert worker.call_args.args[3] == "cpu"


def test_the_weight_cache_is_warmed_once_per_potential(tmp_path, monkeypatch):
    """A different reference potential is warmed again rather than assumed cached."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = 2
    monkeypatch.setattr(sampling_coordinator, "_WARMED_POTENTIALS", set())
    monkeypatch.setattr(
        sampling_coordinator,
        "_sample_worker",
        MagicMock(side_effect=_successful_worker),
    )
    monkeypatch.setattr(sampling_coordinator, "ProcessPoolExecutor", _InlineExecutor)
    get_system = MagicMock()
    monkeypatch.setattr(sampling_coordinator.mlp, "get_ml_omm_system", get_system)

    sample_ligands(**inputs)
    inputs["sampling_settings"] = MMMDSamplingSettings(
        mlp_settings=MLPSettings(ml_potential="egret-1")
    )
    sample_ligands(**inputs)

    assert get_system.call_count == 2


@pytest.mark.slow
def test_a_real_pool_samples_every_ligand_in_a_worker(tmp_path, monkeypatch):
    """A spawned pool round-trips every argument and dataset, keeping molecule order."""
    mols = [Molecule.from_smiles(smiles) for smiles in ["C", "CC", "CCC", "CCCC"]]
    dataset_output_paths = [tmp_path / f"mol{mol_idx}" for mol_idx in range(len(mols))]
    # Warming downloads reference weights, which this protocol never uses.
    monkeypatch.setattr(sampling_coordinator, "_warm_ml_potential", MagicMock())

    result = sample_ligands(
        mols=mols,
        offxml_path=Path("openff-2.0.0.offxml"),
        device_type="cpu",
        sampling_settings=PooledSamplingSettings(),
        output_paths={},
        dataset_output_paths=dataset_output_paths,
        n_processes=2,
    )

    # Datasets survive the trip home, in molecule order rather than completion order.
    assert [dataset["smiles"][0] for dataset in result] == [
        mol.to_smiles() for mol in mols
    ]
    assert [dataset["mol_index"][0] for dataset in result] == list(range(len(mols)))
    # The molecule and force field really were rebuilt from the pickled arguments.
    assert [dataset["n_ff_parameters"][0] for dataset in result] == [
        len(ForceField("openff-2.0.0.offxml").get_parameter_handler("Bonds").parameters)
    ] * len(mols)
    # Sampling happened in the pool, on the device each worker claimed at start-up.
    worker_pids = {dataset["pid"][0] for dataset in result}
    assert os.getpid() not in worker_pids
    assert len(worker_pids) <= 2
    assert [dataset["device"][0] for dataset in result] == ["cpu"] * len(mols)
    assert all(path.exists() for path in dataset_output_paths)


def _ethanol_dataset() -> tuple[Molecule, datasets.Dataset]:
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=1)
    coords = torch.tensor(mol.conformers[0].m_as("angstrom")).reshape(1, -1, 3)
    return mol, create_dataset_with_uniform_weights(
        smiles=mol.to_smiles(mapped=True),
        coords=coords,
        energy=torch.zeros(1, dtype=torch.float64),
        forces=torch.zeros_like(coords),
    )


class TestCorrections:
    """Tests for applying fit and sampling corrections while sampling."""

    def test_corrections_and_charges_reach_every_worker(self, tmp_path, monkeypatch):
        """Workers get the fit corrections, charges and resolved sampling settings."""
        inputs = _generated_inputs(tmp_path)
        gb = GBn2SASACorrectionSettings()
        charges = [np.full(5, 0.1), np.full(5, 0.2)]
        worker = MagicMock(side_effect=_successful_worker)
        monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)

        sample_ligands(**inputs, mm_corrections=[gb], mm_charges=charges)

        for mol_idx, call in enumerate(worker.call_args_list):
            settings = call.args[4]
            assert settings.mm_sampling_corrections == [gb]
            assert settings.ml_sampling_corrections == []
            assert call.args[6] == ()
            assert call.args[7] == (gb,)
            assert np.array_equal(call.args[8], charges[mol_idx])
        # The caller's settings are not modified
        assert inputs["sampling_settings"].mm_sampling_corrections == "from_fit"

    def test_mm_charges_must_match_the_molecules(self, tmp_path):
        """One array of charges is needed per molecule."""
        with pytest.raises(ValueError, match="one array of charges per molecule"):
            sample_ligands(**_generated_inputs(tmp_path), mm_charges=[np.zeros(5)])

    def test_unavailable_corrections_fail_before_sampling(self, tmp_path, monkeypatch):
        """A missing optional dependency is reported before any ligand is sampled."""
        monkeypatch.setitem(sys.modules, "anysolv", None)
        worker = MagicMock(side_effect=_successful_worker)
        monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)

        with pytest.raises(MissingOptionalDependencyError):
            sample_ligands(
                **_generated_inputs(tmp_path),
                reference_corrections=[
                    AnySolvSMDCorrectionSettings(checkpoint="model_compact")
                ],
            )
        worker.assert_not_called()

    def test_worker_applies_fit_corrections(self, tmp_path, monkeypatch):
        """The worker stores the fit corrections for the sampled coordinates."""
        mol, dataset = _ethanol_dataset()
        _register_fake_sampler(monkeypatch, lambda **kwargs: [dataset])
        charges = np.linspace(-0.4, 0.4, mol.n_atoms)

        result, _ = sampling_coordinator._sample_worker(
            mol.to_json(),
            0,
            str(tmp_path / "force-field.offxml"),
            "cpu",
            MMMDSamplingSettings(),
            {},
            (),
            (GBn2SASACorrectionSettings(),),
            charges,
        )

        assert result[0]["mm_energy_correction"][0].item() < 0.0
        assert torch.all(result[0]["reference_energy_correction"] == 0.0)

    def test_worker_drops_non_finite_conformations(self, tmp_path, monkeypatch):
        """Non-finite conformations are removed, or fail the ligand if too many."""
        mol, dataset = _ethanol_dataset()
        entry = dict(dataset[0])
        entry["energy"] = torch.tensor([float("nan")])
        bad = create_weighted_dataset([entry])
        _register_fake_sampler(monkeypatch, lambda **kwargs: [bad])

        with pytest.raises(ValueError, match="non-finite"):
            sampling_coordinator._sample_worker(
                mol.to_json(),
                0,
                str(tmp_path / "force-field.offxml"),
                "cpu",
                MMMDSamplingSettings(),
                {},
            )

    @pytest.mark.parametrize("mode", ["keep", "recompute", "zero"])
    def test_precomputed_reference_modes(self, monkeypatch, fake_anysolv, mode):
        """Stored reference corrections follow the mode; MM ones are always rewritten."""
        mol, dataset = _ethanol_dataset()
        entry = dict(dataset[0])
        entry["reference_energy_correction"] = torch.tensor([123.0])
        entry["mm_energy_correction"] = torch.tensor([456.0])
        stored = create_weighted_dataset([entry])
        monkeypatch.setitem(
            sampling_coordinator._SAMPLING_FNS_REGISTRY,
            PreComputedDatasetSettings,
            MagicMock(return_value=[stored]),
        )
        settings = PreComputedDatasetSettings(
            dataset_paths=[Path("dataset")], reference_corrections_mode=mode
        )

        with patch("presto.sampling_coordinator.ForceField"):
            [result] = sample_ligands(
                mols=[mol],
                offxml_path=Path("force-field.offxml"),
                device_type="cpu",
                sampling_settings=settings,
                output_paths={},
                dataset_output_paths=[Path("unused")],
                n_processes=1,
                reference_corrections=[
                    AnySolvSMDCorrectionSettings(checkpoint="model_compact")
                ],
            )

        reference = result[0]["reference_energy_correction"][0].item()
        if mode == "keep":
            assert reference == 123.0
        elif mode == "zero":
            assert reference == 0.0
        else:
            coords = entry["coords"].reshape(-1, 3).numpy()
            assert reference == pytest.approx(
                -0.1 * coords[:, 0].sum() * 23.060548, rel=1e-5
            )
        # No MM corrections are configured, so the stored ones are removed
        assert result[0]["mm_energy_correction"][0].item() == 0.0
