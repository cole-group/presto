"""Unit tests for workflow lifecycle behavior."""

import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openmm import unit

from presto.outputs import WorkflowPathManager, WorkflowStatus
from presto.settings import (
    MLMDSamplingSettings,
    MMMDMetadynamicsSamplingSettings,
    MMMDSamplingSettings,
)
from presto.workflow import get_bespoke_force_field


@pytest.mark.parametrize(
    ("stage_name", "expected_status"),
    [
        ("test_data", WorkflowStatus.PARTIAL),
        ("training_iteration_2", WorkflowStatus.COMPLETE),
    ],
)
def test_non_clean_fit_fails_before_parameterisation(
    tmp_path, stage_name, expected_status
):
    """The workflow rejects reruns before any molecule setup or output is written."""
    stage_path = tmp_path / stage_name
    stage_path.mkdir()
    if expected_status == WorkflowStatus.COMPLETE:
        (stage_path / "bespoke_ff.offxml").write_text("force field")

    settings = MagicMock()
    settings.get_path_manager.return_value = WorkflowPathManager(
        output_dir=tmp_path, n_iterations=2
    )

    with (
        patch("presto.workflow.parameterise") as parameterise,
        pytest.raises(RuntimeError, match=rf"{expected_status.value}.*presto clean"),
    ):
        get_bespoke_force_field(settings)

    parameterise.assert_not_called()


class _WorkflowRun(typing.NamedTuple):
    """What a faked workflow run did.

    Each molecule's dataset is a string naming where it would be saved, e.g.
    ``"training_iteration_1/train_dataset_0/mol0"``, and memory concatenation joins
    two of them as ``"[a + b]"``.
    """

    sample_calls: list[dict[str, typing.Any]]
    """The keyword arguments of each call to ``sample_ligands``, in order."""
    filtered: list[tuple[str, str]]
    """The (dataset, topology) pairs passed to outlier filtering, in order."""
    training_calls: list[dict[str, typing.Any]]
    """The keyword arguments of each call to the training function, one per
    iteration."""
    scatter_calls: list[typing.Any]
    """The calls to ``write_scatter``, in order."""


def _run_workflow(settings):
    """Run the workflow with parameterisation, sampling and training faked.

    Outlier filtering returns each dataset unchanged, and nothing is written but
    the stage directories.
    """
    import torch

    sample_calls = []

    def fake_sample_ligands(**kwargs):
        """Name each molecule's dataset after where it would be saved."""
        sample_calls.append(kwargs)
        raw = [
            f"{path.parent.parent.name}/{path.parent.name}/mol{i}"
            for i, path in enumerate(kwargs["dataset_output_paths"])
        ]
        process = kwargs["process_dataset"]
        return [d if process is None else process(i, d) for i, d in enumerate(raw)]

    filtered = []

    def fake_filter(dataset, topology, **kwargs):
        filtered.append((dataset, topology))
        return dataset

    trainable = MagicMock()
    trainable.to_values.return_value = torch.zeros(3)
    trainable._param_types = []
    train_fn = MagicMock(
        side_effect=lambda **kwargs: (kwargs["trainable_parameters"], trainable)
    )

    with (
        patch(
            "presto.workflow.parameterise",
            return_value=(["mol0", "mol1"], MagicMock(), ["top0", "top1"], MagicMock()),
        ),
        patch("presto.workflow.Trainable", return_value=trainable),
        patch("presto.workflow.sample_ligands", side_effect=fake_sample_ligands),
        patch("presto.workflow.write_scatter", return_value=(0, 0, 0, 0)) as scatter,
        patch("presto.workflow.convert_to_smirnoff"),
        patch("presto.workflow.analyse_workflow"),
        patch("presto.workflow.filter_dataset_outliers", side_effect=fake_filter),
        patch(
            "presto.workflow.datasets.combine.concatenate_datasets",
            side_effect=lambda parts: f"[{' + '.join(parts)}]",
        ),
        patch.dict("presto.workflow._TRAINING_FNS_REGISTRY", {"adam": train_fn}),
    ):
        get_bespoke_force_field(settings, write_settings=False)

    return _WorkflowRun(
        sample_calls=sample_calls,
        filtered=filtered,
        training_calls=[call.kwargs for call in train_fn.call_args_list],
        scatter_calls=scatter.call_args_list,
    )


def _two_molecule_settings(tmp_path, train_protocols, test_protocols, **kwargs):
    """Settings for ethanol and propane, which ``_run_workflow`` calls mol0 and mol1."""
    from presto.settings import ParamSettings, WorkflowSettings

    return WorkflowSettings(
        output_dir=tmp_path,
        device_type="cpu",
        param_settings=ParamSettings(molecules=["CCO", "CCC"]),
        training_sampling_settings=train_protocols,
        testing_sampling_settings=test_protocols,
        **kwargs,
    )


@pytest.mark.parametrize("memory", [True, False], ids=["memory", "no-memory"])
@pytest.mark.parametrize(
    ("train_protocols", "test_protocols"),
    [
        pytest.param(
            [
                MMMDSamplingSettings(),
                MMMDMetadynamicsSamplingSettings(dataset_weight=3.0),
            ],
            [MLMDSamplingSettings(), MMMDSamplingSettings(dataset_weight=2.0)],
            id="different-protocols",
        ),
        pytest.param(
            [
                MMMDSamplingSettings(temperature=300 * unit.kelvin),
                MMMDSamplingSettings(
                    temperature=600 * unit.kelvin, n_conformers=20, dataset_weight=3.0
                ),
            ],
            [
                MLMDSamplingSettings(temperature=300 * unit.kelvin),
                MLMDSamplingSettings(dataset_weight=2.0),
            ],
            id="same-protocol-different-settings",
        ),
    ],
)
def test_datasets_sampled_in_turn_and_trained_together(
    tmp_path, train_protocols, test_protocols, memory
):
    """Datasets are sampled one after another, then trained on together.

    Each protocol is sampled into its own subdirectory with its own settings, even
    when several share a sampling protocol, and every training dataset is passed to
    a single training call with its weight.
    """
    settings = _two_molecule_settings(
        tmp_path, train_protocols, test_protocols, n_iterations=2, memory=memory
    )

    run = _run_workflow(settings)

    # Sampled one protocol at a time: test datasets, then each iteration's training
    stage_dirs = [
        call["dataset_output_paths"][0].parent.relative_to(tmp_path).as_posix()
        for call in run.sample_calls
    ]
    assert stage_dirs == [
        "test_data/test_dataset_0",
        "test_data/test_dataset_1",
        "training_iteration_1/train_dataset_0",
        "training_iteration_1/train_dataset_1",
        "training_iteration_2/train_dataset_0",
        "training_iteration_2/train_dataset_1",
    ]
    initial_ff = tmp_path / "initial_statistics" / "bespoke_ff.offxml"
    iteration_1_ff = tmp_path / "training_iteration_1" / "bespoke_ff.offxml"
    for call, protocol, offxml in zip(
        run.sample_calls,
        [*test_protocols, *train_protocols, *train_protocols],
        [initial_ff] * 4 + [iteration_1_ff] * 2,
        strict=True,
    ):
        assert call["sampling_settings"] is protocol
        assert call["offxml_path"] == offxml
        assert set(call["output_paths"]) == protocol.output_types
        dataset_dir = call["dataset_output_paths"][0].parent
        assert dataset_dir.is_dir()
        assert all(path.parent == dataset_dir for path in call["output_paths"].values())
        assert [p.name for p in call["dataset_output_paths"]] == [
            "energy_and_force_data_mol0",
            "energy_and_force_data_mol1",
        ]
    # Every dataset is processed, but test data is neither filtered nor remembered
    # (see the unchanged test datasets below)
    assert all(call["process_dataset"] is not None for call in run.sample_calls)

    # Every training dataset is filtered, per dataset and molecule
    assert run.filtered == [
        (f"training_iteration_{i}/train_dataset_{k}/mol{m}", f"top{m}")
        for i in (1, 2)
        for k in (0, 1)
        for m in (0, 1)
    ]

    # One joint training call per iteration, with every dataset and its weight
    assert len(run.training_calls) == 2
    test_datasets = [
        [f"test_data/test_dataset_{k}/mol{m}" for m in (0, 1)] for k in (0, 1)
    ]
    first, second = run.training_calls
    assert first["datasets"] == [
        [f"training_iteration_1/train_dataset_{k}/mol{m}" for m in (0, 1)]
        for k in (0, 1)
    ]
    # Memory appends each dataset to the same dataset from the previous iteration
    assert second["datasets"] == [
        [
            f"[training_iteration_1/train_dataset_{k}/mol{m} + "
            f"training_iteration_2/train_dataset_{k}/mol{m}]"
            if memory
            else f"training_iteration_2/train_dataset_{k}/mol{m}"
            for m in (0, 1)
        ]
        for k in (0, 1)
    ]
    for kwargs in (first, second):
        assert kwargs["datasets_test"] == test_datasets
        assert kwargs["dataset_weights"] == [1.0, 3.0]
        assert kwargs["dataset_weights_test"] == [1.0, 2.0]
        # None for each dataset, which the loss treats as equal weights
        assert kwargs["molecule_weights"] == [None, None]
        assert kwargs["molecule_weights_test"] == [None, None]
        assert kwargs["topologies"] == ["top0", "top1"]

    # Scatter for each test dataset and molecule, initially and after each iteration
    scatter_paths = [
        Path(call.args[4]).relative_to(tmp_path).as_posix()
        for call in run.scatter_calls
    ]
    assert scatter_paths == [
        f"{stage}/test_dataset_{k}/energies_and_forces_mol{m}.hdf5"
        for stage in (
            "initial_statistics",
            "training_iteration_1",
            "training_iteration_2",
        )
        for k in (0, 1)
        for m in (0, 1)
    ]
    assert [call.args[0] for call in run.scatter_calls[:4]] == [
        d for datasets_for_mols in test_datasets for d in datasets_for_mols
    ]


def test_molecule_weights_reach_training(tmp_path):
    """Each protocol's molecule weights are passed to training, with None where unset."""
    settings = _two_molecule_settings(
        tmp_path,
        train_protocols=[
            MMMDSamplingSettings(molecule_weights=[2.0, 1.0]),
            MMMDSamplingSettings(),
        ],
        test_protocols=[
            MLMDSamplingSettings(),
            MLMDSamplingSettings(molecule_weights=[0.0, 1.0]),
        ],
        n_iterations=1,
    )

    (training_call,) = _run_workflow(settings).training_calls

    # One list per dataset, with one weight per molecule
    assert training_call["molecule_weights"] == [[2.0, 1.0], None]
    assert training_call["molecule_weights_test"] == [None, [0.0, 1.0]]


def fake_conformer_weights(dataset):
    """A conformer weights function the workflow tests import by name."""
    raise AssertionError("apply_conformer_weights is patched, so this is never called")


_FAKE_WEIGHTS_SPEC = "presto.tests.unit.test_workflow:fake_conformer_weights"


@pytest.mark.parametrize("memory", [True, False], ids=["memory", "no-memory"])
def test_conformer_weights_applied_after_memory(tmp_path, memory):
    """Conformer weights are applied to the datasets of the protocols that set them.

    Training datasets are weighted after memory adds the earlier iterations, so the
    weights are recomputed over all of them. Testing datasets are weighted too.
    """
    # Training dataset 0 and testing dataset 1 have conformer weights
    settings = _two_molecule_settings(
        tmp_path,
        train_protocols=[
            MMMDSamplingSettings(conformer_weights_fn=_FAKE_WEIGHTS_SPEC),
            MMMDSamplingSettings(),
        ],
        test_protocols=[
            MLMDSamplingSettings(),
            MLMDSamplingSettings(conformer_weights_fn=_FAKE_WEIGHTS_SPEC),
        ],
        n_iterations=2,
        memory=memory,
    )

    def fake_apply(dataset, fn):
        """Mark a dataset as weighted, e.g. "weighted(name)"."""
        assert fn is fake_conformer_weights
        return f"weighted({dataset})"

    with patch("presto.workflow.apply_conformer_weights", side_effect=fake_apply):
        iteration_1, iteration_2 = _run_workflow(settings).training_calls

    for training_call in (iteration_1, iteration_2):
        assert training_call["datasets_test"] == [
            [f"test_data/test_dataset_0/mol{m}" for m in (0, 1)],
            [f"weighted(test_data/test_dataset_1/mol{m})" for m in (0, 1)],
        ]
    assert iteration_1["datasets"] == [
        [f"weighted(training_iteration_1/train_dataset_0/mol{m})" for m in (0, 1)],
        [f"training_iteration_1/train_dataset_1/mol{m}" for m in (0, 1)],
    ]
    # With memory, the weights are recomputed over the remembered and new data
    assert iteration_2["datasets"][0] == [
        f"weighted([weighted(training_iteration_1/train_dataset_0/mol{m}) + "
        f"training_iteration_2/train_dataset_0/mol{m}])"
        if memory
        else f"weighted(training_iteration_2/train_dataset_0/mol{m})"
        for m in (0, 1)
    ]


def test_bad_conformer_weights_fn_fails_before_any_output(tmp_path):
    """A conformer weights function that cannot be imported fails before output is written."""
    from presto.settings import ParamSettings, WorkflowSettings

    output_dir = tmp_path / "out"
    settings = WorkflowSettings(
        output_dir=output_dir,
        device_type="cpu",
        param_settings=ParamSettings(molecules=["CCO"]),
        training_sampling_settings=[
            MMMDSamplingSettings(conformer_weights_fn="not_a_real_module:weights")
        ],
    )

    with (
        patch("presto.workflow.parameterise") as parameterise,
        pytest.raises(ModuleNotFoundError, match="not_a_real_module"),
    ):
        get_bespoke_force_field(settings)

    parameterise.assert_not_called()
    assert not output_dir.exists() or not any(output_dir.iterdir())
