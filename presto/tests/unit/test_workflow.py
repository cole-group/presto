"""Unit tests for workflow lifecycle behavior."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from presto.outputs import WorkflowPathManager, WorkflowStatus
from presto.settings import MMMDSamplingSettings, TrainingSettings
from presto.workflow import get_bespoke_force_field


class _WritingForceField:
    """Minimal force-field writer standing in for a parameterised force field."""

    def __init__(self, contents: str):
        self.contents = contents

    def to_file(self, path: str) -> None:
        Path(path).write_text(self.contents)


@pytest.mark.parametrize(
    ("stage_name", "expected_status"),
    [
        ("test_data", WorkflowStatus.PARTIAL),
        ("training_iteration_2", WorkflowStatus.COMPLETE),
    ],
)
def test_non_clean_fit_fails_before_full_path_manager_or_parameterisation(
    tmp_path, stage_name, expected_status
):
    """The workflow rejects reruns before molecule-dependent setup or mutation."""
    stage_path = tmp_path / stage_name
    stage_path.mkdir()
    if expected_status == WorkflowStatus.COMPLETE:
        (stage_path / "bespoke_ff.offxml").write_text("force field")

    settings = MagicMock()
    settings.output_dir = tmp_path
    settings.n_iterations = 2

    with (
        patch("presto.workflow.parameterise") as parameterise,
        pytest.raises(RuntimeError, match=rf"{expected_status.value}.*presto clean"),
    ):
        get_bespoke_force_field(settings)

    settings.get_path_manager.assert_not_called()
    parameterise.assert_not_called()


def test_initial_stage_is_dirty_before_sampling_starts(tmp_path):
    """Sampling cannot begin while workflow status still reports clean."""

    class SamplingReached(Exception):
        pass

    sampling_settings = MMMDSamplingSettings()
    training_settings = TrainingSettings()
    path_manager = WorkflowPathManager(
        output_dir=tmp_path,
        n_iterations=1,
        n_mols=1,
        training_settings=training_settings,
        training_sampling_settings=sampling_settings,
        testing_sampling_settings=sampling_settings,
    )
    settings = MagicMock()
    settings.output_dir = tmp_path
    settings.n_iterations = 1
    settings.get_path_manager.return_value = path_manager
    settings.param_settings = MagicMock()
    settings.device_type = "cpu"
    settings.device = "cpu"
    settings.training_settings = training_settings
    settings.testing_sampling_settings = sampling_settings

    tensor_force_field = MagicMock()
    tensor_force_field.potentials_by_type = {}

    def stop_at_sampling(**kwargs):
        assert path_manager.status == WorkflowStatus.PARTIAL
        assert (tmp_path / "initial_statistics" / "bespoke_ff.offxml").exists()
        raise SamplingReached

    with (
        patch(
            "presto.workflow.parameterise",
            return_value=(
                [MagicMock()],
                _WritingForceField("initial force field"),
                [MagicMock()],
                tensor_force_field,
            ),
        ),
        patch("presto.workflow.Trainable", return_value=MagicMock()),
        patch("presto.workflow.sample_ligands", side_effect=stop_at_sampling),
        pytest.raises(SamplingReached),
    ):
        get_bespoke_force_field(settings, write_settings=False)
