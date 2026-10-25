"""Integration test for the presto package."""

import subprocess

from ...settings import WorkflowSettings
from ...tests.conftest import skip_if_model_unavailable


def test_workflow_cli(tmp_cwd) -> None:
    """Test running presto via CLI with ethanol, and analysing."""
    skip_if_model_unavailable("aimnet2")

    # Run the command
    args = [
        "presto",
        "train",
        "--device-type",
        "cpu",  # So we can run on GH actions
        "--n-iterations",
        "1",  # Super short run for testing
        "--param-settings.molecule-input-type",
        "smiles",
        "--param-settings.molecules",
        "CCCO",  # Propanol - 1 heavy-atom dihedral to plot
        "--training-settings.n-epochs",
        "10",
        "--training-sampling-settings.sampling-protocol",
        "mm_md",
        "--training-sampling-settings.n-conformers",
        "1",
        "--training-sampling-settings.production-sampling-time-per-conformer",
        "5 ps",
        "--testing-sampling-settings.sampling-protocol",
        "mm_md",
        "--testing-sampling-settings.n-conformers",
        "1",
        "--testing-sampling-settings.production-sampling-time-per-conformer",
        "1 ps",
    ]
    print(" ".join(args))

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )

    # Check the command executed successfully
    assert result.returncode == 0, f"Command failed: {result.stderr}"

    # Create the Path manager and make sure all the expected files have been created
    # and are in the right places
    workflow_settings_yaml = tmp_cwd / "workflow_settings.yaml"
    assert workflow_settings_yaml.exists(), "Workflow settings YAML not found."

    settings = WorkflowSettings.from_yaml(workflow_settings_yaml)
    path_manager = settings.get_path_manager()

    expected_output_files = path_manager.get_all_output_paths(only_if_exists=False)
    for stage, outputs in expected_output_files.items():
        for output_type, path_or_paths in outputs.items():
            # Handle both single paths and lists of paths (for per-molecule outputs)
            paths = (
                path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
            )
            for path in paths:
                assert path.exists(), (
                    f"Expected output {output_type} in stage {stage} not found at {path}."
                )


def test_workflow_multiple_datasets(tmp_path) -> None:
    """Fit to two training and two test datasets, then analyse and clean."""
    skip_if_model_unavailable("aimnet2")

    from openmm import unit

    from ...outputs import WorkflowStatus
    from ...settings import (
        MetadynamicsSettings,
        MLMDSamplingSettings,
        MMMDMetadynamicsSamplingSettings,
        MMMDSamplingSettings,
        ParamSettings,
        TrainingSettings,
    )
    from ...workflow import get_bespoke_force_field

    short = {
        "n_conformers": 1,
        "production_sampling_time_per_conformer": 2 * unit.picoseconds,
    }
    settings = WorkflowSettings(
        output_dir=tmp_path,
        device_type="cpu",
        n_iterations=2,
        memory=True,
        param_settings=ParamSettings(molecules="CCCO"),
        training_settings=TrainingSettings(n_epochs=10),
        training_sampling_settings=[
            MMMDSamplingSettings(**short),
            MMMDMetadynamicsSamplingSettings(
                **short,
                dataset_weight=3.0,
                metadynamics_settings=MetadynamicsSettings(
                    bias_save_frequency=1 * unit.picoseconds
                ),
            ),
        ],
        testing_sampling_settings=[
            MMMDSamplingSettings(**short),
            MLMDSamplingSettings(**short, dataset_weight=0.5),
        ],
    )

    get_bespoke_force_field(settings)

    # The settings written for the run reload as the same datasets
    reloaded = WorkflowSettings.from_yaml(tmp_path / "workflow_settings.yaml")
    assert reloaded.training_sampling_settings == settings.training_sampling_settings
    assert reloaded.testing_sampling_settings == settings.testing_sampling_settings

    path_manager = settings.get_path_manager()
    assert path_manager.status == WorkflowStatus.COMPLETE
    for stage, outputs in path_manager.get_all_output_paths(
        only_if_exists=False
    ).items():
        for output_type, path_or_paths in outputs.items():
            paths = (
                path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
            )
            for path in paths:
                assert path.exists(), (
                    f"Expected output {output_type} in stage {stage} not found at {path}."
                )

    header = (tmp_path / "training_iteration_2" / "metrics.txt").read_text().split()
    assert "loss_train_dataset_1_forces" in header
    assert "loss_test_dataset_1_energy" in header

    path_manager.clean()
    assert path_manager.status == WorkflowStatus.CLEAN
    assert [p.name for p in tmp_path.iterdir()] == ["workflow_settings.yaml"]
