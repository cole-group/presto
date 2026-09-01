"""Tests for runtime workflow molecule-input validation."""

from unittest import mock

import pytest
from openff.toolkit import Molecule

from presto._exceptions import InvalidSettingsError
from presto.settings import (
    MMMDSamplingSettings,
    MSMSettings,
    ParamSettings,
    PreComputedDatasetSettings,
    WorkflowSettings,
)
from presto.workflow import _validate_workflow_molecule_inputs, get_bespoke_force_field


def _write_conformer_sdf(smiles, path):
    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)
    molecule.to_file(str(path), "SDF")


def test_reports_every_etkdg_failure_and_every_requiring_stage():
    """All ETKDG failures and the stages requiring it are reported together."""
    settings = WorkflowSettings(
        param_settings=ParamSettings(
            molecule_input_type="smiles", molecules=["CCO", "CCC", "CCCC"]
        ),
        device_type="cpu",
    )

    with mock.patch.object(
        Molecule,
        "generate_conformers",
        autospec=True,
        side_effect=ValueError("RDKit conformer generation failed"),
    ) as generate:
        with pytest.raises(InvalidSettingsError) as exc_info:
            _validate_workflow_molecule_inputs(settings)

    assert generate.call_count == 3
    message = str(exc_info.value)
    assert "3 of 3 molecules" in message
    for index in range(3):
        assert f"molecule {index} " in message
    assert "testing_sampling_settings" in message
    assert "training_sampling_settings" in message
    assert "param_settings.msm_settings" in message


def test_supplied_geometries_bypass_etkdg_without_library_charges(tmp_path):
    """Supplying every workflow geometry bypasses ETKDG without custom charges."""
    sdf = tmp_path / "ethanol.sdf"
    _write_conformer_sdf("CCO", sdf)
    sampling = MMMDSamplingSettings(starting_conformers=sdf)
    settings = WorkflowSettings(
        param_settings=ParamSettings(
            molecule_input_type="smiles",
            molecules="CCO",
            msm_settings=MSMSettings(starting_conformers=sdf),
        ),
        training_sampling_settings=sampling,
        testing_sampling_settings=sampling,
        device_type="cpu",
    )

    with mock.patch.object(Molecule, "generate_conformers", autospec=True) as generate:
        molecules = _validate_workflow_molecule_inputs(settings)

    generate.assert_not_called()
    assert len(molecules) == 1


def test_precomputed_stages_and_disabled_msm_do_not_use_etkdg(tmp_path):
    """Stages which do not generate geometries do not trigger an ETKDG probe."""
    precomputed = PreComputedDatasetSettings(dataset_paths=[tmp_path / "dataset"])
    settings = WorkflowSettings(
        param_settings=ParamSettings(
            molecule_input_type="smiles", molecules="CCO", msm_settings=None
        ),
        training_sampling_settings=precomputed,
        testing_sampling_settings=precomputed,
        device_type="cpu",
    )

    with mock.patch.object(Molecule, "generate_conformers", autospec=True) as generate:
        _validate_workflow_molecule_inputs(settings)

    generate.assert_not_called()


def test_aggregates_supplied_conformer_mismatches(tmp_path):
    """Every molecule missing from a supplied SDF is reported together."""
    sdf = tmp_path / "ethanol.sdf"
    _write_conformer_sdf("CCO", sdf)
    precomputed = PreComputedDatasetSettings(dataset_paths=[tmp_path / "dataset"])
    settings = WorkflowSettings(
        param_settings=ParamSettings(
            molecule_input_type="smiles",
            molecules=["CCC", "CCCC"],
            msm_settings=MSMSettings(starting_conformers=sdf),
        ),
        training_sampling_settings=precomputed,
        testing_sampling_settings=precomputed,
        device_type="cpu",
    )

    with pytest.raises(InvalidSettingsError) as exc_info:
        _validate_workflow_molecule_inputs(settings)

    message = str(exc_info.value)
    assert "2 of 2 molecules" in message
    assert "molecule 0 (CCC)" in message
    assert "molecule 1 (CCCC)" in message
    assert "param_settings.msm_settings.starting_conformers" in message


def test_preflight_failure_creates_no_output(tmp_path):
    """Input validation precedes creation of the workflow output tree."""
    output_dir = tmp_path / "output"
    settings = WorkflowSettings(
        param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
        output_dir=output_dir,
        device_type="cpu",
    )

    with mock.patch.object(
        Molecule,
        "generate_conformers",
        autospec=True,
        side_effect=ValueError("cannot embed"),
    ):
        with pytest.raises(InvalidSettingsError):
            get_bespoke_force_field(settings)

    assert not output_dir.exists()
