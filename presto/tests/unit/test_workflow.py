"""Unit tests for workflow lifecycle behavior."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import openff.interchange
import pytest
import smee.converters
from openff.toolkit import ForceField, Molecule

from presto._exceptions import MissingOptionalDependencyError
from presto.corrections import get_system_charges
from presto.outputs import WorkflowPathManager, WorkflowStatus
from presto.settings import (
    AnySolvSMDCorrectionSettings,
    GBn2SASACorrectionSettings,
    MLMDSamplingSettings,
    ParamSettings,
    WorkflowSettings,
)
from presto.workflow import _all_corrections, get_bespoke_force_field


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


def _solvated_settings(tmp_path, **kwargs) -> WorkflowSettings:
    return WorkflowSettings(
        param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
        device_type="cpu",
        output_dir=tmp_path,
        n_iterations=1,
        reference_corrections=[
            AnySolvSMDCorrectionSettings(checkpoint="model_compact")
        ],
        mm_corrections=[GBn2SASACorrectionSettings()],
        **kwargs,
    )


def test_missing_correction_dependency_fails_before_parameterisation(
    tmp_path, monkeypatch
):
    """A correction which cannot be evaluated fails before any slow work."""
    monkeypatch.setitem(sys.modules, "anysolv", None)
    settings = _solvated_settings(tmp_path)

    with (
        patch("presto.workflow.parameterise") as parameterise,
        pytest.raises(MissingOptionalDependencyError, match="conda-forge"),
    ):
        get_bespoke_force_field(settings, write_settings=False)

    parameterise.assert_not_called()


def test_all_corrections_include_explicit_sampling_corrections(tmp_path):
    """Every correction the workflow may evaluate is checked up front."""
    sampling_gb = GBn2SASACorrectionSettings(sasa_model="ace")
    settings = _solvated_settings(
        tmp_path,
        testing_sampling_settings=MLMDSamplingSettings(
            mm_sampling_corrections=[sampling_gb]
        ),
    )

    assert _all_corrections(settings) == [
        *settings.reference_corrections,
        *settings.mm_corrections,
        sampling_gb,
    ]


def test_corrections_and_smee_charges_are_passed_to_sampling(tmp_path, fake_anysolv):
    """Training and testing sampling get the fit corrections and the MM charges."""
    settings = _solvated_settings(tmp_path)
    mol = Molecule.from_smiles("CCO")
    force_field = ForceField("openff_unconstrained-2.3.0.offxml")
    interchange = openff.interchange.Interchange.from_smirnoff(
        force_field, mol.to_topology()
    )
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)
    expected_charges = get_system_charges(interchange.to_openmm_system())

    sample_calls = []

    def fake_sample_ligands(**kwargs):
        sample_calls.append(kwargs)
        return [MagicMock()]

    trainable = MagicMock(_param_types=[])
    train_fn = MagicMock(return_value=(MagicMock(), trainable))
    with (
        patch(
            "presto.workflow.parameterise",
            return_value=([mol], force_field, [tensor_top], tensor_ff),
        ),
        patch("presto.workflow.Trainable"),
        patch("presto.workflow.sample_ligands", side_effect=fake_sample_ligands),
        patch("presto.workflow.write_scatter", return_value=(0.0, 0.0, 0.0, 0.0)),
        patch("presto.workflow.convert_to_smirnoff", return_value=force_field),
        patch.dict("presto.workflow._TRAINING_FNS_REGISTRY", {"adam": train_fn}),
        patch("presto.workflow.analyse_workflow"),
    ):
        get_bespoke_force_field(settings, write_settings=False)

    assert len(sample_calls) == 2  # testing, then one training iteration
    for call in sample_calls:
        assert call["reference_corrections"] == settings.reference_corrections
        assert call["mm_corrections"] == settings.mm_corrections
        assert call["min_conformations"] == settings.min_conformations
        assert np.allclose(call["mm_charges"][0], expected_charges, atol=1e-6)
