"""Unit tests for writers."""

import io
from unittest.mock import MagicMock, patch

import h5py
import openff.interchange
import smee
import smee.converters
import torch
from descent.train import ParameterConfig, Trainable
from openff.toolkit import ForceField, Molecule

from presto.data_utils import create_dataset_with_uniform_weights
from presto.loss import DatasetLoss, LossRecord, Prediction
from presto.writers import (
    get_potential_comparison,
    get_potential_summary,
    metrics_columns,
    report,
    write_metrics,
    write_metrics_header,
    write_scatter,
)


def test_write_scatter(tmp_path):
    """Test write_scatter function."""
    # Setup dummy data
    n_conf = 2
    n_atoms = 3
    dataset = [
        {
            "smiles": "C",
            "coords": torch.zeros(n_conf, n_atoms, 3),
            "energy": torch.zeros(n_conf),
            "forces": torch.zeros(n_conf, n_atoms, 3),
        }
    ]

    force_field = MagicMock(spec=smee.TensorForceField)
    topology = MagicMock(spec=smee.TensorTopology)
    device = torch.device("cpu")
    filename = tmp_path / "scatter.h5"

    # Mock predict function from presto.loss
    with patch("presto.writers.predict_with_weights") as mock_predict:
        mock_predict.return_value = Prediction(
            energy_ref=torch.zeros(n_conf),
            energy_pred=torch.ones(n_conf),
            forces_ref=torch.zeros(n_conf * n_atoms * 3),
            forces_pred=torch.ones(n_conf * n_atoms * 3),
            energy_weights=torch.ones(n_conf),
            forces_weights=torch.ones(n_conf),
            energy_conformer_weights=torch.ones(n_conf),
            forces_conformer_weights=torch.ones(n_conf),
            entry_sizes=[n_conf],
        )

        energy_mean, energy_std, forces_mean, forces_std = write_scatter(
            dataset, force_field, topology, device, filename
        )

        assert energy_mean == 1.0
        assert energy_std == 0.0
        assert forces_mean == 1.0
        assert forces_std == 0.0

        # Verify HDF5 file
        assert filename.exists()
        with h5py.File(filename, "r") as f:
            assert "energy_reference" in f
            assert "energy_predicted" in f
            assert "forces_reference" in f
            assert f.attrs["n_conformers"] == n_conf
            # Implementation currently stores n_atoms * 3 in n_atoms attribute
            assert f.attrs["n_atoms"] == n_atoms * 3


def test_get_potential_summary():
    """Test get_potential_summary function."""
    potential = MagicMock(spec=smee.TensorPotential)
    potential.type = "Bonds"
    potential.fn = "harmonic"
    potential.parameters = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    potential.parameter_cols = ("k", "length")
    potential.attributes = None

    summary = get_potential_summary(potential)
    assert "Bonds" in summary
    assert "harmonic" in summary
    assert "1.0000" in summary
    assert "4.0000" in summary


def test_get_potential_comparison():
    """Test get_potential_comparison function."""
    pot1 = MagicMock(spec=smee.TensorPotential)
    pot1.type = "Bonds"
    pot1.fn = "harmonic"
    pot1.parameters = torch.tensor([[1.0, 2.0]])
    pot1.parameter_cols = ("k", "length")
    pot1.attributes = None

    pot2 = MagicMock(spec=smee.TensorPotential)
    pot2.type = "Bonds"
    pot2.parameters = torch.tensor([[1.1, 2.1]])

    comparison = get_potential_comparison(pot1, pot2)
    assert "Bonds" in comparison
    assert "1.0000 --> 1.1000" in comparison
    assert "2.0000 --> 2.1000" in comparison


def test_get_potential_summary_with_attributes():
    """Test get_potential_summary function with attributes."""
    potential = MagicMock(spec=smee.TensorPotential)
    potential.type = "vdW"
    potential.fn = "LennardJones"
    potential.parameters = torch.tensor([[1.0, 2.0]])
    potential.parameter_cols = ("epsilon", "sigma")
    potential.attributes = torch.tensor([0.5])
    potential.attribute_cols = ("scale",)
    potential.attribute_units = ("dimensionless",)

    summary = get_potential_summary(potential)
    assert "vdW" in summary
    assert "attributes=" in summary
    assert "scale" in summary
    assert "0.5000" in summary


def test_get_potential_comparison_with_attributes():
    """Test get_potential_comparison function with attributes."""
    pot1 = MagicMock(spec=smee.TensorPotential)
    pot1.type = "vdW"
    pot1.fn = "LennardJones"
    pot1.parameters = torch.tensor([[1.0, 2.0]])
    pot1.parameter_cols = ("epsilon", "sigma")
    pot1.attributes = torch.tensor([0.5])
    pot1.attribute_cols = ("scale",)
    pot1.attribute_units = ("dimensionless",)

    pot2 = MagicMock(spec=smee.TensorPotential)
    pot2.type = "vdW"
    pot2.parameters = torch.tensor([[1.1, 2.1]])

    comparison = get_potential_comparison(pot1, pot2)
    assert "vdW" in comparison
    assert "attributes=" in comparison
    assert "0.5000" in comparison


def test_metrics_columns():
    """Combined losses come first, then each training and test dataset's losses."""
    assert metrics_columns(2, 1) == [
        "loss_train_energy",
        "loss_train_forces",
        "loss_train_regularisation",
        "loss_test_energy",
        "loss_test_forces",
        "loss_test_regularisation",
        "loss_train_dataset_0_energy",
        "loss_train_dataset_0_forces",
        "loss_train_dataset_1_energy",
        "loss_train_dataset_1_forces",
        "loss_test_dataset_0_energy",
        "loss_test_dataset_0_forces",
    ]


def test_write_metrics_header_and_row():
    """The header names each value written, which is also its TensorBoard tag."""
    writer = MagicMock()
    outfile = io.StringIO()
    loss_train = LossRecord(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0))
    loss_test = LossRecord(torch.tensor(4.0), torch.tensor(5.0), torch.tensor(3.0))
    train_datasets = [
        DatasetLoss(torch.tensor(6.0), torch.tensor(7.0)),
        DatasetLoss(torch.tensor(8.0), torch.tensor(9.0)),
    ]
    test_datasets = [DatasetLoss(torch.tensor(10.0), torch.tensor(11.0))]

    write_metrics_header(outfile, 2, 1)
    write_metrics(
        7, loss_train, loss_test, train_datasets, test_datasets, writer, outfile
    )

    header, row = outfile.getvalue().splitlines()
    values = [float(v) for v in row.split()]
    assert header.split() == metrics_columns(2, 1)
    assert values == [1, 2, 3, 4, 5, 3, 6, 7, 8, 9, 10, 11]
    assert [c.args for c in writer.add_scalar.call_args_list] == [
        (name, value, 7) for name, value in zip(header.split(), values, strict=True)
    ]


def test_report_appends_row_for_each_dataset(tmp_path):
    """The Levenberg-Marquardt reporter appends one row covering every dataset."""
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=1)
    ff = ForceField("openff_unconstrained-2.3.0.offxml")
    interchange = openff.interchange.Interchange.from_smirnoff(ff, mol.to_topology())
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)
    trainable = Trainable(
        tensor_ff,
        {"Bonds": ParameterConfig(cols=["k"], scales={"k": 1.0}, regularize={"k": 1})},
        {},
    )
    x = trainable.to_values()
    coords = torch.tensor(mol.conformers[0].m_as("angstrom")).unsqueeze(0)
    dataset = create_dataset_with_uniform_weights(
        smiles=mol.to_smiles(mapped=True),
        coords=coords,
        energy=torch.ones(1),
        forces=torch.ones_like(coords),
    )
    metrics_file = tmp_path / "metrics.txt"
    with open(metrics_file, "w") as f:
        write_metrics_header(f, 2, 3)

    report(
        0,
        x,
        None,
        None,
        None,
        0.0,
        True,
        trainable=trainable,
        topologies=[tensor_top],
        datasets_train=[[dataset]] * 2,
        datasets_test=[[dataset]] * 3,
        dataset_weights=[1.0, 2.0],
        dataset_weights_test=[1.0, 1.0, 1.0],
        initial_parameters=x.clone(),
        regularisation_target="initial",
        metrics_file=metrics_file,
        experiment_dir=tmp_path / "tensorboard",
    )

    header, row = metrics_file.read_text().splitlines()
    assert len(row.split()) == len(header.split()) == len(metrics_columns(2, 3))
