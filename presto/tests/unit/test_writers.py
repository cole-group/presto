from unittest.mock import MagicMock, patch

import h5py
import smee
import torch

from presto.writers import (
    get_potential_comparison,
    get_potential_summary,
    write_scatter,
)


def test_write_scatter(tmp_path):
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
    device_type = "cpu"
    filename = tmp_path / "scatter.h5"

    # Mock predict function from presto.loss
    with patch("presto.writers.predict_with_weights") as mock_predict:
        # Returns: energy_ref, energy_pred, forces_ref, forces_pred
        mock_predict.return_value = (
            torch.zeros(n_conf),
            torch.ones(n_conf),
            torch.zeros(n_conf * n_atoms * 3),
            torch.ones(n_conf * n_atoms * 3),
        )

        energy_mean, energy_std, forces_mean, forces_std = write_scatter(
            dataset, force_field, topology, device_type, filename
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
