"""Unit tests for workflow.py."""

import openff.interchange
import pytest
import smee
import smee.converters
from descent.train import AttributeConfig, ParameterConfig, Trainable
from openff.toolkit import ForceField, Molecule

from presto.workflow import _prune_configs


@pytest.fixture
def ethanol_tensor_ff():
    """A Lennard-Jones tensor force field for ethanol."""
    mol = Molecule.from_smiles("CCO")
    off_ff = ForceField("openff_unconstrained-2.3.0.offxml")
    interchange = openff.interchange.Interchange.from_smirnoff(
        off_ff, mol.to_topology()
    )
    tensor_ff, _ = smee.converters.convert_interchange([interchange])
    return tensor_ff


class TestPruneConfigs:
    """Tests for dropping training configs for potential types absent from the force field."""

    def test_keeps_present_types(self, ethanol_tensor_ff):
        """A config for a potential the force field contains is left alone."""
        configs = {"Bonds": ParameterConfig(cols=["k"])}
        assert _prune_configs(configs, ethanol_tensor_ff) == configs

    def test_drops_absent_parameter_types(self, ethanol_tensor_ff):
        """A LinearBonds config is meaningless unless harmonics were linearised."""
        configs = {
            "Bonds": ParameterConfig(cols=["k"]),
            "LinearBonds": ParameterConfig(cols=["k1", "k2"]),
        }
        assert set(_prune_configs(configs, ethanol_tensor_ff)) == {"Bonds"}

    def test_drops_absent_attribute_types(self, ethanol_tensor_ff):
        """An attribute config for a missing potential must not reach Trainable.

        ``Trainable`` looks the type up in ``potentials_by_type`` directly, so an
        unpruned config raises a bare ``KeyError``.
        """
        configs = {"NotAPotential": AttributeConfig(cols=["scale_14"])}

        with pytest.raises(KeyError):
            Trainable(ethanol_tensor_ff, {}, configs)

        assert _prune_configs(configs, ethanol_tensor_ff) == {}

    def test_attribute_configs_survive_pruning_and_train(self, ethanol_tensor_ff):
        """A pruned attribute config still yields a working Trainable."""
        configs = {
            "vdW": AttributeConfig(cols=["scale_14"]),
            "NotAPotential": AttributeConfig(cols=["alpha"]),
        }
        pruned = _prune_configs(configs, ethanol_tensor_ff)
        assert set(pruned) == {"vdW"}

        trainable = Trainable(ethanol_tensor_ff, {}, pruned)
        assert trainable.to_values().shape == (1,)
