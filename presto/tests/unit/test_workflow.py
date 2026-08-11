"""Unit tests for workflow.py."""

import openff.interchange
import pytest
import smee
import smee.converters
import torch
from descent.train import AttributeConfig, ParameterConfig, Trainable
from openff.toolkit import ForceField, Molecule
from openff.units import unit

from presto import workflow
from presto.data_utils import create_dataset_with_uniform_weights
from presto.settings import ParamSettings, WorkflowSettings
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


class TestTrainingDeviceSplit:
    """Training is pinned to the CPU while sampling honours ``device_type``.

    This covers the temporary hack in ``WorkflowSettings.training_device_type``;
    delete alongside it once training can run on the GPU again.
    """

    def test_sampling_uses_configured_device_and_training_uses_cpu(
        self, monkeypatch, tmp_path
    ):
        """Samplers get the configured device, everything tensor-side gets the CPU."""
        # Pretend a GPU is present so ``device_type="cuda"`` validates. No CUDA
        # tensor is ever created, since the stubs below stand in for the work that
        # would touch the device.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=1, rms_cutoff=0.0 * unit.angstrom)

        off_ff = ForceField("openff_unconstrained-2.3.0.offxml")
        interchange = openff.interchange.Interchange.from_smirnoff(
            off_ff, mol.to_topology()
        )
        tensor_ff, tensor_tops = smee.converters.convert_interchange([interchange])

        coords = torch.tensor(mol.conformers[0].m_as("angstrom")).unsqueeze(0)
        dataset = create_dataset_with_uniform_weights(
            smiles=mol.to_smiles(mapped=True),
            coords=coords,
            energy=torch.zeros(1, dtype=torch.float64),
            forces=torch.zeros_like(coords, dtype=torch.float64),
            energy_weight=1.0,
            forces_weight=1.0,
        )

        devices: dict[str, list] = {}

        def record(key, device):
            devices.setdefault(key, []).append(device)

        def fake_parameterise(param_settings, device):
            record("parameterise", device)
            return [mol], off_ff, tensor_tops, tensor_ff

        def fake_sample(*, mols, off_ff, device, settings, output_paths):
            record("sample", device)
            return [dataset]

        def fake_train(**kwargs):
            record("train", kwargs["device"])
            return kwargs["trainable_parameters"], kwargs["trainable"]

        def fake_write_scatter(dataset, force_field, topology, device, filename):
            record("scatter", device)
            return 0.0, 0.0, 0.0, 0.0

        def fake_filter(*, dataset, force_field, topology, settings, device):
            record("filter", device)
            return dataset

        monkeypatch.setattr(workflow, "parameterise", fake_parameterise)
        monkeypatch.setattr(workflow, "write_scatter", fake_write_scatter)
        monkeypatch.setattr(workflow, "filter_dataset_outliers", fake_filter)
        monkeypatch.setattr(workflow, "analyse_workflow", lambda settings: None)

        settings = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            device_type="cuda",
            n_iterations=1,
            output_dir=tmp_path,
        )

        for sampling_settings in (
            settings.training_sampling_settings,
            settings.testing_sampling_settings,
        ):
            monkeypatch.setitem(
                workflow._SAMPLING_FNS_REGISTRY, type(sampling_settings), fake_sample
            )
        monkeypatch.setitem(
            workflow._TRAINING_FNS_REGISTRY,
            settings.training_settings.optimiser,
            fake_train,
        )

        workflow.get_bespoke_force_field(settings, write_settings=False)

        # Sampling (and the MSM step inside parameterisation) stay on the GPU
        assert devices["parameterise"] == ["cuda"]
        assert devices["sample"] == [torch.device("cuda"), torch.device("cuda")]

        # Everything that touches the tensor force field runs on the CPU
        assert devices["train"] == [torch.device("cpu")]
        assert devices["filter"] == [torch.device("cpu")]
        assert set(devices["scatter"]) == {torch.device("cpu")}

        assert tensor_ff.potentials[0].parameters.device.type == "cpu"
