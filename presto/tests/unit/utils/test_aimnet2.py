from unittest.mock import MagicMock, patch

import openmm
import pytest
import torch

from presto.utils.aimnet2 import (
    AIMNet2PotentialImpl,
    AIMNet2PotentialImplFactory,
    _register_aimnet2_potentials,
)


def test_aimnet2_factory():
    factory = AIMNet2PotentialImplFactory()
    impl = factory.createImpl("model_name")
    assert isinstance(impl, AIMNet2PotentialImpl)
    assert impl.name == "model_name"


@patch("presto.utils.aimnet2._load_local_model")
@patch("presto.utils.aimnet2.openmmtorch.TorchForce")
@patch("presto.utils.aimnet2.torch.jit.script")
def test_aimnet2_impl_add_forces(
    mock_script, mock_torch_force_cls, mock_load, tmp_path
):
    impl = AIMNet2PotentialImpl("aimnet2_b973c_d3_ens")

    topology = MagicMock(spec=openmm.app.Topology)
    topology.atoms.return_value = [MagicMock(element=MagicMock(atomic_number=6))]

    system = MagicMock(spec=openmm.System)
    mock_load.return_value = MagicMock(spec=torch.jit.ScriptModule)

    mock_torch_force = MagicMock()
    mock_torch_force_cls.return_value = mock_torch_force

    impl.addForces(topology, system, atoms=[0], forceGroup=1, charge=0)

    assert mock_load.called
    assert mock_torch_force_cls.called
    assert system.addForce.called
    mock_torch_force.setForceGroup.assert_called_with(1)


def test_aimnet2_invalid_model():
    impl = AIMNet2PotentialImpl("invalid_model")
    topology = MagicMock()
    system = MagicMock()
    with pytest.raises(ValueError, match="Unsupported AIMNet2 model"):
        impl.addForces(topology, system, atoms=None, forceGroup=0, charge=0)


@patch("presto.utils.aimnet2.MLPotential.registerImplFactory")
def test_register_aimnet2(mock_register):
    _register_aimnet2_potentials()
    assert mock_register.call_count >= 2
