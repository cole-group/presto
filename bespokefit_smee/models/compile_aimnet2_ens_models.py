"""Script to compile AIMNet2 ensemble models for use in BespokeFit."""

import torch
from torch import nn, Tensor
from typing import Dict, List, Literal, get_args
import urllib.request
import tempfile
from loguru import logger

from bespokefit_smee.utils.typing import TorchDevice

_MODEL_URL = "https://storage.googleapis.com/aimnetcentral/AIMNet2/"
AvailableModels = Literal["aimnet2_b973c_d3", "aimnet2_wb97m_d3"]
_AVAILABLE_MODELS = get_args(AvailableModels)


def _download_model(
    method: str, version: int = 0, device: TorchDevice | None = None
) -> torch.jit.ScriptModule:
    """Download an AIMNet2 model directly from storage."""
    url = f"{_MODEL_URL}{method}_{version}.jpt"

    with tempfile.NamedTemporaryFile(suffix=".jpt") as tmp_file:
        urllib.request.urlretrieve(url, filename=tmp_file.name)
        model: torch.jit.ScriptModule = torch.jit.load(  # type: ignore[no-untyped-call]
            tmp_file.name, map_location=device
        )
        return model


# This function was taken from https://github.com/jthorton/AIMNet2/blob/main/pyaimnet2/models/ensemble.py
class EnsembledModel(nn.Module):
    """Create ensemble of AIMNet2 models."""

    def __init__(
        self,
        models: List,
        x=["coord", "numbers", "charge"],
        out=["energy", "forces", "charges"],
        detach=True,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.x = x
        self.out = out
        self.detach = detach

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        res: List[Dict[str, Tensor]] = []
        for model in self.models:
            _in = dict()
            for k in data:
                if k in self.x:
                    _in[k] = data[k]
            _out = model(_in)
            _r = dict()
            for k in _out:
                if k in self.out:
                    _r[k] = _out[k]
                    if self.detach:
                        _r[k] = _r[k].detach()
            res.append(_r)

        for k in res[0]:
            v = []
            for x in res:
                v.append(x[k])
            vv = torch.stack(v, dim=0)
            data[k] = vv.mean(dim=0)
            data[k + "_std"] = vv.std(dim=0)

        return data


def compile_aimnet2_ens_model(
    model_name: AvailableModels,
    n_members: int = 4,
    device: TorchDevice = "cpu",
) -> torch.jit.ScriptModule:
    """Compile an AIMNet2 ensemble model.

    Args:
        model_name: Name of the AIMNet2 model to compile.
        n_members: Number of ensemble members to include.
        device: Torch device to load models onto.

    Returns:
        Compiled AIMNet2 ensemble model.
    """
    if model_name not in get_args(AvailableModels):
        raise ValueError(
            f"Invalid model name: {model_name}. Available models are: {get_args(AvailableModels)}"
        )

    models = []
    for i in range(n_members):
        model = _download_model(model_name, version=i, device=device)
        models.append(model)

    ensemble_model = EnsembledModel(models=models, detach=False)
    scripted_model = torch.jit.script(ensemble_model)  # type: ignore[no-untyped-call]

    return scripted_model


def main():
    """Main function to compile and save AIMNet2 ensemble models."""
    for model_name in get_args(AvailableModels):
        logger.info(f"Compiling ensemble model for {model_name}...")
        ens_model = compile_aimnet2_ens_model(model_name, n_members=4, device="cpu")
        save_path = f"{model_name}_ens.jpt"
        ens_model.save(save_path)
        logger.info(f"Saved ensemble model to {save_path}")


if __name__ == "__main__":
    main()
