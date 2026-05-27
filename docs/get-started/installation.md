# Installation

The easiest way to install `presto` is from the GitHub repo using [pixi](https://pixi.sh/). There is no `conda-forge` release yet (as we have a mix of dependencies which are only available on either `conda-forge` or PyPI).

## Prerequisites

- **pixi** — see [pixi installation docs](https://pixi.sh/latest/).
- **NVIDIA driver compatible with CUDA 12.9 or newer** for the GPU environment. Check with `nvidia-smi`. Older driver versions are not usable because `presto` requires OpenMM 8.5 (for `PythonForce`), and OpenMM 8.5 requires CUDA 12.9.
- A CPU-only environment is available for development and docs builds but is very slow for real fits.

## Install with pixi

```bash
git clone https://github.com/cole-group/presto.git
cd presto
pixi shell
```

This creates the default GPU environment (`gpu-py313-cuda129`) and activates a shell in it. For more on activating pixi environments, see [the pixi shell docs](https://pixi.sh/latest/advanced/pixi_shell/).

## Verify the install

```bash
presto version
```

If `presto version` prints a version string, you're ready to move on to **[Quickstart](quickstart.md)**.

## Optional environments

The default GPU environment includes every supported MLP feature. Defined in `pyproject.toml [tool.pixi.environments]`:

| Environment | Use when |
|---|---|
| `gpu-py313-cuda129` (default) | Normal GPU workflow on Python 3.13. |
| `gpu-py312-cuda129` | GPU workflow on Python 3.12.  |
| `cpu-py312`, `cpu-py313` | Tests, docs, or environments without a CUDA-capable GPU. |
| `base*` (No MLP features) | No MLP deps desired |

Activate a non-default environment with:

```bash
pixi shell -e cpu-py312
```

## MLP licensing notes

The `pixi` `presto` install from GitHub brings in several reference MLPs by default. Most have permissive licences but a few do not:

- **AIMNet2** (MIT) — current default; permits commercial use.
- **Orb-v3 (OMol25 variants)** (Apache-2.0) — permits commercial use.
- **Egret-1** (MIT) — permits commercial use.
- **AceFF-2.0** — permits commercial use, but [is currently broken upstream](https://github.com/openmm/openmm-ml/issues/136).
- **MACE-OFF** — released under the [Academic Software License](https://github.com/gabor1/ASL/blob/main/ASL.md). **Not for commercial use.**

For more, see **[Concepts → MLPs in presto](../concepts/mlps.md)** and **[How-to → Choose an MLP](../how-to/choose-an-mlp.md)**.

## Troubleshooting

If anything went wrong during install or first run, see **[Reference → Troubleshooting](../reference/troubleshooting.md)**.
