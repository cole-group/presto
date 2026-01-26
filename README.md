<h2 align="center">presto</h2>

<p align="center"> Parameter Refinement Engine for Smirnoff Training / Optimisation</p>

<p align="center">
  <a href="https://github.com/cole-lab/presto/actions/workflows/ci.yaml">
    <img src="https://github.com/cole-lab/presto/actions/workflows/ci.yaml/badge.svg" alt="CI" />
  </a>
  <a href="https://codecov.io/gh/cole-lab/presto" >
    <img src="https://codecov.io/gh/cole-lab/presto/graph/badge.svg?token=IBZ2H0NL58"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" />
  </a>
  <a href="https://mypy-lang.org/">
    <img alt="Checked with mypy" src="https://www.mypy-lang.org/static/mypy_badge.svg" />
  </a>
</p>

---

Train bespoke SMIRNOFF force fields using a machine learning potential (MLP). All valence parameters (bonds, angles, proper torsions, and improper torsions) are trained to MLP energies sampled using molecular dynamics. Please see the [**documentation**](https://cole-lab.github.io/presto/latest/).

***Warning**: This code is experimental and under active development. It is not guaranteed to provide correct results,
the documentation and testing is incomplete, and the API may change without notice.*

Please note that the MACE-OFF models are released under the [Academic Software License](https://github.com/gabor1/ASL/blob/main/ASL.md) which **does not permit commercial use**. However, the default AceFF-2.0 model (as well as Egret-1 and AIMNet-2) does.

## Installation

Ensuring that you have pixi installed, run:
```bash
git clone https://github.com/cole-lab/presto.git
cd presto
pixi install
```

## Usage

First, start a shell in the current environment (this must be run from the `presto` base directory)
```bash
pixi shell
```
For more information on activating pixi environments, see [the documentation](https://pixi.sh/latest/advanced/pixi_shell/#traditional-conda-activate-like-activation).

Run with command line arguments:
```bash
presto train --parameterisation-settings.smiles "CCC(CC)C(=O)Nc2cc(NC(=O)c1c(Cl)cccc1Cl)ccn2"
```

Sensible defaults have been set, but all available options can be viewed with:
```bash
presto train --help
```

Run from a yaml file:
```bash
presto write-default-yaml default.yaml
# Modify the yaml to set the desired smiles
presto train-from-yaml default.yaml
```

For more details on the theory and implementation, please see the [documentation](https://cole-lab.github.io/presto/latest/).

## MACE-Model Use

To use models with the MACE architecture, run
```
pixi shell -e mace-runtime
```

## Copyright

Copyright (c) 2025-2026, Finlay Clark, Newcastle University, UK

Copyright (c) 2025-2026, Thomas James Pope, Newcastle University, UK


This package includes models from other projects under the MIT license. See `presto/models/LICENSES.md` for details.

## Acknowledgements

Early development was completed by Thomas James Pope. Many ideas taken from Simon Boothroyd's super helpful [python-template](https://github.com/SimonBoothroyd/python-template).
