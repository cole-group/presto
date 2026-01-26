<h2 align="center">presto</h2>

<p align="center">
  <a href="https://github.com/fjclark/presto/actions/workflows/ci.yaml">
    <img src="https://github.com/fjclark/presto/actions/workflows/ci.yaml/badge.svg" alt="CI" />
  </a>
  <a href="https://codecov.io/gh/fjclark/presto" >
    <img src="https://codecov.io/gh/fjclark/presto/graph/badge.svg?token=IBZ2H0NL58"/>
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

Generate a Bespoke Force-Field Parametrization Quickly and Reliably. Developed in the [Cole Group](https://blogs.ncl.ac.uk/danielcole/about-us/) at Newcastle University. Please see the [**documentation**](https://fjclark.github.io/presto/latest/).

***Warning**: This code is experimental and under active development. It is not guaranteed to provide correct results,
the documentation and testing is incomplete, and the API may change without notice.*

Please note that the MACE-OFF models are released under the [Academic Software License](https://github.com/gabor1/ASL/blob/main/ASL.md) which **does not permit commercial use**. However, the default Egret-1 model and AIMNet-2 models do.

## What is presto?

presto is a Force-Field parametrization tool. For a given molecule, it will generate a data set of conformers using machine learning models in [OpenMM-ML](https://github.com/openmm/openmm-ml) simulations. This dataset is used to optimise the force field parameters.

## Installation

Ensuring that you have pixi installed, run:
```bash
git clone https://github.com/fjclark/presto.git
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

For more details on the theory and implementation, please see the [documentation](https://fjclark.github.io/presto/latest/).

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

All early development was completed by Thomas James Pope. Many ideas taken from Simon Boothroyd's super helpful [python-template](https://github.com/SimonBoothroyd/python-template).
