# Use an ASE calculator

OpenMM-ML supports an `ase` adapter that wraps any [ASE Calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculator.html) as an OpenMM `MLPotential`. This lets you use any MLP that has an ASE interface — not just the ones that ship with OpenMM-ML.

## When you need this

- You're using an MLP that has an ASE calculator but no native OpenMM-ML support.
- You're prototyping a model and want to plug it in without writing an OpenMM-ML wrapper.

## Pass a calculator from Python

The MLP is used in three places — MSM initialisation, training sampling, and testing sampling — so all three must be updated:

```python
from presto.settings import (
    MLMDSamplingSettings,
    MLPSettings,
    MSMSettings,
    ParamSettings,
    WorkflowSettings,
)
from presto.workflow import get_bespoke_force_field

# Replace with your own ASE calculator instance
calculator = ...

ase_mlp = MLPSettings(
    ml_potential="ase",
    ml_system_kwargs={"calculator": calculator},
)

settings = WorkflowSettings(
    param_settings=ParamSettings(
        molecule_input_type="smiles",
        molecules="CCO",
        msm_settings=MSMSettings(mlp_settings=ase_mlp),
    ),
    device_type="cuda",
    training_sampling_settings=MLMDSamplingSettings(mlp_settings=ase_mlp),
    testing_sampling_settings=MLMDSamplingSettings(mlp_settings=ase_mlp),
)

bespoke_ff = get_bespoke_force_field(settings)
```

## Charge handling caveat

!!! warning "Charge handling"
    For non-ASE models, `presto` automatically passes molecular charge to
    `MLPotential.createSystem(...)`. For `ml_potential="ase"`, charge is **not**
    automatically propagated. Supply it explicitly in `mlp_settings.ml_system_kwargs`
    (for example under `info`), or use preconfigured `aseAtoms`.

## Round-tripping settings via YAML

ASE calculators are Python objects and don't serialise to YAML. When `presto` writes a settings YAML, it replaces non-serializable values under `ml_system_kwargs` with a placeholder string. If you try to reload such a YAML without supplying the calculator, validation fails with:

```
InvalidSettingsError: ml_system_kwargs contains runtime-only placeholder values at keys:
['[calculator]']. Supply the actual objects via from_yaml(..., overwrite=...) before validation.
```

Inject the calculator on load:

```python
ase_kwargs = {"ml_system_kwargs": {"calculator": calculator}}

loaded = WorkflowSettings.from_yaml(
    "workflow_settings.yaml",
    overwrite={
        "param_settings": {
            "msm_settings": {"mlp_settings": ase_kwargs}
        },
        "training_sampling_settings": {"mlp_settings": ase_kwargs},
        "testing_sampling_settings": {"mlp_settings": ase_kwargs},
    },
)
```
