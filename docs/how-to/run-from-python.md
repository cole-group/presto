# Run from Python

The CLI is an easy way to run `presto` for one-off fits, but the Python API gives you better control when you're sweeping settings, doing batch fits, or composing `presto` into a larger pipeline. The end-to-end notebook is at **[Walk-through (Python API)](../examples/basic-walk-through-python-api.ipynb)**; this page is the short prose reference.

## Build a `WorkflowSettings` object

```python
from presto.settings import ParameterisationSettings, WorkflowSettings
from presto.workflow import get_bespoke_force_field

settings = WorkflowSettings(
    parameterisation_settings=ParameterisationSettings(
        molecule_input_type="smiles",
        molecules="CCO",
    ),
    device_type="cuda",
)

bespoke_ff = get_bespoke_force_field(settings)
```

`get_bespoke_force_field` returns the final fitted `openff.toolkit.ForceField` and writes the same output tree the CLI would. Pass `write_settings=False` to skip writing `workflow_settings.yaml` (useful when you've loaded settings from a YAML file already).

## Override individual fields after load

`WorkflowSettings.from_yaml` accepts an `overwrite` dict that is deep-merged into the YAML before validation:

```python
settings = WorkflowSettings.from_yaml(
    "workflow_settings.yaml",
    overwrite={"n_iterations": 1, "device_type": "cuda"},
)
```

This is the recommended way to inject runtime objects (e.g. an ASE calculator) that can't round-trip through YAML — see **[Use an ASE calculator](use-ase-calculator.md)**.

## Reference

- API reference: [`WorkflowSettings`](../reference/api/settings.md#presto.settings.WorkflowSettings), [`get_bespoke_force_field`](../reference/api/workflow.md#presto.workflow.get_bespoke_force_field).
