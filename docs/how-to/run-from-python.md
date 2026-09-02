# Run from Python

The CLI is an easy way to run `presto` for one-off fits, but the Python API gives you better control when you're sweeping settings, doing batch fits, or composing `presto` into a larger pipeline. The end-to-end notebook is at **[Walk-through (Python API)](../examples/basic-walk-through-python-api.ipynb)**; this page is the short prose reference.

## Build a `WorkflowSettings` object

```python
from presto.settings import ParamSettings, WorkflowSettings
from presto.workflow import get_bespoke_force_field

def main():
    settings = WorkflowSettings(
        param_settings=ParamSettings(
            molecule_input_type="smiles",
            molecules="CCO",
        ),
        device_type="cuda",
    )

    return get_bespoke_force_field(settings)


if __name__ == "__main__":
    bespoke_ff = main()
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
# Parallel ligand sampling

Independent ligands can be sampled concurrently on one node by setting
`n_sampling_processes` on `WorkflowSettings` (or passing
`--n-sampling-processes` to `presto train`). The default is one process and keeps
the serial behaviour.

For CUDA runs, Presto assigns workers round-robin to the logical devices exposed
by `CUDA_VISIBLE_DEVICES`. For example, `CUDA_VISIBLE_DEVICES=0,2` exposes two
logical devices to the workers. If only one device is visible, every worker uses
it; NVIDIA MPS must be started and configured outside Presto if concurrent GPU
execution is desired.

Each process loads its own force field and ML model, so model and CUDA memory use
is multiplied by the number of workers. More processes can be slower when one
process already saturates a GPU. This option parallelizes ligands within a single
node only; parameterization, fitting, and analysis remain in the parent process.

!!! warning "Guard the Python entry point"
    Parallel sampling starts fresh Python processes. When calling
    `get_bespoke_force_field` from a Python script with
    `n_sampling_processes > 1`, put the call behind an
    `if __name__ == "__main__":` guard, as in the example above. Without the
    guard, every worker re-runs the script while it is starting and may create
    workers recursively or fail with a multiprocessing bootstrap error.

    Interactive Python sessions and notebooks do not provide a reliably
    importable guarded entry point. Use `n_sampling_processes=1` there, or run
    parallel sampling from a guarded `.py` script or the `presto` CLI.
