# Wipe, Re-run, and Re-analyse

How to wipe output and rerun.

## Clean a directory

`presto clean` removes every Presto-owned generated stage directory, but **keeps the settings YAML**:

```bash
presto clean workflow_settings.yaml
```

The directories `initial_statistics/`, `test_data/`, `plots/`, and every
`training_iteration_*/` directory are reserved for Presto and are removed recursively.
Keep personal notes outside those directories. Unrelated files at the output root are
left alone.

The output manager reports a fit as:

- `clean` when none of the generated stage directories exists;
- `partial` when a generated stage exists without the final fitted force field; or
- `complete` when the final iteration's `bespoke_ff.offxml` exists.

Both partial and complete output must be cleaned before another fit can start. This
prevents an interrupted run from silently reusing an old trajectory or metadynamics
bias.

## Re-run from a YAML

After cleaning, any `workflow_settings.yaml` written by a previous run can be replayed with:

```bash
presto train-from-yaml workflow_settings.yaml
```

To make small changes before replaying, either edit the YAML directly or use the `overwrite` argument on `WorkflowSettings.from_yaml`:

```python
from presto.settings import WorkflowSettings
from presto.workflow import get_bespoke_force_field

settings = WorkflowSettings.from_yaml(
    "workflow_settings.yaml",
    overwrite={"n_iterations": 4, "training_settings": {"n_epochs": 2000}},
)
get_bespoke_force_field(settings, write_settings=False)
```
