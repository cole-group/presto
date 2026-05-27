# Wipe, Re-run, and Re-analyse

How to wipe output and rerun.

## Clean a directory

`presto clean` removes everything `presto train` / `train-from-yaml` would generate, but **keeps the settings YAML**:

```bash
presto clean workflow_settings.yaml
```

It walks the expected output layout (see **[Concepts → Output directory layout](../concepts/output-layout.md)**) and deletes each known file/directory plus the empty stage directories. Anything not in the expected layout is left alone — useful if you've added your own notes or scripts to the run directory.

## Re-run from a YAML

Any `workflow_settings.yaml` written by a previous run can be replayed with:

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
