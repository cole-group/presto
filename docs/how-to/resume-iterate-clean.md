# Resume, iterate, and clean a run

Operational mechanics for re-running, refining, or wiping a fit directory.

## Iterations and memory

`WorkflowSettings.n_iterations` controls how many (sample, train) iterations are performed. The default is `2`:

- **Iteration 1** samples MD using the initial force field (after the MSM step has set bond/angle parameters).
- **Iteration 2** samples MD using the force field produced by iteration 1, which usually gives more physically reasonable training data.

`WorkflowSettings.memory` controls how training data accumulates across iterations:

- `memory: false` (default) — each iteration replaces the previous training dataset. Peak GPU memory does not grow across iterations.
- `memory: true` — each iteration appends to the previous training dataset. Peak GPU memory grows roughly linearly with iteration count, but the optimiser sees a richer dataset.

Use `memory: true` if you want each iteration to refine rather than replace the fit. Use `memory: false` if you'd like the bespoke FF to forget about MD sampled on a less-converged FF.

## Re-run from a YAML

Any `workflow_settings.yaml` written by a previous run can be replayed with:

```bash
presto train-from-yaml workflow_settings.yaml
```

This re-runs the entire workflow, overwriting outputs. To make small changes before replaying, either edit the YAML directly or use the `overwrite` argument on `WorkflowSettings.from_yaml`:

```python
from presto.settings import WorkflowSettings
from presto.workflow import get_bespoke_force_field

settings = WorkflowSettings.from_yaml(
    "workflow_settings.yaml",
    overwrite={"n_iterations": 4, "training_settings": {"n_epochs": 2000}},
)
get_bespoke_force_field(settings, write_settings=False)
```

## Clean a directory

`presto clean` removes everything `presto train` / `train-from-yaml` would generate, but **keeps the settings YAML**:

```bash
presto clean workflow_settings.yaml
```

It walks the expected output layout (see **[Concepts → Output directory layout](../concepts/output-layout.md)**) and deletes each known file/directory plus the empty stage directories. Anything not in the expected layout is left alone — useful if you've added your own notes or scripts to the run directory.

## Analyse after the fact

`presto analyse` regenerates the diagnostic plots under `<output_dir>/plots/` from existing training output, without re-running sampling or training:

```bash
presto analyse workflow_settings.yaml
```

Useful when:

- You want to refresh the plots after manually editing the trained `bespoke_ff.offxml`.
- You ran with an older `presto` and want the new analysis plots applied to old data.
- A plot generation step crashed at the end of a long run and you don't want to re-run the whole thing.
