# Use a pre-computed dataset

Instead of running MD sampling inside `presto`, you can supply a dataset of energies and forces on disk and train against it directly. This is the `pre_computed` sampling protocol.

## When to use this

- You have an existing reference dataset (e.g. from QM or a different MLP) you want to fit against.
- You want to repeat a fit with different training settings without re-sampling.
- You're integrating `presto` into a pipeline that produces datasets in a different stage.

## Dataset format

The dataset must be in `descent.train`-compatible format, which is HuggingFace `datasets.save_to_disk` output. Each row holds energies, forces, and coordinates for one snapshot.

The simplest way to produce one is to run `presto` once with a sampling protocol of your choice, then re-use the saved dataset directory from a previous run — for example, `training_iteration_1/energy_and_force_data_mol0`.

## Set `sampling_protocol: pre_computed`

In YAML:

```yaml
training_sampling_settings:
    sampling_protocol: pre_computed
    dataset_paths:
        - path/to/training_dataset
```

Programmatically:

```python
from pathlib import Path
from presto.settings import PreComputedDatasetSettings, WorkflowSettings

settings = WorkflowSettings(
    parameterisation_settings=...,
    training_sampling_settings=PreComputedDatasetSettings(
        dataset_paths=[Path("path/to/training_dataset")],
    ),
)
```

You can use `pre_computed` for `training_sampling_settings`, `testing_sampling_settings`, or both.

## Multi-molecule fits

For congeneric series, `dataset_paths` is a list — one path per molecule, in the same order as `parameterisation_settings.molecules`:

```yaml
parameterisation_settings:
    molecules:
        - CCO
        - CCC
training_sampling_settings:
    sampling_protocol: pre_computed
    dataset_paths:
        - cco_data.hf
        - ccc_data.hf
```

If `molecules` has more entries than `dataset_paths`, `presto` will raise during validation.

## Caveats

- **No trajectory output.** `pre_computed` does not produce `trajectory_mol<n>.pdb` since there is no MD. The corresponding `output_types` set is empty.
- **No outlier filtering by MM comparison.** Outlier filtering still applies but its MM-energy reference is computed on the dataset coordinates, not freshly sampled coordinates.
- **Memory mode behaviour.** With `memory: true`, datasets from earlier iterations are concatenated with the pre-computed dataset for later iterations. Disable `memory` if you want each iteration to train on exactly the supplied dataset.

## API reference

[`PreComputedDatasetSettings`](../reference/api/settings.md#presto.settings.PreComputedDatasetSettings).
