# Use a pre-computed dataset

Instead of running MD sampling inside `presto`, you can supply a dataset of energies and forces on disk and train against it directly. This is the `pre_computed` sampling protocol.

## When to use this

- You have an existing reference dataset (e.g. from QM or a different MLP) you want to fit against.
- You want to repeat a fit with different training settings without re-sampling.
- You're integrating `presto` into a pipeline that produces datasets in a different stage.

## Dataset format

The dataset must be in `descent.train`-compatible format, which is HuggingFace `datasets.save_to_disk` output. Each row is an entry: a set of conformers (with their coordinates, energies and forces) that share an energy offset. A row with a single conformer therefore carries no energy information, and each row gets an equal share of the molecule's loss (see **[The loss function](../concepts/loss-function.md)**).

Each row has these columns, where `n_confs` is the number of conformers in the row and `n_atoms` the number of atoms in the molecule:

| Column | Units | Shape | Contents |
|---|---|---|---|
| `smiles` | | | Mapped SMILES of the molecule. Its atom order is the order of the atoms in `coords` and `forces`. |
| `coords` | Å | `n_confs * n_atoms * 3`, flattened from `(n_confs, n_atoms, 3)` | Coordinates of each conformer. |
| `energy` | kcal/mol | `n_confs` | Reference energy of each conformer. Only differences within the row matter. |
| `forces` | kcal/mol/Å | `n_confs * n_atoms * 3`, flattened like `coords` | Reference forces on each atom. |
| `energy_weights` | | `n_confs` | Optional. The absolute scale of each conformer's energy error (1000 if absent, the default `loss_energy_weight`). |
| `forces_weights` | | `n_confs` | Optional. The absolute scale of each conformer's force error (0.1 if absent, the default `loss_force_weight`). |
| `conformer_weights` | | `n_confs` | Optional. Relative weights within the row (1 if absent). Use a weight of 0, not an absolute weight of 0, to exclude a conformer. |

The datasets do not record their units, so convert your data to these before saving. NaN energies or forces are masked out of the loss automatically. Datasets that `presto` loads, including the one passed to a [conformer weights function](weight-conformers-and-molecules.md#weight-conformers-with-your-own-function), have the same columns in torch format, with `coords` and `forces` still flattened.

The simplest way to produce one is to run `presto` once with a sampling protocol of your choice, then re-use the saved dataset directory from a previous run, for example `training_iteration_1/train_dataset_0/energy_and_force_data_mol0`.

## Set `sampling_protocol: pre_computed`

In YAML:

```yaml
training_sampling_settings:
-   sampling_protocol: pre_computed
    dataset_paths:
        - path/to/training_dataset
```

Programmatically:

```python
from pathlib import Path
from presto.settings import PreComputedDatasetSettings, WorkflowSettings

settings = WorkflowSettings(
    param_settings=...,
    training_sampling_settings=[
        PreComputedDatasetSettings(dataset_paths=[Path("path/to/training_dataset")]),
    ],
)
```

You can use `pre_computed` for any item of `training_sampling_settings`, `testing_sampling_settings`, or both, and combine it with sampled datasets, each weighted by its `dataset_weight` (see **[Multiple datasets](../concepts/sampling-protocols.md#multiple-datasets)**). For example, to fit to MD samples and an external QM dataset together:

```yaml
training_sampling_settings:
-   sampling_protocol: mm_md_metadynamics_torsion_minimisation
-   sampling_protocol: pre_computed
    dataset_paths:
        - path/to/qm_dataset
    dataset_weight: 2.0
```

## Multi-molecule fits

For congeneric series, `dataset_paths` is a list — one path per molecule, in the same order as `param_settings.molecules`:

```yaml
param_settings:
    molecules:
        - CCO
        - CCC
training_sampling_settings:
-   sampling_protocol: pre_computed
    dataset_paths:
        - cco_data.hf
        - ccc_data.hf
```

If the number of `dataset_paths` does not match the number of molecules, `presto` raises an error during validation.

## Caveats

- **No trajectory output.** `pre_computed` does not produce `trajectory_mol<n>.pdb` since there is no MD. The corresponding `output_types` set is empty.
- **Outlier filtering still applies.** Pre-computed training data are loaded again each iteration and filtered against the current force field, like sampled data.
- **No memory mode.** `memory: true` cannot be used with a `pre_computed` training protocol, because the same pre-computed data would be appended again every iteration, and settings validation raises an error.

## API reference

[`PreComputedDatasetSettings`](../reference/api/settings.md#presto.settings.PreComputedDatasetSettings).
