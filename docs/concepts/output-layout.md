# Output directory layout

Every `presto` run writes a fixed tree under `output_dir` (configured in `WorkflowSettings.output_dir`, default `.`). This page describes the structure; for what each diagnostic plot shows, see **[How-to → Inspect outputs and plots](../how-to/inspect-outputs.md)**.

## Top-level directory

```
<output_dir>/
├── workflow_settings.yaml      # snapshot of the settings used for the run
├── initial_statistics/         # offxml + scatter data for the starting force field (after MSM)
├── test_data/                  # validation datasets (energies, forces, trajectories)
├── training_iteration_1/       # first (sample, train) iteration
├── training_iteration_2/       # second iteration (if n_iterations >= 2; the default)
└── plots/                      # aggregated diagnostic plots across iterations
```

`n_iterations` (default 2) controls how many `training_iteration_<n>/` directories are produced.

## Dataset subdirectories

`training_sampling_settings` and `testing_sampling_settings` are lists of sampling protocols, one per dataset (see **[Sampling protocols → Multiple datasets](sampling-protocols.md#multiple-datasets)**). Outputs that belong to one dataset sit in a subdirectory of their stage named after the dataset's position in its list:

- `train_dataset_<k>/` for the `k`-th item of `training_sampling_settings`.
- `test_dataset_<k>/` for the `k`-th item of `testing_sampling_settings`.

These subdirectories are always used, even for a single dataset, so a default run has `train_dataset_0/` and `test_dataset_0/` only.

```
<output_dir>/
├── initial_statistics/
│   ├── bespoke_ff.offxml
│   └── test_dataset_0/energies_and_forces_mol0.hdf5
├── test_data/
│   └── test_dataset_0/{energy_and_force_data_mol0/, trajectory_mol0.pdb}
├── training_iteration_1/
│   ├── bespoke_ff.offxml, metrics.txt, tensorboard/
│   ├── train_dataset_0/{energy_and_force_data_mol0/, trajectory_mol0.pdb, ...}
│   └── test_dataset_0/energies_and_forces_mol0.hdf5
└── plots/
    ├── loss.png, correlation_mol0.png, ...
    ├── train_dataset_0/{loss.png, torsion_sampling_mol0.png}
    └── test_dataset_0/{loss.png, correlation_mol0.png, ...}
```

## `initial_statistics/`

Outputs from the untrained force field, after the modified Seminario method has set bond/angle parameters but before any energy/force fitting. Useful as a baseline to compare against trained iterations.

- `bespoke_ff.offxml`: the SMIRNOFF force field with bespoke types but otherwise initial parameters.
- `test_dataset_<k>/energies_and_forces_mol<n>.hdf5`: per-molecule scatter data on test dataset `k`.

## `test_data/`

Validation datasets that drive the diagnostic plots in `plots/`. Held fixed across iterations. In each `test_dataset_<k>/`:

- `energy_and_force_data_mol<n>/`: per-molecule energies/forces (HuggingFace datasets format).
- `trajectory_mol<n>.pdb`: sampled snapshots (only for non-`pre_computed` protocols).

## `training_iteration_<n>/`

One per iteration. Contains the bespoke force field produced at the end of the iteration plus the data and metrics used to produce it.

- `bespoke_ff.offxml`: the bespoke SMIRNOFF force field after this iteration.
- `metrics.txt`: training metrics: a header row naming the columns, then one row per reporting step. The first six columns are the combined (weighted) training and test losses (`loss_train_energy`, ..., `loss_test_regularisation`), followed by the energy and force loss of each dataset (`loss_train_dataset_<k>_energy`, ..., `loss_test_dataset_<k>_forces`).
- `tensorboard/`: TensorBoard event files, with the same names as the `metrics.txt` columns.
- `train_dataset_<k>/`: the data sampled for training dataset `k`:
    - `energy_and_force_data_mol<n>/`: per-molecule training set used in this iteration.
    - `trajectory_mol<n>.pdb`: per-molecule training trajectories from this iteration's MD sampling.
    - `metadynamics_bias_mol<n>/`: accumulated metadynamics bias (only for metadynamics sampling protocols).
    - `ml_minimised_mol<n>.pdb`, `mm_minimised_mol<n>.pdb`: additional structures (only for the torsion minimisation protocols).
- `test_dataset_<k>/energies_and_forces_mol<n>.hdf5`: scatter data on test dataset `k` evaluated with this iteration's parameters.

## `plots/`

Aggregated diagnostic plots across all iterations. The plots directly in `plots/` combine every dataset; the subdirectories hold the same plots for each dataset.

- `loss.png`: combined training and validation loss per epoch.
- `error_distributions_mol<n>.png`: energy/force error histograms on the test set.
- `correlation_mol<n>.png`: predicted vs reference energies and forces (test set).
- `force_error_by_atom_index_mol<n>.png`: force error broken down by atom.
- `parameter_values_mol<n>.png`: fitted parameter values vs the starting force field.
- `parameter_differences_mol<n>.png`: change in each parameter from the starting force field.
- `torsion_sampling_mol<n>.png`: dihedral coverage during the test (iteration 0) and training trajectories.
- `test_dataset_<k>/`: `loss.png` (that dataset's test loss), the error, correlation and force-by-atom plots on that dataset, and its `torsion_sampling_mol<n>.png`.
- `train_dataset_<k>/`: `loss.png` (that dataset's training loss) and `torsion_sampling_mol<n>.png` for its trajectories.

The combined error and correlation plots pool the conformers of every test dataset, so larger datasets contribute more points, regardless of `dataset_weight`. Energies in each dataset are relative to that dataset's own mean. Torsion sampling plots are only produced for protocols that write trajectories (not `pre_computed`).

## Per-molecule outputs

Outputs marked `_mol<n>` are produced once per input molecule. For a single-molecule fit, `<n>` is always `0`. For congeneric series, you get one file per molecule, indexed by position in `param_settings.molecules`.

The full set of per-molecule output types is defined in `presto.outputs.PER_MOLECULE_OUTPUT_TYPES`.

## Workflow settings snapshot

`workflow_settings.yaml` is written by `get_bespoke_force_field` when called via `presto train` or via the Python API with `write_settings=True`. It records the exact settings used for the run, including any defaults that were filled in automatically. You can re-run the same fit with `presto train-from-yaml workflow_settings.yaml`.

See `OutputType` and `WorkflowPathManager` in [`presto.outputs`](../reference/api/outputs.md) for the authoritative definitions.
