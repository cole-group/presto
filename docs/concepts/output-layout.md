# Output directory layout

Every `presto` run writes a fixed tree under `output_dir` (configured in `WorkflowSettings.output_dir`, default `.`). This page describes the structure; for what each diagnostic plot shows, see **[How-to → Inspect outputs and plots](../how-to/inspect-outputs.md)**.

## Top-level directory

```
<output_dir>/
├── workflow_settings.yaml      # snapshot of the settings used for the run
├── initial_statistics/         # offxml + scatter plots for the starting force field (after MSM)
├── test_data/                  # validation dataset (energies, forces, trajectories)
├── training_iteration_1/       # first (sample, train) iteration
├── training_iteration_2/       # second iteration (if n_iterations >= 2; the default)
└── plots/                      # aggregated diagnostic plots across iterations
```

`n_iterations` (default 2) controls how many `training_iteration_<n>/` directories are produced.

## `initial_statistics/`

Outputs from the untrained force field, after the modified Seminario method has set bond/angle parameters but before any energy/force fitting. Useful as a baseline to compare against trained iterations.

- `bespoke_ff.offxml` — the SMIRNOFF force field with bespoke types but otherwise initial parameters.
- `energies_and_forces_mol<n>.hdf5` — per-molecule scatter data.

## `test_data/`

Validation dataset that drives the diagnostic plots in `plots/`. Held fixed across iterations.

- `energy_and_force_data_mol<n>/` — per-molecule energies/forces (HuggingFace datasets format).
- `trajectory_mol<n>.pdb` — sampled snapshots (only for non-`pre_computed` protocols).

## `training_iteration_<n>/`

One per iteration. Contains the bespoke force field produced at the end of the iteration plus the data and metrics used to produce it.

- `bespoke_ff.offxml` — the bespoke SMIRNOFF force field after this iteration.
- `energy_and_force_data_mol<n>/` — per-molecule training set used in this iteration.
- `energies_and_forces_mol<n>.hdf5` — scatter data on `test_data` evaluated with this iteration's parameters.
- `trajectory_mol<n>.pdb` — per-molecule training trajectories from this iteration's MD sampling.
- `metadynamics_bias_mol<n>/` — accumulated metadynamics bias (only for metadynamics sampling protocols).
- `ml_minimised_mol<n>.pdb`, `mm_minimised_mol<n>.pdb` — additional structures (only for the `mm_md_metadynamics_torsion_minimisation` protocol).
- `metrics.txt` — training metrics (loss components per epoch).
- `tensorboard/` — TensorBoard event files.

## `plots/`

Aggregated diagnostic plots across all iterations.

- `loss.png` — training and validation loss per epoch.
- `error_distributions_mol<n>.png` — energy/force error histograms on the test set.
- `correlation_mol<n>.png` — predicted vs reference energies and forces (test set).
- `force_error_by_atom_index_mol<n>.png` — force error broken down by atom.
- `parameter_values_mol<n>.png` — fitted parameter values vs the starting force field.
- `parameter_differences_mol<n>.png` — change in each parameter from the starting force field.
- `handler_attributes.png` — handler-level attributes (e.g. double-exponential `alpha`/`beta`) across iterations. These are global to the force field, so there is one plot rather than one per molecule.
- `torsion_sampling_mol<n>.png` — dihedral coverage during training trajectories.

## Per-molecule outputs

Outputs marked `_mol<n>` are produced once per input molecule. For a single-molecule fit, `<n>` is always `0`. For congeneric series, you get one file per molecule, indexed by position in `param_settings.molecules`.

The full set of per-molecule output types is defined in `presto.outputs.PER_MOLECULE_OUTPUT_TYPES`.

## Workflow settings snapshot

`workflow_settings.yaml` is written by `get_bespoke_force_field` when called via `presto train` or via the Python API with `write_settings=True`. It records the exact settings used for the run, including any defaults that were filled in automatically. You can re-run the same fit with `presto train-from-yaml workflow_settings.yaml`.

See `OutputType` and `WorkflowPathManager` in [`presto.outputs`](../reference/api/outputs.md) for the authoritative definitions.
