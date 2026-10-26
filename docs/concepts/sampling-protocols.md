# Sampling protocols

`presto` generates training and validation data by running MD on each input molecule. `training_sampling_settings` and `testing_sampling_settings` are each a list of sampling protocols, one per dataset, and the `sampling_protocol` field of each item chooses which protocol generates that dataset. Each protocol is implemented as a subclass of `_SamplingSettingsBase`. To combine several protocols, see **[Multiple datasets](#multiple-datasets)**.

## Why sampling matters in presto

The bespoke parameters are fitted to reproduce MLP energies and forces on the sampled snapshots. The fit can only be as good as the configuration distribution it's trained on:

- **Too narrow** (e.g. all snapshots in the same torsion well) and the fitted parameters won't generalise to other regions of conformational space.
- **Too physically unreasonable** (e.g. highly strained configurations sampled at very high temperature) and high energy snapshots irrelevant to equilibrium sampling at room temperature may dominate the loss.

The default protocol (`mm_md_metadynamics_torsion_minimisation`) is designed to give broad torsional coverage while supplementing high energy samples with samples from brief minimisations (with both the MM force field and MLP).

## `mm_md`

Plain MD using the current MM force field. Fast and reliable but conformer coverage is limited. Use when:

- You want the cheapest possible sampling.

## `ml_md`

MD using the MLP itself as the propagator. Energies and forces are evaluated by the MLP at every timestep. An accurate MLP will produce  more physically realistic configurations than `mm_md`, especially for flexible or strained molecules, but this will be slow. Note that training only on MLP configurations may not be optimal, because you may be missing regions the MM force field would incorrectly sample and which should be penalised. Use when:

- You can afford the MLP evaluation budget and want sampling which is unaffected by your initial MM force field.

This is also the default for `testing_sampling_settings`, where sampling cost is minimised by using relatively few snapshots than training.

## `mm_md_metadynamics`

MM-driven MD with [well-tempered metadynamics](https://doi.org/10.1103/PhysRevLett.100.020603) on rotatable bonds. The metadynamics bias is updated on the fly to push the system out of conformational wells, giving much better torsional coverage than plain `mm_md` at modest extra cost.

Options, all nested under `metadynamics_settings`:

- `bias_height`, `bias_frequency`, `bias_factor`, `bias_width` — control the metadynamics bias.
- `torsion_selection_settings.torsions_to_include_smarts`, `torsion_selection_settings.torsions_to_exclude_smarts` — which torsions are biased (default: all rotatable bonds, with linear torsions excluded).

!!! note "Changing metadynamics settings in Python"

    `metadynamics_settings` is read-only once created, because the bias frequencies are checked against `timestep` and `production_sampling_time_per_conformer` only when the whole object is assigned. Setting a field directly (`settings.metadynamics_settings.bias_factor = 15.0`) raises an error. Replace the object instead:

    ```python
    settings.metadynamics_settings = settings.metadynamics_settings.model_copy(
        update={"bias_factor": 15.0}
    )
    ```

    In YAML, set the values under `metadynamics_settings:` as usual.

## `mm_md_metadynamics_torsion_minimisation` (default for training)

`mm_md_metadynamics` plus short, (optionally torsion-restrained) minimisations at the end of each conformer's trajectory. The minimisations use both the MLP and the MM force field as relaxation potentials. Each minimised snapshot is added to the training set with configurable loss weights. These, epecially the MLP minimisations, improve torsion scan performance. Note that by default, no torsion restraints are applied.

Additional options (alongside `metadynamics_settings`), all nested under `torsion_minimisation_settings`:

- `ml_minimisation_steps`, `mm_minimisation_steps` — how many minimisation iterations.
- `torsion_restraint_force_constant` — strength of the torsion restraint during minimisation.
- `torsion_selection_settings` — which torsions are restrained during minimisation. This is set separately from the torsions biased during MD.
- `loss_*_weight_*_torsion_min` — separate loss weights for the minimised snapshots, in case you want them weighted differently from the MD snapshots. The MD, MLP-minimised and MM-minimised snapshots are separate entries that count equally, so by default each carries a third of each molecule's loss. These weights set the balance between them directly (the loss is linear in them, whatever number of conformers outlier filtering leaves in each), and 0 drops those snapshots from that term. See **[The loss function](loss-function.md)**.

## `mm_md_torsion_restrained_torsion_minimisation`

The same pipeline as `mm_md_metadynamics_torsion_minimisation`, but with no metadynamics bias. Instead, each specified torsion (by default, rotatable torsions) is restrained to the value it takes in the conformer that trajectory started from, so sampling stays close to the conformers you supply rather than exploring away from them. The MLP and MM minimisation stage is unchanged.

Use this when you have already chosen the conformers you care about (e.g. a bound pose, or a set of conformers from a docking or QM study), supply them with `starting_conformers`, and want the fit weighted towards that region of conformational space rather than towards broad torsional coverage. Without `starting_conformers` the restraints hold the molecule near ETKDG-generated conformers instead, which is rarely what you want, so the protocol warns.

Additional options (on top of the minimisation options above):

- `md_torsion_restraint_force_constant` — how tightly sampling is held to the starting torsions. For a harmonic restraint the spread about the target is roughly `sqrt(RT/k)`, so the default (100 kJ/mol/rad²) allows around 12° at the default 500 K. Lower it to let the torsions roam further, raise it to pin them.
- `torsion_selection_settings` — which torsions are restrained during MD. This is set separately from `torsion_minimisation_settings.torsion_selection_settings`.
- Note this is separate from `torsion_minimisation_settings.torsion_restraint_force_constant`, which applies only to the minimisation stage and still defaults to 0.

## `pre_computed`

Skip MD entirely and load a saved dataset from disk. Useful when:

- You have an existing reference dataset (e.g. from QM or a different MLP) you want to fit against.
- You want to repeat a fit with different training settings without re-sampling.

The dataset must be in `descent.train`-compatible format (HuggingFace `datasets.save_to_disk`), with the columns and units listed in **[Dataset format](../how-to/use-precomputed-dataset.md#dataset-format)**. For multi-molecule fits, the order of `dataset_paths` must match the order of `param_settings.molecules`.

See **[How-to → Use a pre-computed dataset](../how-to/use-precomputed-dataset.md)** for the recipe.

## Multiple datasets

You can give several protocols for training, testing, or both. Each one generates its own dataset, and datasets are generated one after another (each still parallel across molecules with `n_sampling_processes`). Training is then a single optimisation over all of the training data.

```yaml
training_sampling_settings:
-   sampling_protocol: mm_md_metadynamics_torsion_minimisation
    dataset_weight: 1.0
-   sampling_protocol: mm_md_torsion_restrained_torsion_minimisation
    starting_conformers: bound_pose.sdf
    dataset_weight: 3.0
testing_sampling_settings:
-   sampling_protocol: ml_md
    temperature: 298 K
-   sampling_protocol: pre_computed
    dataset_paths: [qm_dataset_mol0]
```

Each dataset's loss is computed separately, then combined using the `dataset_weight` of each protocol:

$$
L = \sum_k w_k L_k, \qquad w_k = \frac{\lambda_k}{\sum_j \lambda_j}
$$

where `λ_k` is `dataset_weight` and `L_k` is the energy (or force) loss of dataset `k`: the mean over molecules weighted by `molecule_weights`, as for a single dataset (see **[The loss function](loss-function.md)**). The regularisation loss is added once. Because each dataset is scored separately, its influence is set by its weight, not by how many conformers it contains, and the weights are relative: a single dataset gives exactly the same loss whatever its weight. The test loss combines the test datasets in the same way. Within a dataset, `loss_energy_weight` and `loss_force_weight` still balance the energy and force terms.

Each dataset gets its own `train_dataset_<k>/` or `test_dataset_<k>/` subdirectory, and the metrics and plots report both the combined loss and each dataset's loss (see **[Output layout](output-layout.md#dataset-subdirectories)**). With `memory: true`, each training dataset is appended to the same dataset from earlier iterations, and outlier filtering is applied to each training dataset separately. `memory` cannot be combined with a `pre_computed` training protocol, since the same data would be appended every iteration.

A single protocol written without the list (as in settings files from earlier versions) is read as a one-item list. In Python, pass a list of settings objects, e.g. `training_sampling_settings=[MMMDSamplingSettings(), MLMDSamplingSettings(dataset_weight=0.5)]`.

!!! note "The command line takes one protocol of each kind"

    `presto train` is intended for simple fits, so it takes exactly one training and one testing protocol, set with dotted options such as `--training-sampling-settings.sampling-protocol mm_md`. To fit to several datasets, write them in a YAML file and run `presto train-from-yaml`, or use the Python API.

## Choosing a protocol

| You want… | Use |
|---|---|
| Defaults for a small/medium molecule | `mm_md_metadynamics_torsion_minimisation` (the default) |
| Fastest possible iteration | `mm_md` |
| Most physically realistic sampling (but does not penalise spurious MM minima) | `ml_md` |
| Reproducibility / external dataset | `pre_computed` |
| Sampling to stay near conformers you supply | `mm_md_torsion_restrained_torsion_minimisation` |

For the per-field defaults, see [`SamplingSettings`](../reference/api/settings.md#presto.settings.SamplingSettings) in the API reference.
