# Sampling protocols

`presto` generates training and validation data by running MD on each input molecule. The `sampling_protocol` field on `training_sampling_settings` and `testing_sampling_settings` chooses exactly which sampling protocol is used. Each protocol is implemented as a subclass of `_SamplingSettingsBase`.

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

Options:

- `bias_height`, `bias_frequency`, `bias_factor`, `bias_width` — control the metadynamics bias.
- `torsions_to_include_smarts`, `torsions_to_exclude_smarts` — which torsions are biased (default: all rotatable bonds, with linear torsions excluded).

## `mm_md_metadynamics_torsion_minimisation` (default for training)

`mm_md_metadynamics` plus short, (optionally torsion-restrained) minimisations at the end of each conformer's trajectory. The minimisations use both the MLP and the MM force field as relaxation potentials. Each minimised snapshot is added to the training set with configurable loss weights. These, epecially the MLP minimisations, improve torsion scan performance. Note that by default, no torsion restraints are applied.

Additional options (on top of the metadynamics base class):

- `ml_minimisation_steps`, `mm_minimisation_steps` — how many minimisation iterations.
- `torsion_restraint_force_constant` — strength of the torsion restraint during minimisation.
- `loss_*_weight_*_torsion_min` — separate loss weights for the minimised snapshots, in case you want them weighted differently from the MD snapshots.

## Implicit solvent (MM protocols only)

All of the MM protocols (`mm_md`, `mm_md_metadynamics` and `mm_md_metadynamics_torsion_minimisation`) sample in vacuum by default. Setting `implicit_solvent` switches on a generalised Born solvent model, which is useful when gas-phase sampling collapses onto conformers held together by intramolecular interactions that would be broken in solution.

The solvent term is applied **only to the MM system used to generate configurations** (the MD, the metadynamics bias, and the MM torsion minimisation). Everything else is unchanged and remains in vacuum:

- Reference energies and forces are always recalculated with the MLP in vacuum, so no solvated MM energy ever enters a dataset.
- The MM energies compared against the MLP (the training loss, the scatter plots and outlier filtering) are computed in vacuum.
- The MLP minimisations are performed in vacuum.

In other words, implicit solvent changes *which* configurations you fit to, not *what* you fit them to. The `ml_md` and `pre_computed` protocols do not accept this option.

```yaml
training_sampling_settings:
    sampling_protocol: mm_md_metadynamics_torsion_minimisation
    implicit_solvent:
        model: obc2
        solvent_dielectric: 78.5
        solute_dielectric: 1.0
        surface_area_model: ACE
        salt_concentration: 0.0 M
```

See **[How-to → Sample with implicit solvent](../how-to/use-implicit-solvent.md)** for the recipe.

## `pre_computed`

Skip MD entirely and load a saved dataset from disk. Useful when:

- You have an existing reference dataset (e.g. from QM or a different MLP) you want to fit against.
- You want to repeat a fit with different training settings without re-sampling.

The dataset must be in `descent.train`-compatible format (HuggingFace `datasets.save_to_disk`). For multi-molecule fits, the order of `dataset_paths` must match the order of `param_settings.molecules`.

See **[How-to → Use a pre-computed dataset](../how-to/use-precomputed-dataset.md)** for the recipe.

## Choosing a protocol

| You want… | Use |
|---|---|
| Defaults for a small/medium molecule | `mm_md_metadynamics_torsion_minimisation` (the default) |
| Fastest possible iteration | `mm_md` |
| Most physically realistic sampling | `ml_md` |
| Reproducibility / external dataset | `pre_computed` |

For the per-field defaults, see [`SamplingSettings`](../reference/api/settings.md#presto.settings.SamplingSettings) in the API reference.
