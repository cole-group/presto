# Fit a single molecule

For a single molecule, the defaults are tuned for a sensible time-to-accuracy trade-off on a ~50-atom molecule. This page tells you which ones to keep and which to tweak.

## Defaults you should keep

- **Sampling protocol**: `mm_md_metadynamics_torsion_minimisation`. Combines high-temperature MD with well-tempered metadynamics on rotatable bonds, plus short MLP/MM minimisations. The minimisations clean up steric clashes that the MM force field would otherwise generate during sampling. See **[Concepts → Sampling protocols](../concepts/sampling-protocols.md)**.
- **Reference MLP**: `aimnet2`. Fast and robust; the historical default `aceff-2.0` is [currently broken upstream](https://github.com/openmm/openmm-ml/issues/137).
- **Initial force field**: `openff_unconstrained-2.3.0.offxml`. Modern OpenFF baseline with reliable nonbonded parameters.
- **`expand_torsions: true`** and **`linearise_harmonics: true`** — both stabilise fitting.

## Defaults you may want to change

- **`n_iterations`** (default `2`). Each iteration retrains using MD sampled with the previous iteration's force field. Iteration 2 typically improves test loss over iteration 1. Set to `1` for fast iteration; raise above 2 if loss is still trending down.
- **`training_settings.n_epochs`** (default `1000`). Drop to ~200 for quick checks; raise if `loss.png` has not flattened by the end.
- **`training_sampling_settings.n_conformers`** (default `10`). More conformers means more diverse training data — useful for flexible molecules with many rotatable bonds.
- **`training_sampling_settings.production_sampling_time_per_conformer`** (default `100 ps`). Lengthen if torsion coverage in `plots/torsion_sampling_mol0.png` is sparse.

## Run it (CLI)

```bash
presto train --parameterisation-settings.molecules "CCC(CC)C(=O)Nc2cc(NC(=O)c1c(Cl)cccc1Cl)ccn2"
```

Override individual fields with dotted flags:

```bash
presto train \
    --parameterisation-settings.molecules "CCO" \
    --n-iterations 1 \
    --training-settings.n-epochs 200
```

## Run it (YAML)

```bash
presto write-default-yaml workflow_settings.yaml
# edit parameterisation_settings.molecules, optionally tune the fields above
presto train-from-yaml workflow_settings.yaml
```

See **[Settings reference](../reference/settings-reference.md)** for the YAML shape.

## After the fit

Check the diagnostic plots in `plots/`. The first things to look at are `loss.png`, `correlation_mol0.png`, and `parameter_differences_mol0.png` — see **[Inspect outputs and plots](inspect-outputs.md)**.
