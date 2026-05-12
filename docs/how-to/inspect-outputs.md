# Inspect outputs and plots

After a run, your bespoke force field is at `<output_dir>/training_iteration_<n>/bespoke_ff.offxml`, where `n` is the final iteration. The diagnostic plots under `<output_dir>/plots/` tell you whether to trust it. This guide walks through what each plot should look like and where to look first when something is off.

For the underlying directory tree, see **[Concepts → Output directory layout](../concepts/output-layout.md)**.

## What each plot tells you

### `plots/loss.png`

Training and validation loss per epoch.

- **Good**: both curves drop and flatten by the end of training. Validation loss is comparable to training loss.
- **Bad**: validation loss diverges from training loss (overfitting) or never flattens (under-trained — raise `n_epochs`, or check learning rate).

### `plots/error_distributions_mol<n>.png`

Distribution of per-snapshot energy and force errors on the test set.

- **Good**: roughly Gaussian, mean near zero. Energy error spread under ~1 kcal/mol.
- **Bad**: heavy tails (outlier conformations dominating) or a non-zero mean (a systematic offset has not been removed by the per-snapshot mean shift).

### `plots/correlation_mol<n>.png`

Predicted vs reference energies and forces on the test set.

- **Good**: tight scatter around the diagonal.
- **Bad**: bowing (systematic curvature suggests a missing functional form, e.g. torsion periodicity).

### `plots/force_error_by_atom_index_mol<n>.png`

Force errors broken down by atom index in the molecule.

- **Use**: if one or two atoms dominate the force error, look at their valence environment — likely a torsion or angle that isn't being captured.

### `plots/parameter_values_mol<n>.png` and `parameter_differences_mol<n>.png`

Fitted parameter values, and the change from the starting force field. The "initial" curve corresponds to the force field after the MSM step (not the raw OpenFF input).

- **Use**: look for individual parameters that have moved unreasonably far from their starting value. The regularisation penalty on torsion `k` should keep most torsions close to their starting point.

### `plots/torsion_sampling_mol<n>.png`

Dihedral angle coverage during training trajectories.

- **Good**: rotatable torsions visit most of the (-π, π) range. Aromatic and amide torsions stay localised.
- **Bad**: a key rotatable torsion is stuck in one well — bump `n_conformers` in the training sampling settings, or check that `torsions_to_include_smarts` includes it.

## The bespoke offxml file

`<output_dir>/training_iteration_<n>/bespoke_ff.offxml` is a standard SMIRNOFF `.offxml` file. The bespoke parameters are appended to the end of the input force field; OpenFF's SMIRKS-priority rule means they override the original (less specific) parameters wherever they match.

Use it like any other OpenFF force field:

```python
from openff.toolkit import ForceField, Molecule

ff = ForceField("training_iteration_2/bespoke_ff.offxml")
mol = Molecule.from_smiles("CCO")
system = ff.create_interchange(mol.to_topology()).to_openmm()
```

## The HDF5 energy/force data

Per-iteration training data is saved with HuggingFace `datasets.save_to_disk`. Reload with:

```python
from datasets import load_from_disk
ds = load_from_disk("training_iteration_2/energy_and_force_data_mol0")
```

Each row holds energies and forces for one snapshot, plus the coordinates.

## Where to look first when a fit looks off

| Symptom | Look at | Likely fix |
|---|---|---|
| Validation loss > training loss by a lot | `loss.png` | Raise `n_iterations`, lower `learning_rate`, or check for too-specific types |
| Bowed correlation plot | `correlation_mol<n>.png` | Check `expand_torsions` is on; check torsion sampling |
| Wild parameter changes | `parameter_differences_mol<n>.png` | Add regularisation to bonds/angles, or relax type specificity |
| Sparse torsion coverage | `torsion_sampling_mol<n>.png` | Bump `n_conformers`; verify metadynamics is enabled |
| One atom dominates force error | `force_error_by_atom_index_mol<n>.png` | Inspect valence environment; may need a different `max_extend_distance` |

For more failure modes and their fixes, see **[Reference → Troubleshooting](../reference/troubleshooting.md)**.
