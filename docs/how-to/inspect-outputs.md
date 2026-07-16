# Inspect outputs and plots

After a run, your bespoke force field is at `<output_dir>/training_iteration_<n>/bespoke_ff.offxml`, where `n` is the final iteration. The diagnostic plots under `<output_dir>/plots/` tell you whether to trust it. This guide walks through what each plot should look like and where to look first when something is off.

For the underlying directory tree, see **[Concepts → Output directory layout](../concepts/output-layout.md)**.

## What each plot tells you

### `plots/loss.png`

Training and validation loss per epoch. Note that by default, the training (sampled with MM MD) and validation (sampled with MLP MD) sets are generated in a different way, so e.g. training loss may be far above validation loss due to the introduction of high-energy samples through the metadynamics.

- **Good**: both curves drop and flatten by the end of training. Validation loss is comparable to training loss.
- **Bad**: dramatic rise in validation loss -- often due to changes in connectivity during MLP minimisatinos (e.g. from proton hops). Switch to a sampling protocol without MLP minimisations.

### `plots/error_distributions_mol<n>.png`

Distribution of per-snapshot energy and force errors on the test set.

### `plots/correlation_mol<n>.png`

Predicted vs reference energies and forces on the test set.

- **Good**: tight scatter around the diagonal.
- **Bad**: large outliers. Often caused by non-bonded clashes incorrectly predicted by the MM force field for MLP configurations. There's no easy way to address these (if they are beyond 1-4 interactions) without modifing the non-bonded terms.

### `plots/force_error_by_atom_index_mol<n>.png`

Force errors broken down by atom index in the molecule.

- **Use**: if one or two atoms dominate the force error, look at their valence environment — often pairs of atoms which closely approach have large force errors due to overly repulsive MM non-bonded interactions.

### `plots/parameter_values_mol<n>.png` and `parameter_differences_mol<n>.png`

Fitted parameter values, and the change from the starting force field. The "initial" curve corresponds to the force field after the MSM step (not the raw OpenFF input).

- **Use**: look for individual parameters that have moved unreasonably far from their starting value. The regularisation penalty on torsion `k` biases torsions towards to their starting point, but is fairly weak by default.

### `plots/torsion_sampling_mol<n>.png`

Dihedral angle coverage during training trajectories.

- **Good**: rotatable torsions visit most of the (-π, π) range.
- **Bad**: a key rotatable torsion is stuck in one well — sample for longer or make the metadynamics more aggressive.

## The bespoke offxml file

`<output_dir>/training_iteration_<n>/bespoke_ff.offxml` is a standard SMIRNOFF `.offxml` file. The bespoke parameters are appended to the end of the input force field and all have `bespoke` in their IDs, e.g. `id=p-bespoke-533` for proper torsion 533; as they're placed lower down than the original (more generic) parameters, they override the original (less specific) parameters wherever they match.

Use it like any other OpenFF force field:

```python
from openff.toolkit import ForceField, Molecule

ff = ForceField("training_iteration_2/bespoke_ff.offxml")
mol = Molecule.from_smiles("CCO")
# combine_nonbonded_forces=False is only required for double-exponential (dexp)
# force fields, whose vdW form has no combined NonbondedForce equivalent.
system = ff.create_interchange(mol.to_topology()).to_openmm(
    combine_nonbonded_forces=False
)
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
| Dramatic rise in validation loss | `loss.png` | Likely connectivity changes/ poor equilibrium value MSM initialisation. Try disabling MLP minimisations by switching to mm_md_metadynamics sampling protocol and disabling MSM initialisation (`param_settings.msm_settings: null`). If this doesn't help, possibly add stronger regularisation. |
| Large outliers in correlation plot | `correlation_mol<n>.png` | Often non-bonded clashes in MLP configurations; no easy fix without modifying non-bonded terms |
| Wild parameter changes | `parameter_differences_mol<n>.png` | May also be caused by connectivity changes/ poor MSM initialisation. Try disabling MLP minimisations by switching to `mm_md_metadynamics` sampling protocol and disabling MSM initialisation. If this doesn't help, possibly add stronger regularisation. |
| Sparse torsion coverage | `torsion_sampling_mol<n>.png` | Make metadynamics more aggressive, increase sampling time |
| A few pairs of atoms dominate force error | `force_error_by_atom_index_mol<n>.png` | Likely close atom contacts with overly repulsive MM non-bonded interactions, no easy fix without training nonbonded parameters |

When any of these look off, inspect the sample PDB files e.g. (`training_iteration_<n>/trajectory_mol<n>.pdb`) to see the configurations that went into training — this often helps reveal the root cause (e.g. connectivity changes, steric clashes, or poor conformer diversity).

For more failure modes and their fixes, see **[Reference → Troubleshooting](../reference/troubleshooting.md)**.
