# Troubleshooting and FAQ

First-stop for known failure modes. If your symptom isn't listed here, check the [GitHub issue tracker](https://github.com/cole-group/presto/issues).

## CUDA / OpenMM 8.5 PythonForce mismatch

**Symptom**: `presto train` fails to start with a CUDA initialisation error, or crashes inside OpenMM.

**Cause**: Your NVIDIA driver doesn't support CUDA 12.9. OpenMM 8.5 requires CUDA 12.9 for the `PythonForce` class that `presto` uses to attach the MLP.

**Fix**: Update your NVIDIA driver. Verify with `nvidia-smi` that the driver supports CUDA 12.9+. See **[Installation → Prerequisites](../get-started/installation.md#prerequisites)**.

## AceFF 2.0 fails to import or produces nonsense energies

**Symptom**: Errors from openmm-ml when `ml_potential: aceff-2.0` is selected, or implausible energies in `correlation_*.png`.

**Cause**: [Open upstream issue in openmm-ml](https://github.com/openmm/openmm-ml/issues/137).

**Fix**: Switch to `aimnet2` (the current `presto` default). See **[How-to → Choose an MLP](../how-to/choose-an-mlp.md)**.

## MACE-OFF licence

**Symptom**: not an error, but a deployment block.

**Cause**: MACE-OFF is released under the [Academic Software License](https://github.com/gabor1/ASL/blob/main/ASL.md), which does not permit commercial use.

**Fix**: Use AIMNet2, Egret-1, AceFF-2.0 (when fixed), or Orb-v3 OMOL — all permit commercial use.

## ASE charge handling silently wrong

**Symptom**: Fits to charged species drift or never reach reasonable accuracy when `ml_potential="ase"`.

**Cause**: The ASE bridge in OpenMM-ML does not propagate molecular charge automatically. Other MLPs do — this is specific to ASE.

**Fix**: Pass charge explicitly via `mlp_settings.ml_system_kwargs`:

```python
MLPSettings(
    ml_potential="ase",
    ml_system_kwargs={"calculator": calc, "info": {"charge": -1}},
)
```

See **[How-to → Use an ASE calculator → Charge handling caveat](../how-to/use-ase-calculator.md#charge-handling-caveat)**.

## Reload of an ASE settings YAML fails validation

**Symptom**:

```
InvalidSettingsError: ml_system_kwargs contains runtime-only placeholder values
at keys: ['[calculator]']. Supply the actual objects via from_yaml(..., overwrite=...)
before validation.
```

**Cause**: ASE calculators don't serialise to YAML, so `presto` writes a placeholder. On reload the placeholder fails validation by design.

**Fix**: Inject the calculator on load:

```python
settings = WorkflowSettings.from_yaml(
    "workflow_settings.yaml",
    overwrite={
        "training_sampling_settings": {
            "mlp_settings": {"ml_system_kwargs": {"calculator": calculator}}
        }
    },
)
```

## Loss diverges or oscillates

**Symptom**: `plots/loss.png` does not flatten; training loss grows or oscillates over epochs.

**Likely causes and fixes**:

- **Learning rate too high.** Halve `training_settings.learning_rate` (default `0.01`).
- **Too few conformers.** Bump `training_sampling_settings.n_conformers` (default `10`).
- **Types too specific for a congeneric series.** Set `type_generation_settings.<type>.max_extend_distance: 2` — see **[Fit a congeneric series](../how-to/fit-congeneric-series.md)**.

## Too few conformations after outlier filtering

**Symptom**:

```
Filtering would remove too many conformations: ... below min_conformations=...
```

**Cause**: The outlier filter rejected most conformations as having unreasonable MM-vs-MLP differences.

**Fix**: Raise `outlier_filter_settings.energy_outlier_threshold` and/or `force_outlier_threshold`, or set either to `None` to disable that filter. Lowering `min_conformations` is a last resort — a high outlier rate usually signals genuinely bad sampling.

## Duplicate molecules in input

**Symptom**:

```
ValueError: Duplicate inputs found: ['CCO']
```

**Cause**: `ParameterisationSettings.molecules` contains the same SMILES (or SDF path) more than once. Caught by `ParameterisationSettings.normalize_input`.

**Fix**: Deduplicate the list. If you genuinely want to weight one molecule more than another in a congeneric fit, that's not yet supported — open an issue.

## Version mismatch warning on YAML load

**Symptom**:

```
WARNING: Version mismatch: settings version 0.6.0 may not be compatible with current version 0.8.0.
```

**Cause**: `version` in the YAML disagrees with the installed `presto` version at the major or minor level.

**Fix**: Usually benign for patch-level differences. For major/minor differences, regenerate the YAML with `presto write-default-yaml` and re-apply your customisations, especially if the changelog flags a breaking change.

## `linearise_harmonics` inconsistent with `parameter_configs`

**Symptom**:

```
InvalidSettingsError: ParameterisationSettings.linearise_harmonics is True, but
TrainingSettings.parameter_configs contains valence types that are inconsistent
with this setting: ('Bonds', 'Angles').
```

**Cause**: `parameter_configs` has both linear and non-linear forms of bonds/angles, or has the wrong one given `linearise_harmonics`.

**Fix**: Either keep `linearise_harmonics: true` and remove `Bonds`/`Angles` keys from `parameter_configs` (leaving `LinearBonds`/`LinearAngles`), or set `linearise_harmonics: false` and remove `LinearBonds`/`LinearAngles`. See `WorkflowSettings.validate_parameterisation_training_consistency` in [`presto.settings`](../reference/api/settings.md).

## Torsion sampling plot looks sparse

**Symptom**: `plots/torsion_sampling_mol<n>.png` shows torsions stuck in one well.

**Cause**: Insufficient sampling or metadynamics not picking up that torsion.

**Fix**:

- Bump `training_sampling_settings.n_conformers` so different starting conformers cover different regions.
- Lengthen `production_sampling_time_per_conformer`.
- Check `torsions_to_include_smarts` — your torsion of interest may not match the default SMARTS list.
