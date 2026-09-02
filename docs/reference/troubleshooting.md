# Troubleshooting

First-stop for known failure modes. If your symptom isn't listed here, check the [GitHub issue tracker](https://github.com/cole-group/presto/issues).

## Unstable Fit

**Symptom**: `plots/loss.png` does not flatten; losses blow up; MD is unstable and fit does not finish; parameters change massively.

**Likely causes and fixes**:

- Change in connectivity caused by e.g. protons hopping during the MLP minimisation stages. Inspect the output pdbs to see what samples went into training. Fix by avoiding MLP minimisations by switching to the `mm_md_metadynamics` sampling protocol (and deleting the parameters associated with the minimisation stages). Often occurs for phosphates.
- Poor initial equilibrium values for bonds and angles from MSM initialisation. The [modified Seminario method](../concepts/method-overview.md#initial-force-field) initialises bond and angle equilibrium values directly from MLP-minimised geometries, ignoring the effect of non-bonded interactions on equilibrium geometry. This can introduce instabilities due to, for example, problematic S–N bonds in sulfonamides. If you see large initial bond/angle deviations, try disabling MSM so that bond and angle parameters start from the parent force field values instead. Initial losses will typically be higher but generall converge to similar values as with MSM. Disable MSM by setting **`param_settings.msm_settings`** to `None` (Python) or `null` (YAML).


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

## Too few conformations after outlier filtering

**Symptom**:

```
Filtering would remove too many conformations: ... below min_conformations=...
```

**Cause**: The outlier filter rejected most conformations as having unreasonable MM-vs-MLP differences. See "Unstable Fit" above.


## Molecule input validation fails at the start of training

**Symptom**: the run stops as soon as training starts, before any output directory is created:

```
InvalidSettingsError: Workflow molecule input validation failed for 2 of 3 molecules:
  - molecule 0 (CCO)
      ETKDG required by testing_sampling_settings, training_sampling_settings, param_settings.msm_settings: ToolkitWrapper around The RDKit version 2025.09.5 <class 'openff.toolkit.utils.exceptions.ConformerGenerationError'> : RDKit conformer generation failed.
  - molecule 2 (c1ccccc1)
      ETKDG required by testing_sampling_settings, training_sampling_settings, param_settings.msm_settings: ToolkitWrapper around The RDKit version 2025.09.5 <class 'openff.toolkit.utils.exceptions.ConformerGenerationError'> : RDKit conformer generation failed.
```

**Cause**: RDKit's ETKDG cannot embed those molecules, and at least one configured workflow stage
needs to generate a starting conformer for them. `presto` checks this at the start of training,
before the modified Seminario method or sampling begins, and reports every affected molecule at once.

**Fixes**:

- Remove the offending molecules from `param_settings.molecules`. The error lists every one of
  them, so a whole input set can be fixed in a single pass.
- Or supply geometries for them yourself. Set `starting_conformers` on every stage listed in the
  error (see **[How-to → Use your own starting conformers](../how-to/use-starting-conformers.md)**).

The same check validates supplied starting conformers, and reports two further symptoms:

- A molecule with no matching record in a supplied SDF is listed per stage under the same heading:

  ```
  InvalidSettingsError: Workflow molecule input validation failed for 2 of 2 molecules:
    - molecule 0 (CCC)
        param_settings.msm_settings.starting_conformers (conformers.sdf): ValueError: SDF file conformers.sdf contains no conformers matching the molecule [H]C([H])([H])C([H])([H])C([H])([H])[H].
  ```

  Records are matched by graph isomorphism, so a molecule missing here usually means a different
  protonation or tautomeric state, not a different atom ordering (ordering is handled automatically).

- A missing or unreadable SDF is a problem with the setting rather than the molecules, so it is
  reported once per stage:

  ```
  InvalidSettingsError: Workflow starting-conformer files could not be read:
    - training_sampling_settings.starting_conformers: SDF file does not exist: conformers.sdf
  ```

Charge assignment is validated separately by OpenFF during parameterisation, which stops at the
first molecule it cannot handle rather than collecting them (it is the most expensive phase, and
molecule geometry problems have already been caught by the check above):

```
MoleculeParameterisationError: OpenFF parameterisation/charge assignment failed for molecule 3 (ligand-4, CC(C)...): ...
```

The default Sage 2.3 force field uses the graph-based Ash–GC model and does not need conformers.
Older force fields using AM1-BCC can still fail to parameterise an unembeddable molecule; supplying
library charges is one way to bypass AM1-BCC (see
**[How-to → Use custom charges](../how-to/use-custom-charges.md)**).

`presto clean` and `presto analyse` do not execute this training-only geometry check.


## Version mismatch warning on YAML load

**Symptom**:

```
WARNING: Version mismatch: settings version 0.6.0 may not be compatible with current version 0.8.0.
```

**Cause**: `version` in the YAML disagrees with the installed `presto` version at the major or minor level.

**Fix**: Usually benign for patch-level differences. For major/minor differences, regenerate the YAML with `presto write-default-yaml` and re-apply your customisations, especially if the changelog flags a breaking change.
