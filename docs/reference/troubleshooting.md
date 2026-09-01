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


## Conformer generation fails for some molecules

**Symptom**: settings validation fails immediately, before any fitting starts:

```
InvalidSettingsError: Conformer generation failed for 2 of 40 molecules:
  - molecule 12 (ligand-13, CC(C)...): ... ConformerGenerationError : RDKit conformer generation failed.
  - molecule 27 (ligand-28, c1ccc...): ... ConformerGenerationError : RDKit conformer generation failed.
```

**Cause**: RDKit's ETKDG cannot embed those molecules. `presto` checks this up front, for every
molecule at once, because a molecule that cannot be embedded cannot be parameterised: the modified
Seminario method and every sampling stage need a starting conformer, and AM1BCC charge assignment
generates conformers internally too. Without the check, the failure would only surface part way
through the MSM stage, after the expensive ML potential work had already run for every earlier
molecule.

**Fixes**:

- Remove the offending molecules from `param_settings.molecules`. The error lists every one of
  them, so a whole input set can be fixed in a single pass.
- Or supply geometries for them yourself. Set `starting_conformers` on the relevant stages (see
  **[How-to → Use your own starting conformers](../how-to/use-starting-conformers.md)**). Note this
  is not sufficient on its own: AM1BCC will still try to embed the molecule during charge
  assignment, so you also need to bypass that with library charges (see
  **[How-to → Use custom charges](../how-to/use-custom-charges.md)**).

Note that `presto clean` and `presto analyse` validate settings too, so they will also refuse to
run on a configuration containing an unembeddable molecule.


## Version mismatch warning on YAML load

**Symptom**:

```
WARNING: Version mismatch: settings version 0.6.0 may not be compatible with current version 0.8.0.
```

**Cause**: `version` in the YAML disagrees with the installed `presto` version at the major or minor level.

**Fix**: Usually benign for patch-level differences. For major/minor differences, regenerate the YAML with `presto write-default-yaml` and re-apply your customisations, especially if the changelog flags a breaking change.
