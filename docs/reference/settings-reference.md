# Settings reference

A curated tour of the `WorkflowSettings` YAML. The same shape is consumed by `presto train-from-yaml`, `presto train` (as dotted-flag overrides), and the Python API (`WorkflowSettings(...)`).

This page reads top-down in the order a user typically edits a YAML. For the alphabetised, exhaustive field-level reference, see the auto-generated **[API reference](api/settings.md)** instead. Both views are driven by the same Pydantic Field descriptions in `presto/settings.py`.

Generate a starter YAML with:

```bash
presto write-default-yaml workflow_settings.yaml
```

## Top-level shape

```yaml
version: 0.8.0                # informational; warns on major/minor mismatch
output_dir: .                 # where outputs are written
device_type: cuda             # "cuda" or "cpu" (cuda requires CUDA >= 12.9)
n_iterations: 2               # number of (sample, train) iterations
memory: false                 # if true, accumulate training data across iterations

parameterisation_settings: ...
training_sampling_settings: ...
testing_sampling_settings: ...
training_settings: ...
outlier_filter_settings: ...
```

See **[`WorkflowSettings`](api/settings.md#presto.settings.WorkflowSettings)** for the full schema.

## `parameterisation_settings`

Inputs and how the initial force field is built.

```yaml
parameterisation_settings:
    molecule_input_type: smiles         # or "sdf"
    molecules:
        - CCO                           # SMILES or .sdf paths, one per molecule
    initial_force_field: openff_unconstrained-2.3.0.offxml
    expand_torsions: true               # expand torsion periodicities up to 4
    linearise_harmonics: true           # linearise bonds/angles for stable fitting
    msm_settings: ...                   # modified Seminario method config
    type_generation_settings: ...       # SMIRKS specificity per valence type
```

The two nested sections are covered below under [Cross-cutting](#cross-cutting-mlpsettings). See **[`ParameterisationSettings`](api/settings.md#presto.settings.ParameterisationSettings)** for every field.

### `parameterisation_settings.msm_settings`

```yaml
msm_settings:
    mlp_settings:                       # see Cross-cutting
        ml_potential: aimnet2
        ml_potential_kwargs: {}
        ml_system_kwargs: {}
    finite_step: 0.0005291772 nm
    tolerance: 0.005291772 kcal * mol**-1 * A**-1
    vib_scaling: 1.0
    n_conformers: 1                     # number of conformers for MSM averaging
```

See **[`MSMSettings`](api/settings.md#presto.settings.MSMSettings)** and the [modified Seminario method paper](https://doi.org/10.1021/acs.jctc.7b00785).

### `parameterisation_settings.type_generation_settings`

One block per valence type. `max_extend_distance: -1` means fully specific (the default); reduce for parameter sharing in congeneric series. See **[Concepts → Type generation](../concepts/type-generation.md)**.

```yaml
type_generation_settings:
    Bonds:
        max_extend_distance: -1
        include: []         # mutually exclusive with `exclude`
        exclude: []
    Angles: { ... }
    ProperTorsions:
        max_extend_distance: -1
        include: []
        exclude:            # default excludes linear torsions
            - '[*:1]-[*:2]#[*:3]-[*:4]'
            - '[*:1]~[*:2]-[*:3]#[*:4]'
            - '[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]'
    ImproperTorsions: { ... }
```

See **[`TypeGenerationSettings`](api/settings.md#presto.settings.TypeGenerationSettings)**.

## `training_sampling_settings`

How training data is generated. The `sampling_protocol` field discriminates between several different settings classes — pick one, then set the protocol-specific knobs.

```yaml
training_sampling_settings:
    sampling_protocol: mm_md_metadynamics_torsion_minimisation   # default
    mlp_settings: { ... }
    timestep: 1 fs
    temperature: 500 K
    snapshot_interval: 0.5 ps
    n_conformers: 10
    equilibration_sampling_time_per_conformer: 0.0 ps
    production_sampling_time_per_conformer: 100 ps
    loss_energy_weight: 1000.0
    loss_force_weight: 0.1
    # ... protocol-specific keys follow
```

The available protocols and what they add:

| `sampling_protocol` | Adds on top of MD base | Class |
|---|---|---|
| `mm_md` | nothing — plain MM MD | [`MMMDSamplingSettings`](api/settings.md#presto.settings.MMMDSamplingSettings) |
| `ml_md` | MLP-driven MD | [`MLMDSamplingSettings`](api/settings.md#presto.settings.MLMDSamplingSettings) |
| `mm_md_metadynamics` | well-tempered metadynamics on rotatable bonds | [`MMMDMetadynamicsSamplingSettings`](api/settings.md#presto.settings.MMMDMetadynamicsSamplingSettings) |
| `mm_md_metadynamics_torsion_minimisation` | metadynamics + torsion-restrained minimisation snapshots | [`MMMDMetadynamicsTorsionMinimisationSamplingSettings`](api/settings.md#presto.settings.MMMDMetadynamicsTorsionMinimisationSamplingSettings) |
| `pre_computed` | skip MD, load dataset from disk | [`PreComputedDatasetSettings`](api/settings.md#presto.settings.PreComputedDatasetSettings) |

For what each protocol *means*, see **[Concepts → Sampling protocols](../concepts/sampling-protocols.md)**.

The metadynamics-specific keys (used by the two metadynamics protocols):

```yaml
bias_width: 0.3141592653589793   # ~π/10 radians
bias_factor: 20.0                # well-tempered scaling (typical 5–20)
bias_height: 1.0 kJ * mol**-1
bias_frequency: 0.1 ps
bias_save_frequency: 10 ps
torsions_to_include_smarts: [ ... ]   # rotatable-bond SMARTS, full 4-atom torsions
torsions_to_exclude_smarts: []        # 2-atom bond SMARTS to exclude from included
```

The torsion-minimisation keys (used only by `mm_md_metadynamics_torsion_minimisation`):

```yaml
ml_minimisation_steps: 10
mm_minimisation_steps: 10
torsion_restraint_force_constant: 0.0 kJ * rad**-2 * mol**-1
map_ml_coords_energy_to_mm_coords_energy: false
loss_energy_weight_mm_torsion_min: 1000.0
loss_force_weight_mm_torsion_min: 0.1
loss_energy_weight_ml_torsion_min: 1000.0
loss_force_weight_ml_torsion_min: 0.1
```

For `pre_computed`, only one key is needed:

```yaml
training_sampling_settings:
    sampling_protocol: pre_computed
    dataset_paths:
        - path/to/training_data.hf
```

## `testing_sampling_settings`

Same shape as `training_sampling_settings`. The default uses `ml_md` (MLP-driven) at 298 K to give physically representative test snapshots, with much shorter sampling time per conformer than training:

```yaml
testing_sampling_settings:
    sampling_protocol: ml_md
    mlp_settings: { ... }
    timestep: 1 fs
    temperature: 298 K
    snapshot_interval: 20 fs
    n_conformers: 10
    production_sampling_time_per_conformer: 2 ps
    # ...
```

## `training_settings`

The optimiser, schedule, and which parameters are trainable.

```yaml
training_settings:
    optimiser: adam              # or "lm" for Levenberg-Marquardt
    parameter_configs: { ... }   # per-valence-type ParameterConfig (see below)
    attribute_configs: {}        # optional 1-4 scaling for vdW/Electrostatics
    n_epochs: 1000
    learning_rate: 0.01
    learning_rate_decay: 1.0     # 0.99 = 1% decay per step; 1.0 = no decay
    learning_rate_decay_step: 10
    regularisation_target: initial   # or "zero"
```

`parameter_configs` has one entry per `ValenceType`. The default applies linearised bonds and angles (`LinearBonds`, `LinearAngles` — required when `linearise_harmonics: true`), trains proper and improper torsion `k` values, and regularises torsion `k` against its starting value. See **[`TrainingSettings`](api/settings.md#presto.settings.TrainingSettings)** and the `descent.train.ParameterConfig` documentation.

!!! note "Linearised vs non-linearised harmonics"
    `parameter_configs` must contain `LinearBonds`/`LinearAngles` when `linearise_harmonics: true`, or `Bonds`/`Angles` when `linearise_harmonics: false`. Mixing them raises `InvalidSettingsError` — see **[Troubleshooting](troubleshooting.md)**.

## `outlier_filter_settings`

Reject sampled snapshots that disagree wildly between the MM force field and the MLP. Set to `null` to disable filtering entirely.

```yaml
outlier_filter_settings:
    energy_outlier_threshold: 2.0     # kcal/mol/atom, vs median energy difference
    force_outlier_threshold: 500.0    # kcal/mol/Å, max per-atom force difference
    min_conformations: 1              # error if filtering would drop below this
```

See **[`OutlierFilterSettings`](api/settings.md#presto.settings.OutlierFilterSettings)** and the relevant **[Troubleshooting](troubleshooting.md#too-few-conformations-after-outlier-filtering)** entry.

## Cross-cutting: `MLPSettings`

`MLPSettings` appears under `parameterisation_settings.msm_settings.mlp_settings`, `training_sampling_settings.mlp_settings`, and `testing_sampling_settings.mlp_settings`. By default each is constructed independently — to pin the same MLP everywhere, set it in all three places. See **[How-to → Choose an MLP](../how-to/choose-an-mlp.md#pin-the-same-mlp-across-stages)**.

```yaml
mlp_settings:
    ml_potential: aimnet2            # any OpenMM-ML model name; "ase" for an ASE Calculator
    ml_potential_kwargs: {}          # passed to MLPotential(...)
    ml_system_kwargs: {}             # passed to MLPotential.createSystem(...)
```

For ASE, see **[How-to → Use an ASE calculator](../how-to/use-ase-calculator.md)**.

See **[`MLPSettings`](api/settings.md#presto.settings.MLPSettings)** for serialiser/validator details (runtime-only objects in `ml_system_kwargs` round-trip via YAML using a placeholder string — reload with `from_yaml(..., overwrite=...)`).

## How `sampling_protocol` acts as a discriminator

`training_sampling_settings` and `testing_sampling_settings` are declared as `SamplingSettings`, a Pydantic union discriminated by the `sampling_protocol` field. When loading a YAML, Pydantic uses the value of `sampling_protocol` to pick which subclass to instantiate. This means:

- Setting `sampling_protocol: mm_md` disallows the metadynamics-specific keys (`bias_height`, etc.). They become invalid extras for `MMMDSamplingSettings`.
- Switching protocols mid-YAML requires re-checking which keys are still valid.

If you see `Extra inputs are not permitted` errors, this is usually the cause — confirm the keys you're setting are valid for the chosen `sampling_protocol`.

## Default YAML in full

Use `presto write-default-yaml` to dump the full default. The output is reproduced in **[examples/basic-walk-through-cli.ipynb](../examples/basic-walk-through-cli.ipynb)**.

## See also

- **[CLI reference](cli.md)** — same fields as dotted command-line flags.
- **[API reference: `presto.settings`](api/settings.md)** — every field, alphabetised, with full type annotations.
