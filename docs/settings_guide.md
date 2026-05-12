# Settings guide

`presto` can be run directly using its CLI
```bash
presto train --parameterisation-settings.molecule-input-type smiles --parameterisation-settings.molecules "CCC(CC)C(=O)Nc2cc(NC(=O)c1c(Cl)cccc1Cl)ccn2"
```
or from a YAML file
```bash
presto write-default-yaml default.yaml
# Modify the yaml to set the desired molecule_input_type and molecule input(s)
presto train-from-yaml default.yaml
```

## Using the fitting workflow via the Python API

You can call the workflow directly from Python instead of the CLI:

```python
from presto.settings import ParameterisationSettings, WorkflowSettings
from presto.workflow import get_bespoke_force_field

settings = WorkflowSettings(
    parameterisation_settings=ParameterisationSettings(
        molecule_input_type="smiles",
        molecules="CCO",
    ),
    device_type="cuda",
)

bespoke_ff = get_bespoke_force_field(settings)
```

You can also use an arbitrary ASE calculator through OpenMM-ML by setting
`mlp_settings.ml_potential="ase"` and passing runtime arguments through
`mlp_settings.ml_system_kwargs`:

```python
from presto.settings import MLPSettings, MLMDSamplingSettings, ParameterisationSettings, WorkflowSettings
from presto.workflow import get_bespoke_force_field

# Example only: replace with your own ASE calculator instance
calculator = ...

settings = WorkflowSettings(
    parameterisation_settings=ParameterisationSettings(
        molecule_input_type="smiles",
        molecules="CCO",
    ),
    device_type="cuda",
    training_sampling_settings=MLMDSamplingSettings(
        mlp_settings=MLPSettings(
            ml_potential="ase",
            ml_system_kwargs={"calculator": calculator},
        ),
    ),
)

bespoke_ff = get_bespoke_force_field(settings)
```

Runtime objects (for example an in-memory calculator object) are written to YAML as placeholders.
If you reload such a YAML, inject the calculator before validation:

```python
loaded = WorkflowSettings.from_yaml(
    "workflow_settings.yaml",
    overwrite={
        "training_sampling_settings": {
            "mlp_settings": {"ml_system_kwargs": {"calculator": calculator}}
        }
    },
)
```

!!! warning "Charge handling"
    For non-ASE models, `presto` automatically passes molecular charge to
    `MLPotential.createSystem(...)`. For `ml_potential="ase"`, charge is **not**
    automatically propagated. Supply it explicitly in `mlp_settings.ml_system_kwargs`
    (for example under `info`), or use preconfigured `aseAtoms`.

## How to get help

For details on available options and defaults, see the [settings API reference](reference/settings.md#presto.settings).

Running
```bash
presto train --help
```
will also show available options.

Note that the key option when specifying `training_sampling_settings` or `testing_sampling_settings` is `sampling_protocol`, which determines the available sampling settings. See the available [`SamplingSettings`](reference/settings.md#presto.settings.SamplingSettings) classes for a description of all implemented sampling protocols. See the associated sampling_protocol field in each class for the string identifier which should be supplied to `training_sampling_settings` and `testing_sampling_settings` fields in `WorkflowSettings`.

## Recommended defaults

### Recommended MLP choices

The choices below are installed by default if you pixi install from the GitHub repo. Both have licenses which allow industry use and included charged molecules during training.

- `aimnet2` (MIT): Relatively fast but generally robust default and generally a good first choice. `aimnet2` fits with the default protocol, 1 A4500, and a ~ 50 atom molecule should take ~ 15 minutes.
- `orb-v3-conservative-omol` (Apache-2.0): A very accurate model trained on OMol25. Expect fits to take ~ twice as long as with `aimnet2`, depending on the size of the molecule.

You can use any OpenMM-ML model name (and model-specific kwargs) supported by your
environment; `presto` does not enforce a fixed allowlist. For a helpful MLP benchmark, see [here](https://arxiv.org/abs/2601.16331).

### Single-molecule fit

For single molecule fits, we recommend using the default settings without modification. The default force field, AceFF-2.0, can handle charged species. By default, the [`mm_md_metadynamics_torsion_minimisation`](reference/settings/#presto.settings.MMMDMetadynamicsTorsionMinimisationSamplingSettings) sampling protocol is used, which includes well-tempered metadynamics on all rotatable bonds, as well as samples generated using short MLP (and MM) minimisations. We found these sampling helpful for improving torsion scans, as they often result in configurations with erroneously large steric clashes according to the MM force field.

### Congeneric series fit

For fitting congeneric series, we recommend reducing the specificity of types so that parameters are mostly shared between the common substructures in different molecules. This is intended to reduce noise in the fits by removing noise in chemically equivalent parameters resulting from the generation of different samples with molecular dynamics.

Specifically, the [`max_extend_distance`](reference/settings/#presto.settings.TypeGenerationSettings.max_extend_distance) should be changed from -1 (fully specific). We've found 2 to be a reasonable default which minimally affects the training and test loss. To run a shared fit between two TYK2 ligands, for example, generate the default yaml with
```bash
presto write-default-yaml congeneric_fit.yaml
```
Then modify the `input_type`, `input`, and `max_extend_distance` (within `type_generation_settings`) options in the `parameterisation_settings` section to read:
```yaml
parameterisation_settings:
        molecule_input_type: smiles
        molecules:
      - CCC(CC)C(=O)Nc2cc(NC(=O)c1c(Cl)cccc1Cl)ccn2
      - CCC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1
    initial_force_field: openff_unconstrained-2.3.0.offxml
    expand_torsions: true
    linearise_harmonics: true
    msm_settings:
        mlp_settings:
            ml_potential: aceff-2.0
        finite_step: 0.0005291772 nm
        tolerance: 0.005291772 kcal * mol**-1 * A**-1
        vib_scaling: 0.958
        n_conformers: 1
    type_generation_settings:
        Bonds:
            max_extend_distance: 2
            include: []
            exclude: []
        Angles:
            max_extend_distance: 2
            include: []
            exclude: []
        ProperTorsions:
            max_extend_distance: 2
            include: []
            exclude:
            - '[*:1]-[*:2]#[*:3]-[*:4]'
            - '[*:1]~[*:2]-[*:3]#[*:4]'
            - '[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]'
        ImproperTorsions:
            max_extend_distance: 2
            include: []
            exclude: []
```

For SDF-based fits, set `molecule_input_type: sdf` and provide one or more `.sdf` file paths in the `molecules` list. Multiple unique molecules in a single SDF file are supported.

Run this with
```bash
presto train-from-yaml congeneric_fit.yaml
```
