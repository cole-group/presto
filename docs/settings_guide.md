# Settings guide

`presto` can be run directly using its CLI
```bash
presto train --parameterisation-settings.smiles "CCC(CC)C(=O)Nc2cc(NC(=O)c1c(Cl)cccc1Cl)ccn2"
```
or from a YAML file
```bash
presto write-default-yaml default.yaml
# Modify the yaml to set the desired smiles
presto train-from-yaml default.yaml
```

## How to get help

For details on available options and defaults, see the [settings API reference](reference/settings.md#presto.settings).

Running
```bash
presto train --help
```
will also show available options.

Note that the key option when specifying `training_sampling_settings` or `testing_sampling_settings` is `sampling_protocol`, which determines the available sampling settings. See the available [`SamplingSettings`](reference/settings.md#presto.settings.SamplingSettings) classes for a description of all implemented sampling protocols. See the associated sampling_protocol field in each class for the string identifier which should be supplied to `training_sampling_settings` and `testing_sampling_settings` fields in `WorkflowSettings`.

## Recommended defaults

### Single-molecule fit

For single molecule fits, we recommend using the default settings without modification. The default force field, AceFF-2.0, can handle charged species. By default, the [`mm_md_metadynamics_torsion_minimisation`](reference/settings/#presto.settings.MMMDMetadynamicsTorsionMinimisationSamplingSettings) sampling protocol is used, which includes well-tempered metadynamics on all rotatable bonds, as well as samples generated using short MLP (and MM) minimisations. We found these sampling helpful for improving torsion scans, as they often result in configurations with erroneously large steric clashes according to the MM force field.

### Congeneric series fit

For fitting congeneric series, we recommend reducing the specificity of types so that parameters are mostly shared between the common substructures in different molecules. This is intended to reduce noise in the fits by removing noise in chemically equivalent parameters resulting from the generation of different samples with molecular dynamics.

Specifically, the [`max_extend_distance`](reference/settings/#presto.settings.TypeGenerationSettings.max_extend_distance) should be changed from -1 (fully specific). We've found 2 to be a reasonable default which minimally affects the training and test loss. To run a shared fit between two TYK2 ligands, for example, generate the default yaml with
```bash
presto write-default-yaml congeneric_fit.yaml
```
Then modify the `smiles` and `max_extend_distance` (within `type_generation_settings`) options `parameterisation_settings` section to read:
```yaml
parameterisation_settings:
    smiles:
      - CCC(CC)C(=O)Nc2cc(NC(=O)c1c(Cl)cccc1Cl)ccn2
      - CCC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1
    initial_force_field: openff_unconstrained-2.3.0.offxml
    expand_torsions: true
    linearise_harmonics: true
    msm_settings:
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

Run this with
```bash
presto train-from-yaml congeneric_fit.yaml
```
