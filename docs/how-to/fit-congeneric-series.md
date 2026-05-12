# Fit a congeneric series

When two or more input molecules are supplied, `presto` samples each one separately but trains all parameters together. Whether that *shares* parameters between molecules or just fits each in parallel depends on type specificity.

## Why type specificity matters

By default, `presto` generates SMIRKS patterns specific enough to identify every atom in the molecule. With fully specific types, no SMIRKS matches more than one molecule, so the congeneric fit is mathematically equivalent to fitting each molecule independently.

To share parameters across the series, **reduce specificity** so that SMIRKS for chemically equivalent substructures match across molecules. This is controlled by `max_extend_distance` in `type_generation_settings`.

For the conceptual treatment, see **[Concepts → Type generation and SMIRKS specificity](../concepts/type-generation.md)**.

## `max_extend_distance` recipe

`max_extend_distance` is the number of bonds outward from each tagged atom that the SMIRKS is allowed to extend.

- `-1` (default) — no limit; SMIRKS specifies the whole molecule.
- `2` — typically a reasonable starting point for a congeneric series. Shared substructures up to 2 bonds out collapse onto the same parameter.
- `1` — very loose; parameters share aggressively. May hurt single-molecule loss.

We've found `2` minimally affects training and test loss for typical congeneric series, while letting shared parameters average over the combined dataset (which reduces noise from per-molecule sampling variance).

## Full YAML example

To share parameters between two TYK2 ligands, generate a default YAML and modify the marked fields:

```bash
presto write-default-yaml congeneric_fit.yaml
```

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

## Run it

```bash
presto train-from-yaml congeneric_fit.yaml
```

## SDF inputs

The same applies when you provide multiple unique molecules in a single SDF file. Set `molecule_input_type: sdf` and list `.sdf` paths in `molecules` — see **[Use SDF inputs](use-sdf-inputs.md)**.

## Inspect shared parameters

After the fit, look at `plots/parameter_values_mol<n>.png` for each molecule. Shared parameters will have identical values across molecules; molecule-specific parameters differ. If a parameter you expected to share is showing per-molecule values, your `max_extend_distance` may still be too high — try a smaller value.
