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

We've found `2` minimally affects training and test loss for TYK2 ligands, while letting shared parameters average over the combined dataset (which is intended to reduce noise from per-molecule finite MD sampling variance). Values larger than 2 will very likely be required in some cases.

## Full YAML example

To share parameters between two TYK2 ligands, generate a default YAML and modify the `max_extend_distance` parameters for all valence types:

```bash
presto write-default-yaml congeneric_fit.yaml
```

```yaml
parameterisation_settings:
    molecule_input_type: smiles
    molecules:
        - CCC(CC)C(=O)Nc2cc(NC(=O)c1c(Cl)cccc1Cl)ccn2
        - CCC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1
    ...
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
    ...
```

## Run it

```bash
presto train-from-yaml congeneric_fit.yaml
```

## SDF inputs

The same applies when you provide multiple unique molecules in a single SDF file. Set `molecule_input_type: sdf` and list `.sdf` paths in `molecules` — see **[Use SDF inputs](use-sdf-inputs.md)**.
