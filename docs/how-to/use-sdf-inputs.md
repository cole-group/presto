# Use SDF inputs

By default `presto` reads SMILES strings under `parameterisation_settings.molecules`. To use one or more SDF files instead, set `molecule_input_type: sdf`.

## CLI form

```bash
presto train \
    --parameterisation-settings.molecule-input-type sdf \
    --parameterisation-settings.molecules input_molecule.sdf
```

For multiple SDF files, repeat the `--parameterisation-settings.molecules` flag, or use the YAML form.

## YAML form

```yaml
parameterisation_settings:
    molecule_input_type: sdf
    molecules:
        - ligand_a.sdf
        - ligand_b.sdf
```

## Multi-molecule SDF behaviour

Each `.sdf` may contain one or more molecules. All molecules across all SDF files are loaded into the same list and fitted simultaneously (see **[Fit a congeneric series](fit-congeneric-series.md)** for the shared-parameter recipe). Molecule indices in output filenames (`*_mol0`, `*_mol1`, …) match the load order.
