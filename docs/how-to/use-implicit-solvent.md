# Sample with implicit solvent

MM sampling runs in vacuum by default. For molecules whose gas-phase conformers collapse
onto intramolecular hydrogen bonds or salt bridges that would be broken in solution, this
biases the training set towards configurations you do not care about. Setting
`implicit_solvent` on an MM sampling stage adds a generalised Born solvent term to the MM
system used to generate configurations.

## What it does and does not affect

The solvent term is used **only to influence which configurations are sampled**:

| Stage | Solvent applied? |
|---|---|
| MM MD / metadynamics trajectory | Yes |
| MM torsion-restrained minimisation | Yes |
| MLP energies and forces (the fitting targets) | No, always vacuum |
| MLP minimisation | No, always vacuum |
| MM energies compared against the MLP (loss, scatter plots, outlier filtering) | No, always vacuum |

The force field you fit is therefore still a vacuum force field; only the configuration
distribution it is fitted on changes.

The option is available on `mm_md`, `mm_md_metadynamics` and
`mm_md_metadynamics_torsion_minimisation`. It is not accepted by `ml_md` (the MLP is a
gas-phase potential) or `pre_computed`.

## CLI form

```bash
presto train \
    --param-settings.molecules "CCO" \
    --training-sampling-settings.sampling-protocol mm_md_metadynamics \
    --training-sampling-settings.implicit-solvent.model obc2
```

## YAML form

```yaml
training_sampling_settings:
    sampling_protocol: mm_md_metadynamics_torsion_minimisation
    implicit_solvent:
        model: obc2
        solvent_dielectric: 78.5
        solute_dielectric: 1.0
        surface_area_model: ACE
        salt_concentration: 0.0 M

testing_sampling_settings:
    sampling_protocol: ml_md
    # implicit solvent is not available for ML sampling
```

Leaving `implicit_solvent` unset (or `null`) keeps the stage in vacuum.

## Options

- `model` — the generalised Born model, one of `obc2` (default, Amber `igb=5`), `obc1`
  (`igb=2`), `hct` (`igb=1`), `gbn` (`igb=7`) or `gbn2` (`igb=8`). `obc2` is a sensible
  default for small molecules.
- `solvent_dielectric` — dielectric constant of the solvent, 78.5 (water) by default.
  Lower values mimic less polar solvents.
- `solute_dielectric` — dielectric constant of the solute, 1.0 by default.
- `surface_area_model` — `ACE` (default) adds a non-polar surface area term; `null` omits
  it.
- `salt_concentration` — concentration of monovalent salt, converted to a Debye screening
  parameter exactly as OpenMM does for Amber systems. Zero (the default) means no
  screening.

The radii and screening parameters come from the Amber tables built into OpenMM. These
only cover the elements common in biomolecules, so atoms of other elements (bromine,
iodine, ...) fall back to a default radius of 1.5 Å and a default screening parameter of
0.8. Force fields with virtual sites are not supported.
