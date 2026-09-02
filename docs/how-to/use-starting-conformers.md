# Use your own starting conformers

By default each sampling stage generates its starting conformers with RDKit ETKDG. You can
instead start from your own conformers (for example a curated ensemble, docked or crystal
poses, or a torsion scan) by pointing a stage at an SDF file with `starting_conformers`.

This is available **per stage**, so you can mix supplied conformers and ETKDG freely:

- `training_sampling_settings.starting_conformers` — training-data MD
- `testing_sampling_settings.starting_conformers` — test-data MD
- `param_settings.msm_settings.starting_conformers` — the MSM Hessian step that seeds the
  initial bond/angle parameters

Any stage left unset keeps generating conformers with ETKDG.

## Behaviour

- **The SDF takes precedence.** When `starting_conformers` is set for a stage, that stage
  starts from **every** conformer in the file, and the stage's `n_conformers` is **ignored**
  (a log line makes this explicit).
- **Conformers are matched by connectivity.** Each record is matched to the molecule being
  fitted by graph isomorphism, and its atom ordering is **remapped automatically** to match.
  You do not need to worry about atom ordering, and a single SDF can hold conformers for
  several molecules in a multi-molecule fit — each molecule picks up only its own records.
- **Missing molecules fail fast.** If a configured SDF contains no conformer for a molecule
  being fitted, the run stops during settings validation, before any expensive work.

## CLI form

```bash
presto train \
    --param-settings.molecules "CCO" \
    --training-sampling-settings.starting-conformers my_conformers.sdf
```

## YAML form

```yaml
param_settings:
    molecule_input_type: smiles
    molecules:
        - CCO
    msm_settings:
        # optional: also seed the MSM Hessian step from the same (or a different) SDF
        starting_conformers: my_conformers.sdf

training_sampling_settings:
    sampling_protocol: mm_md_metadynamics_torsion_minimisation
    starting_conformers: my_conformers.sdf

testing_sampling_settings:
    sampling_protocol: ml_md
    # left unset -> ETKDG
```

## Preparing the SDF

The SDF should contain one record per conformer, all of the same molecule (multiple
molecules are fine — group them however you like, they are matched by connectivity). For
example, from an existing OpenFF molecule:

```python
from rdkit import Chem

rdkit_molecule = molecule.to_rdkit()  # molecule with N conformers
with Chem.SDWriter("my_conformers.sdf") as writer:
    for conformer_id in range(rdkit_molecule.GetNumConformers()):
        writer.write(rdkit_molecule, confId=conformer_id)
```
