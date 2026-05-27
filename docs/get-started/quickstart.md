# Quickstart

Go from a SMILES to a bespoke force field. Assumes you've finished **[Installation](installation.md)**.

## Train a force field from one SMILES

```bash
presto train --parameterisation-settings.molecules "CCO"
```

By default this runs two iterations of sampling + training using the AIMNet2 reference potential. On a single A4500 GPU, a 50-atom molecule takes around 15 minutes.

## Inspect the output

When the run finishes, the bespoke force field is at:

```
training_iteration_2/bespoke_ff.offxml
```

The loss curve and per-iteration diagnostic plots are under `plots/`. To check whether the fit looks reasonable, see **[Inspect outputs and plots](../how-to/inspect-outputs.md)**.

## Use the force field

`bespoke_ff.offxml` is a standard SMIRNOFF file:

```python
from openff.toolkit import ForceField, Molecule

ff = ForceField("training_iteration_2/bespoke_ff.offxml")
mol = Molecule.from_smiles("CCO")
system = ff.create_interchange(mol.to_topology()).to_openmm()
```

## Run from a YAML file instead

For reproducible runs, edit a settings file:

```bash
presto write-default-yaml workflow_settings.yaml
# edit the `parameterisation_settings.molecules` field
presto train-from-yaml workflow_settings.yaml
```

See the **[API reference](../reference/api/settings.md)** for what every field does.

## Common next steps

- **[How-to → Fit a single molecule](../how-to/fit-single-molecule.md)** — which defaults to keep and which to change.
- **[How-to → Fit a congeneric series](../how-to/fit-congeneric-series.md)** — share parameters across related molecules.
- **[How-to → Run from Python](../how-to/run-from-python.md)** — build `WorkflowSettings` programmatically.

## If anything goes wrong

See **[Reference → Troubleshooting](../reference/troubleshooting.md)** for known failure modes and their fixes.
