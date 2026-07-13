# Use custom charges

By default a `presto` fit takes partial charges from the base force field's charge model (usually Ash-GC). If you have your own partial charges — from a higher level of theory, a custom charge model, or an external pipeline — you can bake them into the force field as **library charges** so they are used in place of the default model.

`presto` provides [`add_library_charges_to_forcefield`](../reference/api/create_types.md#presto.create_types.add_library_charges_to_forcefield) for this. It takes an OpenFF `Molecule` (or list of molecules) that already has `partial_charges` set, plus a `ForceField`, and returns a copy of the force field with `LibraryCharges` written for those charges.

This is a standalone helper: it is **not** part of the automatic fitting pipeline. You call it yourself to produce a force field (or `.offxml` file) that you then use as your starting point.

## Set partial charges on the molecule

Any method that populates `Molecule.partial_charges` works — assign them with the toolkit, or set them directly from your own data:

```python
import numpy as np
import openff.toolkit
from openff.units import unit

mol = openff.toolkit.Molecule.from_smiles("CCO")

# Option A: assign with a toolkit charge method
mol.assign_partial_charges("am1bcc")

# Option B: set your own charges directly (must be ordered by atom index)
mol.partial_charges = np.array([...]) * unit.elementary_charge
```

The charges must sum to the molecule's formal charge. If they do not, `add_library_charges_to_forcefield` raises a `ValueError` up front rather than letting the error surface later when the system is built.

## Write the library charges

```python
from presto.create_types import add_library_charges_to_forcefield

ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")
ff_with_charges = add_library_charges_to_forcefield(mol, ff)

# Persist for reuse / inspection
ff_with_charges.to_file("custom_charges.offxml")
```

The original `ff` is left unchanged; a modified copy is returned. The new parameters are tagged with `l-bespoke-*` IDs under the `LibraryCharges` handler.

Pass a list to write charges for several molecules at once:

```python
ff_with_charges = add_library_charges_to_forcefield([mol_a, mol_b], ff)
```

## How the charges are matched

The library charges reproduce your input charges and always keep the molecule net-neutral (or at its formal charge). This is achieved by reusing `presto`'s [type generation](../concepts/type-generation.md) machinery:

- One library charge is generated per atom, using a whole-molecule SMARTS with a single tagged atom (the non-tagged hydrogens are merged onto their heavy atoms, which keeps SMARTS matching fast).
- Because each pattern spans the whole molecule, it only matches that atom's symmetry class, so every atom is covered exactly once.
- Symmetry-equivalent atoms collapse onto one library charge whose value is the **mean** of their input charges. Averaging both symmetrises equivalent atoms and preserves the total charge exactly.

A whole-molecule SMARTS is used (rather than a smaller `max_extend_distance`) to ensure that charges are only applied to the molecule they were derived for: a `LibraryCharges` handler is only applied if it covers every atom of the molecule, and the assigned charges must sum to the formal charge. Whole-molecule patterns guarantee both.

## Use the charges in a fit

Point `initial_force_field` at the `.offxml` you wrote:

```yaml
param_settings:
    molecule_input_type: smiles
    molecules:
        - CCO
    initial_force_field: custom_charges.offxml
```

`presto` only retrains valence parameters (bonds, angles, torsions), so your custom charges are carried through the fit unchanged. Make sure the molecules you fit are the same ones you wrote charges for, so the library charges match.

## Reference

- API reference: [`add_library_charges_to_forcefield`](../reference/api/create_types.md#presto.create_types.add_library_charges_to_forcefield).
- Concept: [Type generation and SMIRKS specificity](../concepts/type-generation.md).
