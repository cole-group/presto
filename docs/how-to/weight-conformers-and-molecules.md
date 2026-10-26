# Weight conformers and molecules

By default every molecule in a dataset counts equally, and every conformer of an entry counts equally. This page shows how to change both. See **[The loss function](../concepts/loss-function.md)** for exactly how the weights enter the loss.

## Weight molecules

Give a protocol one `molecule_weights` value per molecule, in the order the molecules are loaded from `param_settings.molecules`:

```yaml
param_settings:
    molecules:
        - CCO
        - CCCO
        - CCCCO
training_sampling_settings:
-   sampling_protocol: mm_md_metadynamics_torsion_minimisation
    molecule_weights: [2.0, 1.0, 1.0]
```

The weights are normalised by their sum, so this gives the first molecule half of the dataset's loss and the others a quarter each, and multiplying every weight by a constant changes nothing. A molecule with weight 0 is still sampled but is left out of the loss. Each training and testing protocol has its own `molecule_weights`, and `None` (the default) weights the molecules equally.

This is the only way to give a molecule more influence. A molecule with more conformers does not count for more.

## Weight conformers with your own function

`conformer_weights_fn` names a Python function that returns a relative weight for each conformer of a molecule's dataset:

```yaml
training_sampling_settings:
-   sampling_protocol: pre_computed
    dataset_paths: [torsion_scan_ala]
    conformer_weights_fn: ramachandran_weights:ramachandran_weights
```

The value is `"package.module:func"`. The module must be importable, so put a script's directory on `PYTHONPATH`, e.g. `PYTHONPATH=$PWD presto train-from-yaml settings.yaml`. The function is imported when the fit starts, before any output is written, so a typo fails straight away. `presto clean` and `presto analyse` never import it.

### The contract

The function is called once per molecule with that molecule's dataset (a `datasets.Dataset` in torch format) and must return an array-like of one weight per conformer:

- The weights are ordered entry by entry, so there are `sum(len(entry["energy"]) for entry in dataset)` of them.
- They must be finite and non-negative, with at least one positive.
- Only ratios within an entry matter. The weights move weight between the conformers of one entry, but cannot move it between entries or molecules (use the absolute loss weights or `molecule_weights` for that).
- A weight of 0 excludes the conformer from both the energy and force loss, and from the energy offset. An entry whose weights are all 0 drops out.
- Conformers with missing (non-finite) reference data are masked automatically, so the function does not need to handle them.
- The function must not read an existing `conformer_weights` column: it is overwritten each time.

The function runs on training data after outlier filtering and after `memory` has appended earlier iterations, so it sees exactly the conformers the loss will. It also runs on testing data from protocols that set it, and on `pre_computed` data.

The dataset's `"smiles"` column holds a mapped SMILES whose atom order matches the coordinates, so `Molecule.from_mapped_smiles` gives you atom indices to measure geometry with.

### Example: Ramachandran weights for capped amino acids

This function weights torsion scans of capped amino acids (`ACE-X-NME`) by a per-residue grid over the backbone \(\phi\) and \(\psi\) angles. Each grid is a NumPy array of shape `(n_phi, n_psi)` whose bins evenly span \(-180°\) to \(180°\) in each angle.

```python
"""Ramachandran weights for capped amino acids, saved as ramachandran_weights.py."""

import numpy as np
from openff.toolkit import Molecule

PHI_SMARTS = "[#6X3:1](=[#8])-[#7X3:2]-[#6X4:3]-[#6X3:4](=[#8])"
PSI_SMARTS = "[#7X3:1]-[#6X4:2]-[#6X3:3](=[#8])-[#7X3:4]"

# One grid per residue, keyed by canonical SMILES
GRIDS = {
    Molecule.from_smiles(smiles).to_smiles(): np.load(path)
    for smiles, path in {
        "CC(=O)N[C@@H](C)C(=O)NC": "grids/ala.npy",
        "CC(=O)NCC(=O)NC": "grids/gly.npy",
    }.items()
}


def _dihedrals(coords: np.ndarray, idxs: tuple[int, ...]) -> np.ndarray:
    """Dihedral angles in degrees, for coords of shape (n_confs, n_atoms, 3)."""
    p0, p1, p2, p3 = (coords[:, i] for i in idxs)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    x = (v * w).sum(-1)
    y = (np.cross(b1, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def _bin(angles: np.ndarray, n_bins: int) -> np.ndarray:
    return np.floor((angles + 180.0) / 360.0 * n_bins).astype(int) % n_bins


def ramachandran_weights(dataset) -> np.ndarray:
    mol = Molecule.from_mapped_smiles(
        dataset[0]["smiles"], allow_undefined_stereo=True
    )
    grid = GRIDS[mol.to_smiles()]
    phi_idxs = mol.chemical_environment_matches(PHI_SMARTS)[0]
    psi_idxs = mol.chemical_environment_matches(PSI_SMARTS)[0]

    weights = []
    for entry in dataset:
        n_confs = len(entry["energy"])
        coords = entry["coords"].reshape(n_confs, -1, 3).cpu().numpy()
        phi = _bin(_dihedrals(coords, phi_idxs), grid.shape[0])
        psi = _bin(_dihedrals(coords, psi_idxs), grid.shape[1])
        weights.append(grid[phi, psi])
    return np.concatenate(weights)
```

Swap the bin lookup for an interpolation if your grid is coarse. For a torsion scan, which covers \(\phi\) and \(\psi\) evenly, the grid can be the target probability density itself. For MD samples, which are already distributed unevenly, divide the target density by an estimate of the sampled density to reweight towards the target.

The same function can serve several residues in one fit, since it looks up each molecule's grid by its SMILES.
