# The loss function

This page defines the loss that `presto` minimises during training, and what each weight setting does to it.

## Entries

The training data for one molecule in one dataset is made of one or more **entries**. An entry is a set of conformers that share an energy offset: the snapshots of an MD run, a set of minimised structures, or a row of a pre-computed dataset. For example, the torsion-minimisation protocols give each molecule three entries per iteration (the MD snapshots, the MLP-minimised and the MM-minimised structures), and `memory: true` adds the entries of earlier iterations.

## Weights

Each level of the loss is a mean over the level below, and each level has its own weight:

| Level | Setting | Meaning |
|---|---|---|
| Dataset | `dataset_weight` | Relative, normalised across the training (or testing) datasets. |
| Molecule | `molecule_weights` | Relative, normalised across the molecules of a dataset. 0 leaves a molecule out. |
| Entry | none | Each active entry of a molecule gets one equal share. |
| Conformer, absolute | `loss_energy_weight`, `loss_force_weight` and the `*_torsion_min` variants, stored as `energy_weights` and `forces_weights` | The absolute scale of each conformer's energy and force errors. |
| Conformer, relative | `conformer_weights_fn`, stored as `conformer_weights` | Relative importance within an entry. Only ratios matter, and 0 excludes a conformer. Defaults to 1. |

Absolute weights set the scale: energies against forces against regularisation, and one entry against another. Relative weights only move weight between the conformers of one entry and cannot change the scale. Conformers whose reference energy (or forces) are not finite are excluded from that term automatically.

Conformer counts are never weights. Adding conformers to an entry does not change its influence, and adding conformers or `memory` iterations does not change a molecule's. Only `molecule_weights` changes a molecule's influence. The one count that matters is the number of entries: each active entry of a molecule gets one share.

## Per-molecule loss

For conformer \(i\) of entry \(e\), let \(w_i\) be its absolute weight for the term, \(r_i\) its relative conformer weight (set to 0 if its reference data for the term are missing) and \(d_i\) its squared error: the squared energy error per atom, or the mean squared force component error. The entry's loss is the weighted mean

$$
L_e = \frac{\sum_{i \in e} w_i r_i d_i}{\sum_{i \in e} r_i}
$$

and the molecule's loss is the mean of \(L_e\) over the entries that are **active** for that term. An entry is active when \(\sum_i w_i r_i > 0\) and, for energies only, it has at least two conformers with \(r_i > 0\): with one, the energy offset makes its error zero, so it carries no energy information. Inactive entries drop out of the mean instead of counting as zero, so an entry type with a weight of 0 (e.g. `loss_force_weight_ml_torsion_min: 0`) drops out of that term entirely. If no entry is active, the term is 0.

This loss is linear in every absolute weight, so doubling `loss_energy_weight` doubles the energy loss, and raising any conformer's weight never lowers the loss. With a constant error, \(L_e\) is the mean weight of the entry.

## Energy offset

Only relative energies are meaningful, so each entry's reference and predicted energies are offset by their own weighted means before they are compared:

$$
E_0 = \frac{\sum_{i \in e} r_i E_i}{\sum_{i \in e} r_i}
$$

The offset uses the relative weights only, so excluded conformers do not shift it, a relative weight of 2 is the same as duplicating a conformer, and a weight of 0 is the same as removing it. It ignores the absolute weights so that the loss stays linear in them.

## Molecules, datasets and regularisation

The loss of dataset \(k\) is the weighted mean over its molecules,

$$
L_k = \frac{\sum_j m_j L_j}{\sum_j m_j}
$$

where \(m_j\) is the molecule's entry in `molecule_weights` (all 1 by default). The datasets are combined with their normalised `dataset_weight`, as described in **[Multiple datasets](sampling-protocols.md#multiple-datasets)**. The energy and force losses are combined in the same way, and the regularisation loss is added once:

$$
L_\mathrm{reg} = \frac{1}{N} \sum_p \lambda_p (\theta_p - \theta_p^0)^2
$$

where the sum runs over the \(N\) regularised parameters, \(\lambda_p\) is the parameter's `regularize` weight and \(\theta_p^0\) its target. The gradient and, for the Levenberg-Marquardt optimiser, the Hessian use the same weights. The test loss is computed in the same way, with the testing protocols' weights.

## Comparison with earlier versions

Before this version the per-molecule loss was \(\sum_i w_i d_i / \mathrm{ESS}(w)\), pooled over all of a molecule's entries, with \(\mathrm{ESS}(w) = (\sum_i w_i)^2 / \sum_i w_i^2\). With equal weights and entries of equal size, the two are identical, so fits with default settings are unchanged. Results change when:

- the absolute weights within a molecule differ, e.g. non-default `loss_*_weight_*_torsion_min`, where the old loss could fall as a weight rose;
- outlier filtering leaves a molecule's entries with different numbers of conformers, or a pre-computed dataset has rows of different sizes, since each entry now keeps one share;
- a `regularize` config has unequal weights, since the regularisation loss is now divided by the number of regularised parameters.

A per-conformer absolute weight of 0 no longer excludes that conformer from its entry's mean: use a conformer weight of 0 instead.

See **[Weight conformers and molecules](../how-to/weight-conformers-and-molecules.md)** for how to set the weights.
