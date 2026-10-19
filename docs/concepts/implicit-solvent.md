# Implicit solvent

Torsional energy profiles in solution can differ from those in the gas phase, particularly for polar molecules where intramolecular hydrogen bonds and electrostatic interactions are screened by the solvent. A force field which will be used with explicit water should arguably reproduce solution-phase, rather than gas-phase, conformational energies. ff19SB ([Tian et al., J. Chem. Theory Comput. 2020](https://doi.org/10.1021/acs.jctc.9b00591)) addressed this by fitting its backbone parameters with both the reference and the MM model in implicit solvent. `presto` can do the same with an MLP as the reference. For the recipe, see **[How-to → Fit in implicit solvent](../how-to/fit-in-implicit-solvent.md)**.

## The decomposition

The MLPs available in `presto` are trained on gas-phase data. Rather than retraining them, a solvation correction is added to each side of the fit:

$$
E_\text{ref}(\mathbf{x}) = E_\text{MLP}(\mathbf{x}) + \Delta E_\text{SMD}(\mathbf{x}),
\qquad
E_\text{MM}(\mathbf{x}; \theta) = E_\text{MM,gas}(\mathbf{x}; \theta) + E_\text{GBn2}(\mathbf{x}) + E_\text{SA}(\mathbf{x}),
$$

and likewise for the forces, where:

- $\Delta E_\text{SMD}$ is the SMD implicit-solvent correction, $E_\text{solv} - E_\text{gas}$ at a fixed geometry, predicted by [AnySolv](https://github.com/Ant-on-knee/anysolv). AnySolv is an ML model trained to reproduce this correction, so it can be added to the energies and forces of any gas-phase potential.
- $E_\text{GBn2}$ is the GBneck2 generalised Born energy (Amber `igb=8`), evaluated with the charges of the MM force field and mbondi3 radii.
- $E_\text{SA}$ is a non-polar term proportional to the solvent-accessible surface area, estimated with LCPO (Amber `gbsa=1`) by default.

Both corrections are single-point solvation energies at fixed geometry, so they are directly comparable. Since energies are compared relative to each other within a dataset entry, only differences between conformations matter.

## Why the GB term can be fixed

$E_\text{GBn2}$ and $E_\text{SA}$ depend only on the coordinates, charges and radii. `presto` trains valence parameters, not charges or radii, so these terms are constant with respect to the trained parameters $\theta$. They are therefore evaluated once per structure with OpenMM and stored with the dataset, rather than implemented as differentiable terms in the MM model. The charges used are the same as those of the MM model in the fit.

## Sampling is independent of fitting

Corrections can also be added to the systems used for sampling, for example to sample conformations with the MM model in implicit solvent. Recorded reference energies and forces always exclude these sampling corrections. The corrections in the fit are always evaluated separately, on the final coordinates of each structure. This means that how structures are sampled and what they are fitted to can be chosen independently.

## Limitations

- **Consistency of the two solvent models.** AnySolv approximates SMD; GBneck2 + LCPO is a different, cheaper model. The bespoke torsions absorb any systematic difference between them, so the resulting force field is specific to GBneck2-style solvation in the same way as ff19SB's parameters.
- **Charges and radii.** The GBneck2 radii and parameters were optimised with Amber charges. With the OpenFF charges used by `presto`, the GB energies may be less accurate, and the fitted torsions also absorb this difference.
- **Element coverage.** GBneck2 has fitted parameters only for H, C, N, O and S. Other elements use OpenMM's default parameters. LCPO has no parameters for bromine or iodine.
- **Water only.** Both models are used with water as the solvent.
- **The modified Seminario Hessians stay in the gas phase.** They are only used to initialise the bond and angle parameters before fitting.

## API reference

[`presto.corrections`](../reference/api/corrections.md), [`AnySolvSMDCorrectionSettings`](../reference/api/settings.md#presto.settings.AnySolvSMDCorrectionSettings) and [`GBn2SASACorrectionSettings`](../reference/api/settings.md#presto.settings.GBn2SASACorrectionSettings).
