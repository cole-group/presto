# Bespoke SMIRNOFF in 5 minutes

If you have a background in MD but you're new to the SMIRNOFF format, read this before the **[Method overview](method-overview.md)**.

## What a SMIRNOFF force field is

SMIRNOFF (SMIRKS Native Open Force Field) is OpenFF's force field format. A few key points:

1. **Parameters are organised into handlers** — one each for bonds, angles, proper torsions, improper torsions, van der Waals (vdW), electrostatics, etc.
2. **Parameters are matched to atoms via SMIRKS patterns** (more properly SMARTS patterns with atom-index tags). A bond parameter with SMIRKS `[#6X4:1]-[#6X4:2]` applies to every bond between two tetrahedral carbons in your molecule.
3. **Parameters that appear later in a SMIRNOFF specification override those which come earlier** if they both match the same bond/ angle/ torsion etc.

## What "bespoke" means in presto

`presto` starts from a SMIRNOFF force field (default: `openff_unconstrained-2.3.0.offxml`) and appends a small set of (usually) highly specific SMIRKS parameters tailored to your molecule. Because later parameters take precedence, the bespoke parameters take precedence over the generic ones whenever they match.

By default only the **valence** parameters are bespoke: bonds, angles, proper torsions, improper torsions. The vdW and electrostatic parameters from the input force field are left alone unless you opt in — see **[Train double-exponential vdW parameters](../how-to/train-double-exponential.md)**.

## What an MLP gives you here

A machine-learning potential trained on QM data can give you near-QM energies and forces at orders-of-magnitude lower cost than the underlying QM method. In `presto`, the MLP plays three roles:

1. **Reference energies/forces** that the bespoke parameters are fitted to reproduce.
2. **Hessian source** for the modified Seminario method, which initialises bond and angle parameters.
3. (Optional) **Sampling potential** in the `ml_md` protocol, where MD is run on the MLP instead of the current MM force field.

## How presto stitches these together

![Workflow summary](../images/workflow-summary.png)

In short:

1. Generate bespoke SMIRKS for your molecule.
2. Initialise bond/angle parameters from the MLP Hessian (modified Seminario method).
3. Sample the molecule with high-temperature MD (default: MM-driven + metadynamics on rotatable bonds).
4. Run short minimisations (with the MLP and MM force field) to supplement the high temperature samples (only in the "torsion_minimisation" protocol, which is default).
5. Evaluate MLP energies and forces on the sampled snapshots.
6. Optimise the bespoke valence parameters to reproduce the MLP energies and forces.
7. Repeat (3-6) for additional iterations using the bespoke force field for sampling.
8. Save the bespoke `.offxml` and diagnostic plots.

For more algorithmic detail, see **[Method overview](method-overview.md)**.

## What presto does not do

- **Fit charges.** `presto` never refits the charges assigned by the input force field.
- **Fit vdW parameters by default.** Only valence terms are bespoke out of the box. vdW fitting is opt-in and off unless you configure it — see **[Train double-exponential vdW parameters](../how-to/train-double-exponential.md)**.
- **Add polarisability.** SMIRNOFF is a fixed-charge format.
- **Provide transferable parameters.** The bespoke parameters are tuned to *your* molecules.
- **Check you've selected a suitable MLP for your system.** The MLP's accuracy ceiling is the QM method it was trained on, and your molecule(s) of interest may be dissimilar to the MLP training set (e.g. your molecule is charged but the MLP training set is neutral). Picking the right MLP for your chemistry matters — see **[MLPs in presto](mlps.md)**.

## Glossary

For one-liners on SMIRNOFF, SMIRKS, MSM, congeneric series, and similar terms, see **[Reference → Glossary](../reference/glossary.md)**.
