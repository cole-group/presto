# Bespoke SMIRNOFF in 5 minutes

If you have a background in MD and QM but you're new to OpenFF or the SMIRNOFF format, read this before the **[Method overview](method-overview.md)**.

## What a SMIRNOFF force field is

SMIRNOFF is OpenFF's force field format. Two ideas you need:

1. **Parameters are organised into handlers** — one each for bonds, angles, proper torsions, improper torsions, van der Waals (vdW), electrostatics, etc.
2. **Parameters are matched to atoms via SMIRKS patterns**, which are SMARTS patterns with atom-index tags. A bond parameter with SMIRKS `[#6X4:1]-[#6X4:2]` applies to every bond between two tetrahedral carbons in your molecule.

When OpenFF builds a system from a SMIRNOFF force field, it walks the parameter list **bottom-up**, so later (more specific) parameters override earlier (less specific) ones.

## What "bespoke" means in presto

`presto` does not invent new parameters from scratch. It starts from a standard OpenFF force field (default: `openff_unconstrained-2.3.0.offxml`) and appends a small set of highly specific SMIRKS parameters tailored to your molecule. Because OpenFF prefers later parameters, the bespoke parameters take precedence over the generic ones whenever they match.

Only the **valence** parameters are bespoke: bonds, angles, proper torsions, improper torsions. The vdW and electrostatic parameters from the input force field are left alone. This is a deliberate scoping decision — see "What `presto` doesn't do" below.

## What an MLP gives you here

A machine-learning potential, trained on QM, gives you near-QM energies and forces at orders-of-magnitude lower cost than the underlying QM method. In `presto`, the MLP plays three roles:

1. **Reference energies/forces** that the bespoke parameters are fitted to reproduce.
2. **Hessian source** for the modified Seminario method, which initialises bond and angle parameters.
3. (Optional) **Sampling potential** in the `ml_md` protocol, where MD is run on the MLP instead of the current MM force field.

The MLP is never modified. `presto` is fitting MM parameters *to* the MLP, not the other way around.

## How presto stitches these together

![Workflow summary](../images/workflow-summary.png)

In short:

1. Generate bespoke SMIRKS for your molecule.
2. Initialise bond/angle parameters from the MLP Hessian (modified Seminario).
3. Sample the molecule with high-temperature MD (default: MM-driven + metadynamics on rotatable bonds).
4. Evaluate MLP energies and forces on the sampled snapshots.
5. Optimise the bespoke valence parameters to reproduce the MLP energies and forces.
6. Repeat (3-5) for additional iterations using the bespoke force field for sampling.
7. Save the bespoke `.offxml` and diagnostic plots.

For the full algorithmic detail, see **[Method overview](method-overview.md)**.

## What presto does not do

- **Fit charges or vdW parameters.** Only valence terms are bespoke. If your molecule has unusual electrostatic or non-bonded behaviour, that's outside `presto`'s scope.
- **Add polarisability.** SMIRNOFF is a fixed-charge format.
- **Test transferability.** The bespoke parameters are tuned to *your* molecules. They will be more accurate for those molecules than the input OpenFF force field, but only marginally usable for unrelated chemistry.
- **Replace QM calculations on the MLP's training set.** The MLP's accuracy ceiling is the QM method it was trained on. Picking the right MLP for your chemistry matters — see **[MLPs in presto](mlps.md)**.

## Glossary

For one-liners on SMIRNOFF, SMIRKS, MSM, congeneric series, and similar terms, see **[Reference → Glossary](../reference/glossary.md)**.
