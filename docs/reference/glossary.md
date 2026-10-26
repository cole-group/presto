# Glossary

One-line definitions for terms used throughout these docs.

**Bespoke parameter**
: A force field parameter generated specifically for the user's molecule(s) of interest, with a SMIRKS pattern derived from the actual atomic connectivity.

**Congeneric series**
: A set of related molecules sharing a common scaffold. In `presto`, congeneric fits share parameters across the series wherever SMIRKS overlap (controlled by `max_extend_distance`).

**Conformer weight**
: A relative weight for one conformer within its entry, set with `conformer_weights_fn`. Only ratios within an entry matter, and 0 excludes the conformer. See **[The loss function](../concepts/loss-function.md)**.

**Dataset weight**
: The relative weight of a training (or testing) dataset in the loss, set with `dataset_weight` and normalised across datasets.

**Entry**
: A set of conformers of one molecule that share an energy offset, such as the snapshots of an MD run, a set of minimised structures, or a row of a pre-computed dataset. Each active entry gets an equal share of its molecule's loss.

**Interchange**
: OpenFF's intermediate representation between SMIRNOFF and downstream MD engines. Produced by `ForceField.create_interchange(topology)`.

**Linearised harmonics**
: Harmonic bond/angle parameters reparametrised as two linear coefficients to stabilise fitting. See the footnote in **[Method overview](../concepts/method-overview.md)**.

**MLP** (machine learning potential)
: A neural-network model trained on QM energies and forces, used as a fast surrogate for QM during sampling and force-field fitting.

**MSM** (modified Seminario method)
: An algorithm for deriving bond and angle force constants and equilibrium values from the molecular Hessian. See [the paper](https://doi.org/10.1021/acs.jctc.7b00785).

**Molecule weight**
: The relative weight of a molecule within a dataset's loss, set with `molecule_weights` and normalised across the dataset's molecules.

**offxml**
: The XML serialisation of a SMIRNOFF force field. Produced by OpenFF and consumed by anything that parses SMIRNOFF.

**SMIRNOFF**
: OpenFF's force field format. Parameters are organised into handlers (Bonds, Angles, …) and matched to atoms via SMIRKS.

**SMIRKS**
: Often used to mean a SMARTS pattern with numerical atom-index tags. `presto` generates one bespoke SMIRKS per bond/angle/torsion in your molecule.

**Type generation**
: The process of building bespoke SMIRKS patterns for a molecule's valence terms. Controlled by `param_settings.type_generation_settings`.

**Valence parameter**
: A force-field parameter governing bonded interactions: bonds, angles, proper torsions, or improper torsions. `presto` only fits valence parameters.

**Well-tempered metadynamics**
: A metadynamics variant where the bias height decays over time according to a `bias_factor`. Used in `mm_md_metadynamics*` protocols to enhance torsion sampling. See [the paper](https://doi.org/10.1103/PhysRevLett.100.020603).
