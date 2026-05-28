# MLPs in presto

`presto` uses a machine learning potential (MLP) as the source of reference energies, forces, and (optionally) the Hessian for the modified Seminario method. This page is about what each supported MLP gives you and when to swap. For the recipe-level "how do I configure it" treatment, see **[How-to → Choose an MLP](../how-to/choose-an-mlp.md)**.

## What an MLP is in this context

An MLP here is a machine learning potential trained on QM energies and forces, exposed through OpenMM-ML's `MLPotential` API. From `presto`'s perspective, the MLP is a black box that takes coordinates and returns energies and forces. It is never modified. The MLP's accuracy ceiling is the QM method it was trained on.

## Reference vs sampling potential

`presto` distinguishes between the **sampling** potential (used to propagate MD) and the **reference** potential (used to compute the energies/forces that the bespoke parameters are fitted to):

- For `mm_md*` protocols, the sampling potential is the current MM force field; the reference is the MLP.
- For `ml_md`, both sampling and reference are the MLP.
- For `pre_computed`, there is no sampling; the reference is implicit in the saved dataset.

Both roles are configured via `MLPSettings`. The MSM step (modified Seminario for initial bond/angle parameters) uses its own `MLPSettings` under `param_settings.msm_settings`. By default, each `MLPSettings` instance is constructed independently — see **[Choose an MLP → Pin the same MLP across stages](../how-to/choose-an-mlp.md#pin-the-same-mlp-across-stages)** if you want consistency.

## What changes when you swap MLPs

- **Cost.** AIMNet2 is fast; Orb-v3 OMol fits are roughly 2× slower at similar molecule size.
- **Charge handling.** Most OpenMM-ML MLPs accept molecular charge via `MLPotential.createSystem(...)`. `presto` propagates this automatically — **except** for `ml_potential="ase"`, which doesn't expose a charge interface. See **[Use an ASE calculator → Charge handling caveat](../how-to/use-ase-calculator.md#charge-handling-caveat)**.
- **Licence.** Most defaults are MIT or Apache-2.0. MACE-OFF is Academic Software License — not for commercial use. See the table in **[Installation → MLP licensing notes](../get-started/installation.md#mlp-licensing-notes)**.
- **Accuracy on charged species.** MLPs that included charged molecules in their training set (AIMNet2, AceFF-2.0, Orb-v3 OMOL) should handle them better than MLPs which did not.

## Why AIMNet2 is the current default

AIMNet2 is default because it's fast, MIT-licensed, fairly robust on small molecules, and trained on a dataset that includes charged species.

## Why ASE is special

The `ml_potential="ase"` adapter lets you bring any [ASE Calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculator.html) as the MLP. This is the right escape hatch when:

- The MLP you want has an ASE interface but no native OpenMM-ML support.
- You're prototyping a new model and don't want to write an OpenMM-ML wrapper yet.

Cost: the ASE bridge doesn't accept molecular charge automatically, and the calculator object cannot round-trip through YAML. See **[Use an ASE calculator](../how-to/use-ase-calculator.md)** for the patterns.

## Benchmarking

For an independent comparison of MLPs, see [this paper](https://arxiv.org/abs/2601.16331).

## API reference

[`MLPSettings`](../reference/api/settings.md#presto.settings.MLPSettings).
