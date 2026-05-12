# How-to guides

Short, task-oriented recipes. Each page assumes you've finished **[Get started](../get-started/index.md)**.

## By task

- **[Fit a single molecule](fit-single-molecule.md)** — defaults you should keep, defaults you may want to change.
- **[Fit a congeneric series](fit-congeneric-series.md)** — share parameters across related molecules using `max_extend_distance`.
- **[Use SDF inputs](use-sdf-inputs.md)** — switch from SMILES to one or more `.sdf` files.
- **[Resume, iterate, and clean a run](resume-iterate-clean.md)** — re-run, re-analyse, or wipe a fit directory.

## By component

- **[Choose an MLP](choose-an-mlp.md)** — what each supported MLP gives you, and how to pin it.
- **[Use an ASE calculator](use-ase-calculator.md)** — bring your own ML potential via ASE.
- **[Use a pre-computed dataset](use-precomputed-dataset.md)** — skip sampling and train on existing energy/force data.
- **[Inspect outputs and plots](inspect-outputs.md)** — what each file in the output directory means.
- **[Run from Python](run-from-python.md)** — build `WorkflowSettings` programmatically instead of via the CLI.
