# Fit in implicit solvent

By default `presto` fits the MM force field to gas-phase MLP energies and forces. To fit in implicit solvent, in the spirit of ff19SB, add a solvation correction to each side of the fit:

- the **reference** becomes the MLP energies and forces plus an SMD implicit-solvent correction predicted by [AnySolv](https://github.com/Ant-on-knee/anysolv);
- the **MM model** becomes the MM energies and forces plus GBneck2 (Amber `igb=8`) and an LCPO surface area term (Amber `gbsa=1`).

For the background, see **[Concepts → Implicit solvent](../concepts/implicit-solvent.md)**.

## Install AnySolv

AnySolv is only available from PyPI, so it is an optional dependency. The full pixi environments (`default`, `gpu-*`, `cpu-*`) include it through the `anysolv` feature. If you installed `presto-fit` from conda-forge, or use a `base*` environment, install it yourself:

```bash
pip install "anysolv[hub]==0.1.0"
```

The GBneck2 correction only needs OpenMM, so MM-only corrections work without AnySolv. `presto` checks that every configured correction can be evaluated before parameterisation starts, and raises an error explaining how to install anything missing.

### Model weights

The default AnySolv checkpoint, `model_moe`, is the most accurate but its weights are gated:

1. Request access and accept the FAIR Chemistry License at [huggingface.co/antonknee/anysolv](https://huggingface.co/antonknee/anysolv).
2. Log in with `hf auth login`.

`presto` downloads the weights (~1.1 GB) to the Hugging Face cache on first use. Alternatively, set `checkpoint: model_compact` to use the smaller model bundled with AnySolv (MIT licence, no login), or give the path to a downloaded `.pt` file.

## Fit in implicit solvent

Add both corrections to the workflow settings:

```yaml
reference_corrections:
    - correction_type: anysolv_smd
mm_corrections:
    - correction_type: gbn2_sasa
```

Programmatically:

```python
from presto.settings import (
    AnySolvSMDCorrectionSettings,
    GBn2SASACorrectionSettings,
    WorkflowSettings,
)

settings = WorkflowSettings(
    param_settings=...,
    reference_corrections=[AnySolvSMDCorrectionSettings()],
    mm_corrections=[GBn2SASACorrectionSettings()],
)
```

This applies the corrections to both the training and the test data, so the losses, outlier filtering and plots all compare the solvated reference with the solvated MM model.

By default:

- **MM sampling is in implicit solvent.** Each sampling stage has `mm_sampling_corrections: from_fit`, which adds the `mm_corrections` (GBneck2 + LCPO) to the MM systems used for MD and minimisation.
- **ML sampling is in the gas phase.** Each sampling stage has `ml_sampling_corrections: []`. The AnySolv correction is still evaluated on every stored structure, but not at every MD or minimisation step, which would be much more expensive.

## Choose what is used where

Whether a correction is part of the fit and whether it is used during sampling are independent. The fit corrections are set once in `reference_corrections` and `mm_corrections`. The sampling corrections are set per sampling stage in `ml_sampling_corrections` and `mm_sampling_corrections`, each of which is a list of corrections or `from_fit`.

Recorded reference energies and forces never include the sampling corrections. They are evaluated separately for the fit, so changing how you sample never changes what you fit to.

### Sample in implicit solvent, fit in the gas phase

```yaml
training_sampling_settings:
    sampling_protocol: mm_md_metadynamics_torsion_minimisation
    mm_sampling_corrections:
        - correction_type: gbn2_sasa
```

With no `reference_corrections` or `mm_corrections`, the fit is unchanged, but the training structures are sampled with the MM model in implicit solvent.

### Run the MLP in implicit solvent during sampling

```yaml
training_sampling_settings:
    sampling_protocol: mm_md_metadynamics_torsion_minimisation
    ml_sampling_corrections: from_fit
testing_sampling_settings:
    sampling_protocol: ml_md
    ml_sampling_corrections: from_fit
```

This adds AnySolv to the MLP for the torsion-restrained MLP minimisations and for ML MD. AnySolv is then evaluated at every step, which can make sampling several times slower with `model_moe`. If you need this, consider `inference_settings: fast_gpu` (CUDA) for `model_moe`, or `checkpoint: model_compact` with `inference_settings: fast`.

## Pre-computed datasets

Saved datasets store the uncorrected reference in `energy` and `forces`, and the corrections in separate columns (`reference_energy_correction`, `reference_forces_correction`, `mm_energy_correction` and `mm_forces_correction`). A dataset from a solvated fit can therefore be reused for a gas-phase fit, and vice versa.

For `pre_computed` sampling, `reference_corrections_mode` controls the reference corrections:

| Mode | Reference corrections |
|---|---|
| `keep` (default) | Use the corrections stored in the dataset (zero if there are none). |
| `recompute` | Evaluate the workflow's `reference_corrections` on the stored coordinates. |
| `zero` | Fit to the uncorrected reference. |

The MM corrections are always recomputed from the workflow's `mm_corrections`, as they describe the MM model rather than the data.

```yaml
training_sampling_settings:
    sampling_protocol: pre_computed
    dataset_paths:
        - path/to/gas_phase_dataset
    reference_corrections_mode: recompute
```

## Options

| Setting | Default | Meaning |
|---|---|---|
| `anysolv_smd.checkpoint` | `model_moe` | `model_moe`, `model_compact`, or a path to a `.pt` file. |
| `anysolv_smd.solvent` | `water` | Only water is supported. |
| `anysolv_smd.dtype` | `float32` | Inference precision. |
| `anysolv_smd.inference_settings` | `default` | `default`, `fast` or `fast_gpu`. |
| `gbn2_sasa.solvent_dielectric` | 78.5 | Solvent dielectric constant. |
| `gbn2_sasa.solute_dielectric` | 1.0 | Solute dielectric constant. |
| `gbn2_sasa.sasa_model` | `lcpo` | `lcpo` (Amber `gbsa=1`), `ace` (OpenMM's ACE approximation) or `null`. |
| `gbn2_sasa.surface_tension` | 0.005 kcal/mol/Å² | LCPO surface tension (Amber's `surften`). |
| `gbn2_sasa.probe_radius` | 1.4 Å | LCPO solvent probe radius. |

## Caveats

- **Cost.** AnySolv is evaluated once per stored structure, for the training data in every iteration and once for the test data. This takes about 40 ms per structure on a CPU for `model_compact` (small molecules); `model_moe` is more expensive, so run on a GPU.
- **Element coverage.** GBneck2 has no fitted parameters for halogens or phosphorus, so OpenMM's defaults are used for them, with a warning. LCPO has no parameters for bromine or iodine; use `sasa_model: ace` for such molecules.
- **Outlier filtering** compares the solvated reference and MM model, so the thresholds in `outlier_filter_settings` apply to solvated differences.
- **Charges are fixed.** GBneck2 uses the charges of the MM model, and is not differentiated with respect to the trained parameters. This is valid because `presto` does not train charges.

## API reference

[`AnySolvSMDCorrectionSettings`](../reference/api/settings.md#presto.settings.AnySolvSMDCorrectionSettings), [`GBn2SASACorrectionSettings`](../reference/api/settings.md#presto.settings.GBn2SASACorrectionSettings), and [`presto.corrections`](../reference/api/corrections.md).
