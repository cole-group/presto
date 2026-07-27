# Train double-exponential vdW parameters

`presto` can fit a force field that uses the double-exponential (DE) vdW functional form
supplied by [`smirnoff-plugins`](https://github.com/openforcefield/smirnoff-plugins), such
as the [de-forcefields](https://github.com/jthorton/de-forcefields) releases. This guide
covers fitting the two **handler-level shape parameters** of that form, `alpha` and `beta`.

Nothing here is on by default: with no extra configuration `presto` fits only the valence
terms, exactly as it does for a Lennard-Jones force field.

## Per-atom parameters vs handler attributes

The DE form has two kinds of parameter, and they are configured differently:

| | Where they live | Config field |
|---|---|---|
| `epsilon`, `r_min` | one row per SMIRKS pattern | `TrainingSettings.parameter_configs` |
| `alpha`, `beta` | a single value for the whole force field | `TrainingSettings.attribute_configs` |

`alpha` and `beta` are global, so they are *attributes* rather than *parameters*, and the
knob for them is descent's
[`AttributeConfig`](https://github.com/openforcefield/descent) rather than
`ParameterConfig`.

## Configuring the fit

```python
from descent.train import AttributeConfig

from presto.settings import ParamSettings, TrainingSettings, WorkflowSettings

settings = WorkflowSettings(
    smiles=["CCO"],
    param_settings=ParamSettings(
        initial_force_field="de-force_unconstrained-1.0.3.offxml",
    ),
    training_settings=TrainingSettings(
        attribute_configs={
            "vdW": AttributeConfig(
                cols=["alpha", "beta"],
                # The DE energy contains 1 / (alpha - beta), so keep the two ranges
                # well separated or the fit can walk into the singularity and
                # produce NaN losses.
                limits={"alpha": (8.0, 40.0), "beta": (1.0, 8.0)},
                scales={"alpha": 1.0, "beta": 1.0},
            )
        },
    ),
)
```

Three things are easy to get wrong:

- **The config key is `"vdW"`, not `"DoubleExponential"`.** `smee` relabels a DE potential's
  type as `"vdW"` when it converts the force field, so both nonbonded forms share one key.
- **Always set `limits`.** The DE repulsion and attraction prefactors are
  `beta / (alpha - beta)` and `alpha / (alpha - beta)`. Unbounded training can drive
  `alpha` towards `beta` and blow the energy up. Ranges that cannot overlap, as above, rule
  this out.
- **If you also fit the per-atom parameters, the columns are `epsilon` and `r_min`** — not
  `epsilon` and `sigma`, which is what a Lennard-Jones force field uses:

  ```python
  parameter_configs={
      "vdW": ParameterConfig(cols=["epsilon", "r_min"], ...),
      ...,
  }
  ```

  Passing the Lennard-Jones column names for a DE force field trips an assertion inside
  `descent`.

## Inspecting the result

Fitted `alpha` and `beta` are written back into the `DoubleExponential` section of the
bespoke `.offxml` in each `training_iteration_*` directory:

```xml
<DoubleExponential version="0.3" ... alpha="17.1 * dimensionless ** 1" beta="4.2 * dimensionless ** 1">
```

Their trajectory across fitting iterations is plotted in
`plots/handler_attributes.png`. Unlike `parameter_values.png` this plot is *not* per
molecule, because the attributes are global to the force field.

## Notes and caveats

- Force fields using a custom vdW form must be loaded with plugins enabled. `presto` does
  this everywhere it reads an `.offxml`, and it keeps the nonbonded forces separate when
  building OpenMM systems (`combine_nonbonded_forces=False`), since a DE potential becomes
  a `CustomNonbondedForce` that cannot be folded into a single `NonbondedForce`.
- `alpha` and `beta` are shared by every molecule in the fit. Fitting them against a single
  small molecule is unlikely to give a transferable improvement; treat them as global
  quantities and fit them against as broad a set as you can.
