# Choose an MLP

`presto` supports any [OpenMM-ML](https://github.com/openmm/openmm-ml) model name as the reference potential. The choice affects fit accuracy, speed, charge handling, and (importantly) licence. For the conceptual treatment, see **[Concepts → MLPs in presto](../concepts/mlps.md)**.

## Recommended defaults

The MLPs below are installed by default in the pixi environment. Both have licences that permit industrial use and were trained including charged molecules.

- **`aimnet2`** (MIT) — relatively fast and robust. Currently the `presto` default. A ~50-atom molecule on a single A4500 fits in ~15 minutes.
- **`orb-v3-conservative-omol`** (Apache-2.0) — more accurate, trained on OMol25. Expect fits to take ~2× longer than AIMNet2 at the same molecule size.

You can use any OpenMM-ML model name (and model-specific kwargs) supported by your environment; `presto` does not enforce a fixed allowlist. For a benchmark of MLPs see [this paper](https://arxiv.org/abs/2601.16331).

## Some other supported names

Other models which are installed in the default pixi environment when installing from GitHub.

- **`aceff-2.0`** — historically the default. **warning "Currently broken upstream"***
    There are open [issues with AceFF 2.0 in OpenMM-ML](https://github.com/openmm/openmm-ml/issues/136). Use alternatives until they're resolved.
- **`mace-off`** variants — ***warning "Academic licence only"***
    [MACE-OFF](https://github.com/ACEsuit/mace) is released under the [Academic Software License](https://github.com/gabor1/ASL/blob/main/ASL.md), which **does not permit commercial use**.
- **`egret-1`** — permissive licence; supported by OpenMM-ML.

## Pin the same MLP across stages

By default each sampling-settings object instantiates its own `MLPSettings`. To use the same MLP for training-sampling, testing-sampling, and MSM, set it in three places:

```yaml
param_settings:
    msm_settings:
        mlp_settings:
            ml_potential: aimnet2
training_sampling_settings:
    mlp_settings:
        ml_potential: aimnet2
testing_sampling_settings:
    mlp_settings:
        ml_potential: aimnet2
```

Programmatically:

```python
from presto.settings import MLPSettings, ...

mlp = MLPSettings(ml_potential="aimnet2")
settings = WorkflowSettings(
    param_settings=ParamSettings(
        molecules="CCO",
        msm_settings=MSMSettings(mlp_settings=mlp),
    ),
    training_sampling_settings=MMMDMetadynamicsTorsionMinimisationSamplingSettings(
        mlp_settings=mlp,
    ),
    testing_sampling_settings=MLMDSamplingSettings(mlp_settings=mlp),
)
```

## Pass model-specific kwargs

`MLPSettings` has two kwargs dicts:

- `ml_potential_kwargs` — passed to `openmmml.MLPotential(...)` (e.g. model variant, precision).
- `ml_system_kwargs` — passed to `MLPotential.createSystem(...)` at simulation time.

Refer to the OpenMM-ML docs for the supported keys per model.

For ASE-backed models, see **[Use an ASE calculator](use-ase-calculator.md)**.
