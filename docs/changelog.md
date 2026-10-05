# Changelog

## Unreleased (051_double_exponential)

### Improvements

- Load force fields with ``load_plugins=True`` so that SMIRNOFF plugin handlers (e.g. ``DoubleExponential``) can be used as the initial force field. Valence terms are fitted as normal and the plugin vdW parameters are kept fixed.
- Keep vdW and electrostatics as separate OpenMM forces when building MM systems, as plugin vdW forms cannot be combined into a single ``NonbondedForce``.

### Fixes

- Guard against lack of any constraints in the input force field ([#56](https://github.com/cole-group/presto/pull/56)). This allows use of the unconstrained version of Parsely.
- Never remove parameters without an ID when dropping unused bespoke parameters.
- Coerce dimensionless handler attributes (e.g. ``DoubleExponential`` ``alpha``/``beta``) read as raw floats into ``Quantity`` objects, which Interchange requires.

## 0.5.1

### Fixes

- Fixes typo in default settings which meant that linear torsions matching ``[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]`` were not excluded from training ([#32](https://github.com/cole-group/presto/pull/32))

## 0.5.0

### Improvements

- Use MDTraj to calculate torsions rather than doing this manually
- Make metadynamics more aggressive, allow user to specify torsions targeted by metadynamics, and include in-ring aliphatic torsions by default in [#26](https://github.com/cole-group/presto/pull/26)

### Fixes

- Remove unused settings fields in [#25](https://github.com/cole-group/presto/pull/25#issuecomment-3861009692)

## 0.4.0

### Improvements

- Improve cleanliness/prettiness of CLI output
- Reduce memory use during congeneric series fitting in [#17](https://github.com/cole-group/presto/pull/17)
- Improve test quality, speed, and coverage in [#19](https://github.com/cole-group/presto/pull/19)
- Update environments with explicit python and cuda versions and document installation with different cuda versions
- Plot sampling of rotatable torsions in [#23](https://github.com/cole-group/presto/pull/23)

### Fixes

- Update ParameterConfig defaults so that linear torsions are not trained in [#18](https://github.com/cole-group/presto/pull/18)

## 0.3.0

- Renamed `bespokefit_smee` -> `presto` (Parameter Refinement Engine for Smirnoff Training / Optimisation)
- Added more documentation on the method, recommended settings, and outputs.

## 0.2.0

### New Features

- Implement new default protocol with MLP-minimised configurations
- Add support for loading multiple pre-computed datasets for training
- Implement flexible bespoke SMARTS type generation with `MergeQueryHs`
- Add support for multi-molecule simultaneous fits
- Add dataset filtering function for preprocessing
- Add function for calculating Hessian matrices

### Improvements

- **GPU Memory Management**: Significantly improved GPU memory handling
  - Add GPU memory cleanup utility function
  - Clear GPU memory after sampling operations
  - Reduce GPU memory usage throughout training pipeline
  - Fix GPU memory leaks with LM optimizer
  - Make CUDA operations conditional on availability for CPU-only environments
- **Modified Seminario Method (MSM)**
  - Complete reimplementation of MSM for better performance
- **ML Potential Updates**
  - Add aceff-2.0 and make it the default MLP
  - Update default sampling settings (consistent with higher speed of aceff-2.0)
- **Force Field Updates**
  - Update default MM-FF to 2.3.0
  - Ensure we train FF with bespoke types added
  - Ensure parameter names are not overwritten
- **Regularization Improvements**
  - Overhaul regularization and calculation of loss
  - Normalize regularization per-parameter
  - Decouple type generation from regularization
- **Optimizer Fixes**
  - Fix LM optimizer implementation
  - Clear cache between iterations with Adam
  - Avoid GPU memory leaks with LM optimizer
- **Path Management**
  - Fix path management for multiple molecules
  - Improve handling of output paths for per-molecule outputs

### Maintenance

- Remove bespoke toolkit wrapper

## 0.1.1

### Documentation

- Added example notebook

## 0.1.0

- Initial implementation.
