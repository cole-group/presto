# Changelog

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
