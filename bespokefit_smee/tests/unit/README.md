"""
Comprehensive Unit Tests for bespokefit_smee
=============================================

This test suite provides comprehensive unit tests for the bespokefit_smee package.

Test Coverage
-------------

The test suite covers the following modules:

1. **test_settings.py** - Comprehensive tests for all settings classes:
   - SamplingSettings (MM MD, ML MD, Metadynamics)
   - RegularisationSettings
   - TrainingSettings
   - MSMSettings
   - ParameterisationSettings
   - WorkflowSettings
   - Uses hypothesis for property-based testing

2. **test_outputs.py** - Tests for output management:
   - OutputType and StageKind enums
   - OutputStage dataclass
   - WorkflowPathManager
   - File creation and cleanup functionality
   - delete_path utility

3. **test_find_torsions.py** - Tests for torsion finding:
   - get_single_torsion_by_rot_bond
   - get_unwanted_bonds
   - get_rot_torsions_by_rot_bond
   - SMARTS pattern matching
   - Uses hypothesis for property-based testing on alkanes

4. **test_exceptions.py** - Tests for custom exceptions:
   - InvalidSettingsError

5. **test_mlp.py** - Tests for machine learning potential loading:
   - Model registry and caching
   - EGRET-1 model loading
   - Multiple model support

6. **test_cli.py** - Tests for CLI commands:
   - WriteDefaultYAML
   - TrainFromYAML
   - Clean
   - Analyse
   - Integration tests with subprocess

7. **test_loss_functions.py** - Tests for loss and regularisation:
   - get_regularised_parameter_idxs
   - compute_regularisation_penalty
   - Parameter scaling and caching

8. **test_writers.py** - Tests for output writers:
   - write_scatter (HDF5 output)
   - open_writer (TensorBoard)
   - write_metrics
   - get_potential_summary

9. **test_utils.py** - Tests for utility modules:
   - Type definitions
   - Registry decorator
   - Output suppression

10. **test_workflow.py** - Tests for workflow orchestration:
    - Workflow stage creation
    - Settings persistence
    - Memory mode
    - Device handling
    - Iteration configuration

Running the Tests
-----------------

To run all unit tests:
```bash
pixi run test-unit
```

To run all tests (unit + integration):
```bash
pixi run test
```

To run with coverage:
```bash
pixi run pytest -v --cov=bespokefit_smee --cov-report=html bespokefit_smee/tests/unit
```

To run specific test files:
```bash
pixi run pytest bespokefit_smee/tests/unit/test_settings.py -v
```

To run tests matching a pattern:
```bash
pixi run pytest -k "test_regularisation" -v
```

Test Fixtures
-------------

Common fixtures are defined in conftest.py:

- `tmp_cwd`: Temporary working directory
- `jnk1_lig_smiles`: Complex molecule SMILES for testing
- `ethanol_molecule`: Simple ethanol molecule
- `ethanol_with_conformers`: Ethanol with generated conformers
- `simple_force_field`: OpenFF force field
- `ethanol_tensor_topology_and_ff`: Tensor representations
- `simple_trainable`: Trainable object for optimization tests

Property-Based Testing
----------------------

Several tests use the Hypothesis library for property-based testing:

- Sampling time calculations
- Parameter ranges (learning rates, regularisation strengths)
- SMILES string validation
- Alkane torsion detection

This approach tests a wide range of inputs automatically to catch edge cases.

Test Categories
---------------

Tests are organized by type:

1. **Unit tests** (`tests/unit/`):
   - Test individual functions and classes in isolation
   - Use mocking where necessary
   - Fast execution

2. **Integration tests** (`tests/integration/`):
   - Test full workflows end-to-end
   - Test CLI integration
   - May be slower

Notes on Test Design
--------------------

1. **Isolation**: Tests are independent and can run in any order

2. **Fixtures**: Common setup is extracted to fixtures for reusability

3. **Parametrization**: pytest.mark.parametrize is used extensively to test
   multiple scenarios efficiently

4. **Hypothesis**: Property-based tests explore edge cases automatically

5. **Mocking**: Used sparingly, primarily for filesystem and network operations

6. **Coverage**: Aims for >80% code coverage with focus on critical paths

7. **Fast execution**: Unit tests should run in < 1 minute total

Known Limitations
-----------------

Some functionality is difficult to test in unit tests:

- Full MD simulations (tested in integration tests)
- GPU-specific code (requires CUDA-enabled CI)
- ML model predictions (mocked in unit tests)
- Long-running optimizations (tested with small examples)

For these cases, integration tests and end-to-end tests provide coverage.

Contributing Tests
------------------

When adding new functionality:

1. Add unit tests for new functions/classes
2. Add integration tests for new workflows
3. Use hypothesis for numeric inputs when appropriate
4. Update this README if adding new test categories
5. Ensure all tests pass before submitting PR

Running with Different Settings
-------------------------------

Run tests with different verbosity:
```bash
pixi run pytest -v      # Verbose
pixi run pytest -vv     # Very verbose
pixi run pytest -q      # Quiet
```

Run specific test classes:
```bash
pixi run pytest bespokefit_smee/tests/unit/test_settings.py::TestWorkflowSettings -v
```

Run only failed tests:
```bash
pixi run pytest --lf    # Last failed
pixi run pytest --ff    # Failed first
```

Debug failed tests:
```bash
pixi run pytest --pdb   # Drop into debugger on failure
```
