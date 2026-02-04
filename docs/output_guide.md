# Output guide

Your bespoke force field will be available at `<your fitting directory>/training_iteration_<n>/bespoke_ff.offxml>`, where `n` is the number of iterations you requested.

Plots showing the changes in parameters and energy/ force errors will be generated in `<your fitting directory>/plots`. It's worth checking that these look reasonable (new parameters are reasonable, validation loss decreases).

For more details on the outputs, see the [API reference for the OutputType enum](reference/outputs.md#presto.outputs.OutputType).
