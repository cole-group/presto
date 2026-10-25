"""Functionality for handling the outputs of a workflow."""

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import SamplingSettings, TrainingSettings


class OutputType(Enum):
    """An enumeration of the different types of outputs produced by bespoke fitting functions.

    Every run lays out files in the following stages under ``output_dir``:

    - ``workflow_settings.yaml`` at the top, recording the inputs.
    - ``test_data/`` containing the validation datasets.
    - ``initial_statistics/`` with the offxml and scatter data for the starting force field.
    - ``training_iteration_<n>/`` for each iteration, containing the bespoke offxml,
      sampled trajectories, energies/forces, and TensorBoard metrics.
    - ``plots/`` aggregating loss and parameter diagnostic plots across iterations.

    Outputs belonging to one training or testing dataset sit in a
    ``train_dataset_<k>/`` or ``test_dataset_<k>/`` subdirectory of their stage.

    The string value of each member is the filename or directory name on disk.
    """

    WORKFLOW_SETTINGS = "workflow_settings.yaml"
    """The settings yaml file which is written if the user runs using `presto train`
    (rather than `presto train-from-yaml <settings file>`). This provides a record of the
    settings used and allows easy re-running of the workflow later."""

    ENERGIES_AND_FORCES = "energy_and_force_data"
    """Directory containing energies and forces data files in HDF5 format."""

    TENSORBOARD = "tensorboard"
    """Directory containing TensorBoard logs."""

    TRAINING_METRICS = "metrics.txt"
    """File containing training metrics."""

    OFFXML = "bespoke_ff.offxml"
    """The output OpenFF ForceField file containing the bespoke parameters.
    One bespoke FF file is produced per training iteration."""

    SCATTER = "energies_and_forces.hdf5"
    """HDF5 file containing scatter data for energies and forces."""

    PDB_TRAJECTORY = "trajectory.pdb"
    """PDB trajectory file containing structures sampled during sampling."""

    METADYNAMICS_BIAS = "metadynamics_bias"
    """Directory containing metadynamics bias files."""

    LOSS_PLOT = "loss.png"
    """Plot of training and validation loss over training epochs."""

    ERROR_PLOT = "error_distributions.png"
    """Plot of error distributions for energies and forces. These are with
    respect to the 'test' data."""

    CORRELATION_PLOT = "correlation.png"
    """Plot of predicted vs reference energies and forces. These are with respect
    to the 'test' data."""

    FORCE_ERROR_BY_ATOM_INDEX_PLOT = "force_error_by_atom_index.png"
    """Plot of force errors broken down by atom index. These are with respect to
    the 'test' data."""

    PARAMETER_VALUES_PLOT = "parameter_values.png"
    """Plot of parameter values before and after fitting. Note that the 'before' force field
    is the one used for the initial sampling, after the MSM step."""

    PARAMETER_DIFFERENCES_PLOT = "parameter_differences.png"
    """Plot of parameter differences (fitted - initial) after fitting. Note that the 'initial' force field
    is the one used for the initial sampling, after the MSM step."""

    # PDB outputs for torsion-minimised structures
    ML_MINIMISED_PDB = "ml_minimised.pdb"
    """PDB file containing structures minimised using the machine-learned potential."""

    MM_MINIMISED_PDB = "mm_minimised.pdb"
    """PDB file containing structures minimised using the molecular mechanics potential."""

    TORSION_SAMPLING_PLOT = "torsion_sampling.png"
    """Plot of dihedral angles for all rotatable torsions during trajectories."""


# Output types that are generated per-molecule (one file/directory per molecule)
PER_MOLECULE_OUTPUT_TYPES: set[OutputType] = {
    OutputType.ENERGIES_AND_FORCES,
    OutputType.SCATTER,
    OutputType.PDB_TRAJECTORY,
    OutputType.METADYNAMICS_BIAS,
    OutputType.ERROR_PLOT,
    OutputType.CORRELATION_PLOT,
    OutputType.FORCE_ERROR_BY_ATOM_INDEX_PLOT,
    OutputType.PARAMETER_VALUES_PLOT,
    OutputType.PARAMETER_DIFFERENCES_PLOT,
    OutputType.ML_MINIMISED_PDB,
    OutputType.MM_MINIMISED_PDB,
    OutputType.TORSION_SAMPLING_PLOT,
}


class StageKind(StrEnum):
    """Enumeration of output directory names for each stage of the workflow."""

    BASE = ""
    INITIAL_STATISTICS = "initial_statistics"
    TESTING = "test_data"
    TRAINING = "training_iteration"
    PLOTS = "plots"


class DatasetSplit(StrEnum):
    """Names of the per-dataset subdirectories inside a stage directory."""

    TRAIN = "train_dataset"
    TEST = "test_dataset"


class WorkflowStatus(StrEnum):
    """The state of generated output for a workflow fit."""

    CLEAN = "clean"
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass(frozen=True)
class OutputStage:
    """A specific stage in the workflow output directory structure.

    A stage with a ``split`` and ``dataset`` is the subdirectory of its parent stage
    holding the outputs for one training or testing dataset, e.g.
    ``training_iteration_1/train_dataset_0``.
    """

    kind: StageKind
    index: int | None = None
    split: DatasetSplit | None = None
    dataset: int | None = None

    def __post_init__(self) -> None:
        """Check that the dataset fields are either both set or both unset."""
        if (self.split is None) != (self.dataset is None):
            raise ValueError("split and dataset must be set together")
        if self.dataset is not None and self.dataset < 0:
            raise ValueError(f"dataset must be non-negative, got {self.dataset}")

    @property
    def parent(self) -> "OutputStage":
        """The stage without its dataset subdirectory."""
        return OutputStage(self.kind, self.index)

    def __str__(self) -> str:
        """String representation."""
        stage = (
            f"{self.kind.value}_{self.index}"
            if self.index is not None
            else self.kind.value
        )
        if self.split is None:
            return stage
        return f"{stage}/{self.split.value}_{self.dataset}"


def _torsion_plot_if_trajectory(protocol: "SamplingSettings") -> set[OutputType]:
    """Predict a torsion sampling plot only for protocols which write trajectories."""
    if OutputType.PDB_TRAJECTORY in protocol.output_types:
        return {OutputType.TORSION_SAMPLING_PLOT}
    return set()


@dataclass
class WorkflowPathManager:
    """Manages paths for workflow outputs based on WorkflowSettings."""

    output_dir: Path
    n_iterations: int = 1
    n_mols: int = 1
    training_settings: "TrainingSettings | None" = None
    training_sampling_settings: "list[SamplingSettings]" = field(default_factory=list)
    testing_sampling_settings: "list[SamplingSettings]" = field(default_factory=list)

    def _generated_stages(self) -> list[OutputStage]:
        """Return every stage these settings predict whose directory exists."""
        return [
            stage
            for stage in self.outputs_by_stage
            if stage.kind is not StageKind.BASE
            and (
                (stage_path := self.get_stage_path(stage)).is_dir()
                or stage_path.is_symlink()
            )
        ]

    def _foreign_children(self, stage: OutputStage) -> list[Path]:
        """Return entries in a stage directory which these settings do not predict.

        Dataset subdirectories are searched too, unless they are symlinks, which are
        unlinked without touching their targets.
        """
        predicted = {
            path
            for paths in self.get_all_output_paths(only_if_exists=False)
            .get(stage, {})
            .values()
            for path in (paths if isinstance(paths, list) else [paths])
        }
        sub_stages = {
            self.get_stage_path(sub_stage): sub_stage
            for sub_stage in self.outputs_by_stage
            if sub_stage.split is not None and sub_stage.parent == stage
        }
        foreign = []
        for child in self.get_stage_path(stage).iterdir():
            if child in sub_stages and child.is_symlink():
                continue
            if child in sub_stages and child.is_dir():
                foreign.extend(self._foreign_children(sub_stages[child]))
            elif child not in predicted:
                foreign.append(child)
        return foreign

    @property
    def final_force_field(self) -> Path:
        """The bespoke force field written by the last training iteration."""
        return self.get_output_path(
            OutputStage(StageKind.TRAINING, self.n_iterations), OutputType.OFFXML
        )

    @property
    def status(self) -> WorkflowStatus:
        """Return whether generated fit output is clean, partial, or complete."""
        # is_symlink covers a broken link, which exists() reports as missing.
        if self.final_force_field.exists() or self.final_force_field.is_symlink():
            return WorkflowStatus.COMPLETE
        if self._generated_stages():
            return WorkflowStatus.PARTIAL
        return WorkflowStatus.CLEAN

    def require_clean(self) -> None:
        """Raise if generated output must be cleaned before starting a fit."""
        if (status := self.status) is not WorkflowStatus.CLEAN:
            raise RuntimeError(
                f"The output directory {self.output_dir} contains a {status.value} "
                "Presto fit. Run `presto clean` or use a new output directory before "
                "starting another fit."
            )

    @property
    def outputs_by_stage(self) -> dict[OutputStage, set[OutputType]]:
        """Return a dictionary mapping each stage to expected output types.

        Outputs which belong to one training or testing dataset are placed in that
        dataset's subdirectory stage, e.g. ``test_data/test_dataset_0``.
        """
        training_types = (
            self.training_settings.output_types if self.training_settings else set()
        )
        n_test = len(self.testing_sampling_settings)

        def test_scatter_stages(
            kind: StageKind, index: int | None = None
        ) -> dict[OutputStage, set[OutputType]]:
            return {
                OutputStage(kind, index, DatasetSplit.TEST, k): {OutputType.SCATTER}
                for k in range(n_test)
            }

        outputs_by_stage: dict[OutputStage, set[OutputType]] = {
            OutputStage(StageKind.BASE): {OutputType.WORKFLOW_SETTINGS},
            OutputStage(StageKind.TESTING): set(),
            **{
                OutputStage(StageKind.TESTING, None, DatasetSplit.TEST, k): (
                    protocol.output_types | {OutputType.ENERGIES_AND_FORCES}
                )
                for k, protocol in enumerate(self.testing_sampling_settings)
            },
            OutputStage(StageKind.INITIAL_STATISTICS): {OutputType.OFFXML},
            **test_scatter_stages(StageKind.INITIAL_STATISTICS),
        }
        for i in range(1, self.n_iterations + 1):
            outputs_by_stage[OutputStage(StageKind.TRAINING, i)] = training_types | {
                OutputType.OFFXML
            }
            for k, protocol in enumerate(self.training_sampling_settings):
                outputs_by_stage[
                    OutputStage(StageKind.TRAINING, i, DatasetSplit.TRAIN, k)
                ] = protocol.output_types | {OutputType.ENERGIES_AND_FORCES}
            outputs_by_stage.update(test_scatter_stages(StageKind.TRAINING, i))

        all_protocols = [
            *self.training_sampling_settings,
            *self.testing_sampling_settings,
        ]
        outputs_by_stage[OutputStage(StageKind.PLOTS)] = {
            OutputType.LOSS_PLOT,
            OutputType.ERROR_PLOT,
            OutputType.CORRELATION_PLOT,
            OutputType.FORCE_ERROR_BY_ATOM_INDEX_PLOT,
            OutputType.PARAMETER_VALUES_PLOT,
            OutputType.PARAMETER_DIFFERENCES_PLOT,
        }.union(*(_torsion_plot_if_trajectory(p) for p in all_protocols))
        for k, protocol in enumerate(self.testing_sampling_settings):
            outputs_by_stage[
                OutputStage(StageKind.PLOTS, None, DatasetSplit.TEST, k)
            ] = {
                OutputType.LOSS_PLOT,
                OutputType.ERROR_PLOT,
                OutputType.CORRELATION_PLOT,
                OutputType.FORCE_ERROR_BY_ATOM_INDEX_PLOT,
            } | _torsion_plot_if_trajectory(protocol)
        for k, protocol in enumerate(self.training_sampling_settings):
            outputs_by_stage[
                OutputStage(StageKind.PLOTS, None, DatasetSplit.TRAIN, k)
            ] = {OutputType.LOSS_PLOT} | _torsion_plot_if_trajectory(protocol)
        return outputs_by_stage

    def get_stage_path(self, stage: OutputStage) -> Path:
        """Get the directory path for a workflow stage."""
        if stage not in self.outputs_by_stage:
            raise ValueError(f"Unknown stage: {stage}")
        return self.output_dir / str(stage)

    def mk_stage_dir(self, stage: OutputStage) -> None:
        """Create the directory for a workflow stage."""
        path = self.get_stage_path(stage)
        path.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, stage: OutputStage, output_type: OutputType) -> Path:
        """Get the path for an output type in a stage.

        Note: For per-molecule output types (those in PER_MOLECULE_OUTPUT_TYPES),
        use get_output_path_for_mol instead.
        """
        if stage not in self.outputs_by_stage:
            raise ValueError(f"Unknown stage: {stage}")
        if output_type not in self.outputs_by_stage.get(stage, set()):
            raise ValueError(f"Output type {output_type} not expected in stage {stage}")

        return self.get_stage_path(stage) / output_type.value

    def get_output_path_for_mol(
        self, stage: OutputStage, output_type: OutputType, mol_idx: int
    ) -> Path:
        """Get the path for a per-molecule output type in a stage.

        Parameters
        ----------
        stage : OutputStage
            The workflow stage.
        output_type : OutputType
            The type of output (must be a per-molecule output type).
        mol_idx : int
            The molecule index.

        Returns:
        -------
        Path
            The path for the per-molecule output.

        Raises:
        ------
        ValueError
            If the output type is not a per-molecule output type, or if mol_idx
            is out of range.
        """
        if output_type not in PER_MOLECULE_OUTPUT_TYPES:
            raise ValueError(
                f"Output type {output_type} is not a per-molecule output type. "
                f"Use get_output_path instead."
            )
        if mol_idx < 0 or mol_idx >= self.n_mols:
            raise ValueError(
                f"mol_idx {mol_idx} is out of range for n_mols={self.n_mols}"
            )

        base_path = self.get_output_path(stage, output_type)

        # For directory-type outputs (no suffix), just append _mol{mol_idx}
        return get_mol_path(base_path, mol_idx)

    def get_all_output_paths(
        self, only_if_exists: bool = True
    ) -> dict[OutputStage, dict[OutputType, Path | list[Path]]]:
        """Get all expected output paths organized by stage.

        For per-molecule output types, returns a list of paths (one per molecule).
        For other output types, returns a single path.
        """
        all_paths: dict[OutputStage, dict[OutputType, Path | list[Path]]] = {}

        for stage in self.outputs_by_stage:
            paths_for_stage: dict[OutputType, Path | list[Path]] = {}
            for output_type in self.outputs_by_stage.get(stage, set()):
                if output_type in PER_MOLECULE_OUTPUT_TYPES:
                    # Collect paths for all molecules
                    mol_paths = []
                    for mol_idx in range(self.n_mols):
                        path = self.get_output_path_for_mol(stage, output_type, mol_idx)
                        if not only_if_exists or path.exists():
                            mol_paths.append(path)
                    if mol_paths:
                        paths_for_stage[output_type] = mol_paths
                else:
                    path = self.get_output_path(stage, output_type)
                    if not only_if_exists or path.exists():
                        paths_for_stage[output_type] = path

            if paths_for_stage:
                all_paths[stage] = paths_for_stage

        return all_paths

    def clean(self) -> None:
        """Remove every stage directory these settings predict.

        Stages beyond the predicted ones, such as an iteration past
        ``n_iterations``, are left alone, as is anything outside a stage directory.

        Raises:
        ------
        RuntimeError
            If a predicted stage directory holds a path these settings do not
            predict. Nothing is deleted in that case.
        """
        # Deleting a stage removes its dataset subdirectories with it.
        stages = [stage for stage in self._generated_stages() if stage.split is None]
        for stage in stages:
            stage_path = self.get_stage_path(stage)
            if stage_path.is_symlink() or not (
                foreign := self._foreign_children(stage)
            ):
                continue
            raise RuntimeError(
                f"{stage_path} contains files presto did not generate "
                f"({', '.join(child.name for child in foreign[:3])}). Delete it "
                "yourself or use a different output_dir."
            )
        for stage in stages:
            delete_path(self.get_stage_path(stage), recursive=True)


def delete_path(path: Path, recursive: bool = False) -> None:
    """Delete an output file or directory if it exists.

    Deletes the entire contents of a directory.

    Parameters
    ----------
    path : Path
        The path to delete.

    recursive : bool, optional
        Whether to delete directories recursively, by default False. If False, only
        empty directories will be deleted.
    """
    if not path.exists() and not path.is_symlink():
        return

    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        if recursive:
            for child in path.iterdir():
                delete_path(child, recursive=True)
        path.rmdir()  # Will only remove if empty
    else:
        path.unlink()


def get_mol_path(base_path: Path, mol_idx: int) -> Path:
    """Get a molecule-specific path from a base path.

    This function applies the standard naming convention for per-molecule outputs,
    where the molecule index is appended to the filename/directory name.

    Parameters
    ----------
    base_path : Path
        The base path (e.g., from get_output_path or output_paths dict).
    mol_idx : int
        The molecule index.

    Returns:
    -------
    Path
        The path with the molecule index appended.

    Examples:
    --------
    >>> get_mol_path(Path("output/scatter.hdf5"), 0)
    PosixPath('output/scatter_mol0.hdf5')
    >>> get_mol_path(Path("output/energy_data"), 1)
    PosixPath('output/energy_data_mol1')
    """
    if not base_path.suffix:
        # Directory-type output
        return base_path.parent / f"{base_path.name}_mol{mol_idx}"
    else:
        # File-type output
        return base_path.parent / f"{base_path.stem}_mol{mol_idx}{base_path.suffix}"
