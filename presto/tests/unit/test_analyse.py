"""Unit tests for the analyse module."""

from importlib import resources
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from openff.toolkit import ForceField, Molecule
from PIL import Image

from presto.analyse import (
    calculate_dihedrals_for_trajectory,
    get_mol_image_with_atom_idxs,
    load_force_fields,
    plot_all_ffs,
    plot_distributions_of_errors,
    plot_energy_correlation,
    plot_error_statistics,
    plot_ff_differences,
    plot_ff_values,
    plot_force_error_by_atom_idx,
    plot_loss,
    plot_mean_errors,
    plot_rmse_of_errors,
    plot_torsion_sampling,
    read_errors,
    read_losses,
)
from presto.outputs import DatasetSplit, OutputStage, OutputType, StageKind
from presto.settings import WorkflowSettings
from presto.writers import SCATTER_KEYS, combined_metrics_columns, metrics_columns


# Use non-interactive backend for tests
matplotlib.use("Agg")


def get_example_data_path() -> Path:
    """Get path to example data using importlib.resources."""
    return Path(
        resources.files("presto.data") / "example_run_single"  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def workflow_settings() -> WorkflowSettings:
    """Load workflow settings from the example data.

    The example data has output_dir: . so we override it to the actual
    example data directory path.
    """
    example_data_dir = get_example_data_path()
    settings = WorkflowSettings.from_yaml(example_data_dir / "workflow_settings.yaml")
    settings.output_dir = example_data_dir
    return settings


@pytest.fixture(scope="module")
def path_manager(workflow_settings: WorkflowSettings):
    """Get path manager from workflow settings."""
    return workflow_settings.get_path_manager()


@pytest.fixture(scope="module")
def test_molecule(workflow_settings: WorkflowSettings) -> Molecule:
    """Get the test molecule from workflow settings."""
    return workflow_settings.param_settings.openff_molecules[0]


@pytest.fixture(scope="module")
def scatter_paths(path_manager) -> dict[int, Path]:
    """Get scatter paths for testing using path manager.

    The example data uses the _mol{idx} naming convention for per-molecule outputs.
    """
    paths: dict[int, Path] = {}

    # Initial statistics (iteration 0)
    initial_stage = OutputStage(
        StageKind.INITIAL_STATISTICS, None, DatasetSplit.TEST, 0
    )
    paths[0] = path_manager.get_output_path_for_mol(
        initial_stage, OutputType.SCATTER, mol_idx=0
    )

    # Training iterations
    for i in range(1, path_manager.n_iterations + 1):
        training_stage = OutputStage(StageKind.TRAINING, i, DatasetSplit.TEST, 0)
        paths[i] = path_manager.get_output_path_for_mol(
            training_stage, OutputType.SCATTER, mol_idx=0
        )

    return paths


@pytest.fixture(scope="module")
def training_metrics_paths(path_manager) -> dict[int, Path]:
    """Get training metrics paths for testing using path manager."""
    paths: dict[int, Path] = {}
    for i in range(1, path_manager.n_iterations + 1):
        training_stage = OutputStage(StageKind.TRAINING, i)
        paths[i] = path_manager.get_output_path(
            training_stage, OutputType.TRAINING_METRICS
        )

    return paths


@pytest.fixture(scope="module")
def offxml_paths(path_manager) -> dict[int, Path]:
    """Get offxml paths for testing using path manager."""
    paths: dict[int, Path] = {}

    # Initial statistics (iteration 0)
    initial_stage = OutputStage(StageKind.INITIAL_STATISTICS)
    paths[0] = path_manager.get_output_path(initial_stage, OutputType.OFFXML)

    # Training iterations
    for i in range(1, path_manager.n_iterations + 1):
        training_stage = OutputStage(StageKind.TRAINING, i)
        paths[i] = path_manager.get_output_path(training_stage, OutputType.OFFXML)

    return paths


@pytest.fixture(scope="module")
def errors_data(scatter_paths: dict[int, Path]) -> dict:
    """Load errors data from scatter files."""
    return read_errors(scatter_paths)


@pytest.fixture(scope="module")
def losses_data(training_metrics_paths: dict[int, Path]) -> pd.DataFrame:
    """Load losses data from training metrics files."""
    return read_losses(training_metrics_paths)


@pytest.fixture(scope="module")
def force_fields(offxml_paths: dict[int, Path]) -> dict[int, ForceField]:
    """Load force fields from offxml files."""
    return load_force_fields(offxml_paths)


class TestReadErrors:
    """Tests for read_errors function."""

    def test_read_errors_returns_correct_keys(
        self, scatter_paths: dict[int, Path]
    ) -> None:
        """Test that read_errors returns the expected keys."""
        errors = read_errors(scatter_paths)

        expected_keys = {
            "energy_reference",
            "energy_predicted",
            "energy_differences",
            "forces_reference",
            "forces_predicted",
            "forces_differences",
            "n_atoms",
            "n_conformers",
        }
        assert set(errors.keys()) == expected_keys

    def test_read_errors_returns_correct_iterations(
        self, scatter_paths: dict[int, Path], path_manager
    ) -> None:
        """Test that read_errors returns data for correct iterations."""
        errors = read_errors(scatter_paths)
        expected_iterations = set(range(path_manager.n_iterations + 1))

        for key in ["energy_reference", "energy_predicted", "energy_differences"]:
            assert set(errors[key].keys()) == expected_iterations

    def test_read_errors_returns_numpy_arrays(
        self, scatter_paths: dict[int, Path]
    ) -> None:
        """Test that read_errors returns numpy arrays."""
        errors = read_errors(scatter_paths)

        for key in ["energy_reference", "energy_predicted", "energy_differences"]:
            for arr in errors[key].values():
                assert isinstance(arr, np.ndarray)

    def test_read_errors_energy_shape(self, errors_data: dict) -> None:
        """Test that energy arrays have expected shape."""
        n_conformers = errors_data["n_conformers"]
        for arr in errors_data["energy_reference"].values():
            assert arr.shape == (n_conformers,)

    def test_read_errors_forces_shape(self, errors_data: dict) -> None:
        """Test that forces arrays have expected shape."""
        n_atoms = errors_data["n_atoms"]
        n_conformers = errors_data["n_conformers"]
        for arr in errors_data["forces_reference"].values():
            assert arr.shape == (n_conformers * n_atoms, 3)


class TestReadLosses:
    """Tests for read_losses function."""

    def test_read_losses_returns_dataframe(
        self, training_metrics_paths: dict[int, Path]
    ) -> None:
        """Test that read_losses returns a pandas DataFrame."""
        losses = read_losses(training_metrics_paths)
        assert isinstance(losses, pd.DataFrame)

    def test_read_losses_has_iteration_column(self, losses_data: pd.DataFrame) -> None:
        """Test that losses DataFrame has iteration column."""
        assert "iteration" in losses_data.columns

    def test_read_losses_has_loss_columns(self, losses_data: pd.DataFrame) -> None:
        """Test that losses DataFrame has expected loss columns."""
        loss_columns = [c for c in losses_data.columns if c.startswith("loss_")]
        assert len(loss_columns) > 0

        # Check for train and test losses
        train_cols = [c for c in loss_columns if "train" in c]
        test_cols = [c for c in loss_columns if "test" in c]
        assert len(train_cols) > 0
        assert len(test_cols) > 0

    def test_read_losses_columns_from_header(self, losses_data: pd.DataFrame) -> None:
        """Columns are named by the metrics file header."""
        assert list(losses_data.columns) == [*metrics_columns(1, 1), "iteration"]

    def test_read_losses_multiple_iterations(
        self, losses_data: pd.DataFrame, path_manager
    ) -> None:
        """Test that losses DataFrame contains multiple iterations."""
        iterations = losses_data["iteration"].unique()
        assert len(iterations) == path_manager.n_iterations
        assert set(iterations) == set(range(1, path_manager.n_iterations + 1))


class TestLoadForceFields:
    """Tests for load_force_fields function."""

    def test_load_force_fields_returns_dict(
        self, offxml_paths: dict[int, Path]
    ) -> None:
        """Test that load_force_fields returns a dictionary."""
        ffs = load_force_fields(offxml_paths)
        assert isinstance(ffs, dict)

    def test_load_force_fields_correct_iterations(
        self, offxml_paths: dict[int, Path], path_manager
    ) -> None:
        """Test that force fields are loaded for correct iterations."""
        ffs = load_force_fields(offxml_paths)
        expected_iterations = set(range(path_manager.n_iterations + 1))
        assert set(ffs.keys()) == expected_iterations

    def test_load_force_fields_returns_forcefield_objects(
        self, force_fields: dict[int, ForceField]
    ) -> None:
        """Test that loaded objects are ForceField instances."""
        for ff in force_fields.values():
            assert isinstance(ff, ForceField)


class TestGetMolImageWithAtomIdxs:
    """Tests for get_mol_image_with_atom_idxs function."""

    def test_returns_pil_image(self, test_molecule: Molecule) -> None:
        """Test that function returns a PIL Image."""
        img = get_mol_image_with_atom_idxs(test_molecule)
        assert isinstance(img, Image.Image)

    def test_respects_dimensions(self, test_molecule: Molecule) -> None:
        """Test that function respects width and height parameters."""
        width, height = 400, 300
        img = get_mol_image_with_atom_idxs(test_molecule, width=width, height=height)
        assert img.size == (width, height)

    def test_default_dimensions(self, test_molecule: Molecule) -> None:
        """Test default dimensions."""
        img = get_mol_image_with_atom_idxs(test_molecule)
        assert img.size == (300, 300)


class TestPlotLoss:
    """Tests for plot_loss function."""

    def test_plot_loss_creates_plot(self, losses_data: pd.DataFrame) -> None:
        """Test that plot_loss creates a plot without errors."""
        fig, ax = plt.subplots()
        plot_loss(fig, ax, losses_data, combined_metrics_columns())
        plt.close(fig)

    def test_plot_loss_has_labels(self, losses_data: pd.DataFrame) -> None:
        """Test that plot has axis labels."""
        fig, ax = plt.subplots()
        plot_loss(fig, ax, losses_data, combined_metrics_columns())
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_plot_loss_has_legend(self, losses_data: pd.DataFrame) -> None:
        """Test that plot has a legend."""
        fig, ax = plt.subplots()
        plot_loss(fig, ax, losses_data, combined_metrics_columns())
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)


class TestPlotEnergyCorrelation:
    """Tests for plot_energy_correlation function."""

    def test_plot_energy_correlation_creates_plot(self, errors_data: dict) -> None:
        """Test that plot_energy_correlation creates a plot without errors."""
        fig, ax = plt.subplots()
        plot_energy_correlation(
            fig,
            ax,
            errors_data["energy_reference"],
            errors_data["energy_predicted"],
        )
        plt.close(fig)

    def test_plot_energy_correlation_has_labels(self, errors_data: dict) -> None:
        """Test that plot has axis labels and title."""
        fig, ax = plt.subplots()
        plot_energy_correlation(
            fig,
            ax,
            errors_data["energy_reference"],
            errors_data["energy_predicted"],
        )
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""
        plt.close(fig)


class TestPlotForceErrorByAtomIdx:
    """Tests for plot_force_error_by_atom_idx function."""

    def test_plot_force_error_creates_plot(
        self, errors_data: dict, test_molecule: Molecule
    ) -> None:
        """Test that plot_force_error_by_atom_idx creates a plot without errors."""
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_force_error_by_atom_idx(
            fig, ax, errors_data["forces_differences"], test_molecule
        )
        plt.close(fig)

    def test_plot_force_error_has_labels(
        self, errors_data: dict, test_molecule: Molecule
    ) -> None:
        """Test that plot has axis labels."""
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_force_error_by_atom_idx(
            fig, ax, errors_data["forces_differences"], test_molecule
        )
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)


class TestPlotDistributionsOfErrors:
    """Tests for plot_distributions_of_errors function."""

    def test_plot_energy_distribution(self, errors_data: dict) -> None:
        """Test plotting energy error distribution."""
        fig, ax = plt.subplots()
        plot_distributions_of_errors(
            fig, ax, errors_data["energy_differences"], "energy"
        )
        plt.close(fig)

    def test_plot_force_distribution(self, errors_data: dict) -> None:
        """Test plotting force error distribution."""
        fig, ax = plt.subplots()
        plot_distributions_of_errors(
            fig, ax, errors_data["forces_differences"], "force"
        )
        plt.close(fig)

    def test_plot_with_missing_iterations(self, errors_data: dict) -> None:
        """Iterations missing from the keys keep their real labels."""
        energies = errors_data["energy_differences"]
        fig, ax = plt.subplots()
        plot_distributions_of_errors(
            fig, ax, {0: energies[0], 5: energies[2]}, "energy"
        )
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == ["Iteration 0", "Iteration 5"]
        plt.close(fig)

    def test_plot_has_legend(self, errors_data: dict) -> None:
        """Test that distribution plot has a legend."""
        fig, ax = plt.subplots()
        plot_distributions_of_errors(
            fig, ax, errors_data["energy_differences"], "energy"
        )
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)


class TestPlotMeanErrors:
    """Tests for plot_mean_errors function."""

    def test_plot_mean_energy_errors(self, errors_data: dict) -> None:
        """Test plotting mean energy errors."""
        fig, ax = plt.subplots()
        plot_mean_errors(fig, ax, errors_data["energy_differences"], "energy")
        plt.close(fig)

    def test_plot_mean_force_errors(self, errors_data: dict) -> None:
        """Test plotting mean force errors."""
        fig, ax = plt.subplots()
        plot_mean_errors(fig, ax, errors_data["forces_differences"], "force")
        plt.close(fig)


class TestPlotRmseOfErrors:
    """Tests for plot_rmse_of_errors function."""

    def test_plot_rmse_energy_errors(self, errors_data: dict) -> None:
        """Test plotting RMSE of energy errors."""
        fig, ax = plt.subplots()
        plot_rmse_of_errors(fig, ax, errors_data["energy_differences"], "energy")
        plt.close(fig)

    def test_plot_rmse_force_errors(self, errors_data: dict) -> None:
        """Test plotting RMSE of force errors."""
        fig, ax = plt.subplots()
        plot_rmse_of_errors(fig, ax, errors_data["forces_differences"], "force")
        plt.close(fig)


class TestPlotErrorStatistics:
    """Tests for plot_error_statistics function."""

    def test_plot_error_statistics_creates_plots(self, errors_data: dict) -> None:
        """Test that plot_error_statistics creates plots without errors."""
        fig, axs = plt.subplots(2, 2, figsize=(13, 12))
        plot_error_statistics(fig, axs, errors_data)
        plt.close(fig)


class TestPlotFfValues:
    """Tests for plot_ff_values function."""

    def test_plot_bond_values(
        self, force_fields: dict[int, ForceField], test_molecule: Molecule
    ) -> None:
        """Test plotting bond parameter values."""
        fig, ax = plt.subplots()
        plot_ff_values(fig, ax, force_fields, test_molecule, "Bonds", "length")
        plt.close(fig)

    def test_plot_angle_values(
        self, force_fields: dict[int, ForceField], test_molecule: Molecule
    ) -> None:
        """Test plotting angle parameter values."""
        fig, ax = plt.subplots()
        plot_ff_values(fig, ax, force_fields, test_molecule, "Angles", "angle")
        plt.close(fig)

    def test_plot_torsion_values(
        self, force_fields: dict[int, ForceField], test_molecule: Molecule
    ) -> None:
        """Test plotting proper torsion parameter values."""
        fig, ax = plt.subplots()
        plot_ff_values(fig, ax, force_fields, test_molecule, "ProperTorsions", "k1")
        plt.close(fig)

    def test_plot_with_missing_iterations(
        self, force_fields: dict[int, ForceField], test_molecule: Molecule
    ) -> None:
        """Iterations missing from the keys keep their real labels."""
        fig, ax = plt.subplots()
        plot_ff_values(
            fig,
            ax,
            {0: force_fields[0], 5: force_fields[2]},
            test_molecule,
            "Bonds",
            "length",
        )
        _, labels = ax.get_legend_handles_labels()
        assert set(labels) == {"Iteration 0", "Iteration 5"}
        plt.close(fig)


class TestPlotFfDifferences:
    """Tests for plot_ff_differences function."""

    def test_plot_bond_differences(
        self, force_fields: dict[int, ForceField], test_molecule: Molecule
    ) -> None:
        """Test plotting bond parameter differences."""
        fig, ax = plt.subplots()
        differences = plot_ff_differences(
            fig, ax, force_fields, test_molecule, "Bonds", "length"
        )
        assert isinstance(differences, dict)
        plt.close(fig)

    def test_plot_angle_differences(
        self, force_fields: dict[int, ForceField], test_molecule: Molecule
    ) -> None:
        """Test plotting angle parameter differences."""
        fig, ax = plt.subplots()
        differences = plot_ff_differences(
            fig, ax, force_fields, test_molecule, "Angles", "angle"
        )
        assert isinstance(differences, dict)
        plt.close(fig)


class TestPlotAllFfs:
    """Tests for plot_all_ffs function."""

    def test_plot_all_ffs_values(
        self, force_fields: dict[int, ForceField], test_molecule: Molecule
    ) -> None:
        """Test plotting all force field parameter values."""
        fig, axs = plot_all_ffs(force_fields, test_molecule, "values")
        assert fig is not None
        assert axs is not None
        plt.close(fig)

    def test_plot_all_ffs_differences(
        self, force_fields: dict[int, ForceField], test_molecule: Molecule
    ) -> None:
        """Test plotting all force field parameter differences."""
        fig, axs = plot_all_ffs(force_fields, test_molecule, "differences")
        assert fig is not None
        assert axs is not None
        plt.close(fig)


class TestCalculateDihedralsForTrajectory:
    """Tests for calculate_dihedrals_for_trajectory function."""

    def test_calculate_dihedrals_for_trajectory(self, tmp_path: Path) -> None:
        """Test calculating dihedrals for a simple trajectory using MDTraj."""
        import mdtraj

        # Create a simple 3-frame trajectory with proper non-colinear geometry
        n_frames = 3
        n_atoms = 4
        topology = mdtraj.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("MOL", chain)
        for _ in range(n_atoms):
            topology.add_atom("C", mdtraj.element.carbon, residue)

        # Create trajectory positions (in nanometers for MDTraj)
        positions = np.zeros((n_frames, n_atoms, 3))

        # Frame 0: zig-zag pattern
        positions[0] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.20, 0.10, 0.0],
                [0.35, 0.10, 0.0],
            ]
        )

        # Frame 1: rotated around central bond
        positions[1] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.20, 0.10, 0.0],
                [0.35, 0.10, 0.05],
            ]
        )

        # Frame 2: more rotated
        positions[2] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.20, 0.10, 0.0],
                [0.35, 0.10, 0.10],
            ]
        )

        # Create trajectory and save as PDB
        traj = mdtraj.Trajectory(positions, topology)
        pdb_path = tmp_path / "test_trajectory.pdb"
        traj.save_pdb(str(pdb_path))

        torsions = {(1, 2): (0, 1, 2, 3)}
        dihedrals = calculate_dihedrals_for_trajectory(pdb_path, torsions)

        assert (0, 1, 2, 3) in dihedrals
        assert len(dihedrals[(0, 1, 2, 3)]) == n_frames
        # Check that angles change across frames
        angles = dihedrals[(0, 1, 2, 3)]
        assert not np.allclose(angles[0], angles[1])


class TestPlotTorsionSampling:
    """Tests for plot_torsion_sampling function."""

    def test_plot_torsion_sampling_creates_plot(self, test_molecule: Molecule) -> None:
        """Test that plot_torsion_sampling creates a plot without errors."""
        # Create mock dihedral data
        n_frames = 10
        dihedrals_by_iteration = {
            0: {
                (0, 1, 2, 3): np.linspace(-180, 180, n_frames),
                (1, 2, 3, 4): np.linspace(0, 90, n_frames),
            },
            1: {
                (0, 1, 2, 3): np.linspace(-90, 90, n_frames),
                (1, 2, 3, 4): np.linspace(45, 135, n_frames),
            },
        }

        # Create subplots for 2 torsions
        fig, axs = plt.subplots(1, 2, figsize=(16, 5), squeeze=False)
        plot_torsion_sampling(fig, axs, dihedrals_by_iteration, test_molecule)

        # Check that each subplot has content
        for ax in axs.flat:
            assert ax.get_xlabel() != "" or not ax.get_visible()
        plt.close(fig)


def test_add_legend_if_labels_no_labels():
    """Add legend if labels no labels."""
    from presto.analyse import _add_legend_if_labels

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])  # No label
    _add_legend_if_labels(ax)
    assert ax.get_legend() is None
    plt.close(fig)


def test_plot_loss_with_regularisation(losses_data):
    """Plot loss with regularisation."""
    from presto.analyse import plot_loss

    fig, ax = plt.subplots()
    # Add a regularisation loss column
    losses_with_reg = losses_data.copy()
    losses_with_reg["loss_test_regularisation"] = 0.1
    plot_loss(fig, ax, losses_with_reg, combined_metrics_columns())
    # Check that it didn't crash
    plt.close(fig)


def test_plot_ff_differences_empty(force_fields, test_molecule):
    """Plot ff differences empty."""
    from presto.analyse import plot_ff_differences

    fig, ax = plt.subplots()
    # Use a parameter key that doesn't exist
    res = plot_ff_differences(
        fig, ax, force_fields, test_molecule, "Bonds", "non-existent"
    )
    assert res == {}
    plt.close(fig)


def test_plot_ff_differences_mismatched_ids(test_molecule):
    """Plot ff differences mismatched ids."""
    from unittest.mock import MagicMock, PropertyMock

    from presto.analyse import plot_ff_differences

    # Mock parameter objects
    p1 = MagicMock()
    type(p1).id = PropertyMock(return_value="b1")
    p2 = MagicMock()
    type(p2).id = PropertyMock(return_value="b2")

    # Mock force fields with different IDs
    ff1 = MagicMock()
    ff1.label_molecules.return_value = [{"Bonds": {"(0,1)": p1}}]
    ff2 = MagicMock()
    ff2.label_molecules.return_value = [{"Bonds": {"(0,1)": p2}}]

    ffs = {0: ff1, 1: ff2}
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Force field has different Bonds ids"):
        plot_ff_differences(fig, ax, ffs, test_molecule, "Bonds", "k")
    plt.close(fig)


def test_plot_ff_values_empty(force_fields, test_molecule):
    """Plot ff values empty."""
    from presto.analyse import plot_ff_values

    fig, ax = plt.subplots()
    # Use a potential type that doesn't exist
    plot_ff_values(fig, ax, force_fields, test_molecule, "vdW", "epsilon")
    plt.close(fig)


def test_plot_ff_values_mismatched_ids(test_molecule):
    """Plot ff values mismatched ids."""
    from unittest.mock import MagicMock, PropertyMock

    from presto.analyse import plot_ff_values

    # Mock parameter objects
    p1 = MagicMock()
    type(p1).id = PropertyMock(return_value="b1")
    val1 = MagicMock()
    val1.units = "kcal/mol"
    # val1 / units.unit.degrees should work
    val1.__truediv__.return_value = 1.0
    p1.to_dict.return_value = {"k": val1}

    p2 = MagicMock()
    type(p2).id = PropertyMock(return_value="b2")
    val2 = MagicMock()
    val2.units = "kcal/mol"
    val2.__truediv__.return_value = 1.1
    p2.to_dict.return_value = {"k": val2}

    # Mock force fields with different IDs
    ff1 = MagicMock()
    ff1.label_molecules.return_value = [{"Bonds": {"(0,1)": p1}}]
    ff2 = MagicMock()
    ff2.label_molecules.return_value = [{"Bonds": {"(0,1)": p2}}]

    ffs = {0: ff1, 1: ff2}
    fig, ax = plt.subplots()
    with pytest.raises(
        ValueError, match="Force field at iteration 1 has different Bonds ids"
    ):
        plot_ff_values(fig, ax, ffs, test_molecule, "Bonds", "k")
    plt.close(fig)


def test_read_errors_multiple_iterations(tmp_path):
    """Read errors multiple iterations."""
    import h5py

    from presto.analyse import read_errors

    path1 = tmp_path / "iter0.h5"
    path2 = tmp_path / "iter1.h5"

    for p in [path1, path2]:
        with h5py.File(p, "w") as f:
            f.create_dataset("energy_reference", data=[1.0])
            f.create_dataset("energy_predicted", data=[1.1])
            f.create_dataset("energy_differences", data=[0.1])
            f.create_dataset("forces_reference", data=[[[0.1, 0.1, 0.1]]])
            f.create_dataset("forces_predicted", data=[[[0.11, 0.11, 0.11]]])
            f.create_dataset("forces_differences", data=[[[0.01, 0.01, 0.01]]])
            f.attrs["n_atoms"] = 1
            f.attrs["n_conformers"] = 1

    paths = {0: path1, 1: path2}
    res = read_errors(paths)
    assert len(res["energy_reference"]) == 2


def _copy_example_run(tmp_path: Path) -> WorkflowSettings:
    """Copy the example run, without its plots, and point its settings at the copy."""
    import shutil

    output_dir = tmp_path / "run"
    shutil.copytree(
        get_example_data_path(), output_dir, ignore=shutil.ignore_patterns("plots")
    )
    settings = WorkflowSettings.from_yaml(output_dir / "workflow_settings.yaml")
    settings.output_dir = output_dir
    return settings


def _plot_paths(output_dir: Path) -> set[str]:
    return {
        str(path.relative_to(output_dir / "plots"))
        for path in (output_dir / "plots").rglob("*.png")
    }


class TestAnalyseWorkflow:
    """Tests for analysing a whole run, combined and per dataset."""

    def test_single_dataset(self, tmp_path: Path) -> None:
        """Combined plots go in plots/ and per-dataset plots in subdirectories."""
        from presto.analyse import analyse_workflow

        settings = _copy_example_run(tmp_path)
        analyse_workflow(settings)

        # The example's test trajectory is unreadable, so it has no torsion plot
        assert _plot_paths(settings.output_dir) == {
            "loss.png",
            "error_distributions_mol0.png",
            "correlation_mol0.png",
            "force_error_by_atom_index_mol0.png",
            "parameter_values_mol0.png",
            "parameter_differences_mol0.png",
            "torsion_sampling_mol0.png",
            "test_dataset_0/loss.png",
            "test_dataset_0/error_distributions_mol0.png",
            "test_dataset_0/correlation_mol0.png",
            "test_dataset_0/force_error_by_atom_index_mol0.png",
            "train_dataset_0/loss.png",
            "train_dataset_0/torsion_sampling_mol0.png",
        }

    def test_multiple_datasets(self, tmp_path: Path) -> None:
        """Every dataset gets its own plots, alongside the plots pooling all of them."""
        import shutil

        from presto.analyse import analyse_workflow

        settings = _copy_example_run(tmp_path)
        output_dir = settings.output_dir
        train, test = (
            settings.training_sampling_settings[0],
            settings.testing_sampling_settings[0],
        )
        settings.training_sampling_settings = [train, train.model_copy()]
        settings.testing_sampling_settings = [test, test.model_copy(), test]
        for stage in ("test_data", "initial_statistics", "training_iteration_1"):
            for k in (1, 2):
                shutil.copytree(
                    output_dir / stage / "test_dataset_0",
                    output_dir / stage / f"test_dataset_{k}",
                )
        for i in (1, 2):
            iteration = output_dir / f"training_iteration_{i}"
            shutil.copytree(
                iteration / "train_dataset_0", iteration / "train_dataset_1"
            )
            # Iteration 2 lacks test dataset 2's scatter, which is skipped
            if i == 2:
                shutil.copytree(
                    iteration / "test_dataset_0", iteration / "test_dataset_1"
                )
            metrics = iteration / "metrics.txt"
            _header, *rows = metrics.read_text().splitlines()
            rows = [" ".join(row.split()[:6] + ["1.0"] * 10) for row in rows]
            metrics.write_text(
                "\n".join([" ".join(metrics_columns(2, 3)), *rows]) + "\n"
            )

        analyse_workflow(settings)

        plots = _plot_paths(output_dir)
        for k in range(3):
            assert {
                f"test_dataset_{k}/loss.png",
                f"test_dataset_{k}/error_distributions_mol0.png",
                f"test_dataset_{k}/correlation_mol0.png",
                f"test_dataset_{k}/force_error_by_atom_index_mol0.png",
            } <= plots
        for k in range(2):
            assert {
                f"train_dataset_{k}/loss.png",
                f"train_dataset_{k}/torsion_sampling_mol0.png",
            } <= plots
        assert {
            "loss.png",
            "error_distributions_mol0.png",
            "torsion_sampling_mol0.png",
            "parameter_values_mol0.png",
        } <= plots


def test_pool_errors_concatenates_each_iteration():
    """Pooled errors join every dataset's values, per iteration, and drop scalars."""
    from presto.analyse import _pool_errors

    def errors(value, iterations):
        return {
            **{
                key: {i: np.full(2, value + i, dtype=float) for i in iterations}
                for key in SCATTER_KEYS
            },
            "n_atoms": 3,
            "n_conformers": 2,
        }

    pooled = _pool_errors([errors(0.0, [0, 1, 2]), errors(10.0, [0, 2])])

    assert set(pooled) == set(SCATTER_KEYS)
    energies = pooled["energy_differences"]
    assert list(energies) == [0, 1, 2]
    np.testing.assert_array_equal(energies[0], [0, 0, 10, 10])
    np.testing.assert_array_equal(energies[1], [1, 1])
    np.testing.assert_array_equal(energies[2], [2, 2, 12, 12])


def test_pool_dihedrals_concatenates_each_torsion():
    """Pooled dihedrals join every trajectory's angles for each torsion."""
    from presto.analyse import _pool_dihedrals

    pooled = _pool_dihedrals(
        [
            {(0, 1, 2, 3): np.array([1.0]), (1, 2, 3, 4): np.array([2.0])},
            {(0, 1, 2, 3): np.array([3.0, 4.0]), (1, 2, 3, 4): np.array([5.0])},
        ]
    )
    np.testing.assert_array_equal(pooled[(0, 1, 2, 3)], [1, 3, 4])
    np.testing.assert_array_equal(pooled[(1, 2, 3, 4)], [2, 5])


def test_existing_paths_keyed_by_real_iteration(tmp_path, path_manager):
    """A missing iteration is skipped, not renumbered."""
    from presto.analyse import _existing_paths_for_mol, _iteration_stages
    from presto.outputs import WorkflowPathManager

    manager = WorkflowPathManager(
        output_dir=tmp_path,
        n_iterations=3,
        testing_sampling_settings=path_manager.testing_sampling_settings,
    )
    stages = _iteration_stages(manager, DatasetSplit.TEST, 0)
    for i in (0, 2, 3):
        path = manager.get_output_path_for_mol(stages[i], OutputType.SCATTER, 0)
        path.parent.mkdir(parents=True)
        path.touch()

    paths = _existing_paths_for_mol(manager, stages, OutputType.SCATTER, 0)

    assert list(paths) == [0, 2, 3]
    assert paths[2].parent.name == "test_dataset_0"
    assert paths[2].parent.parent.name == "training_iteration_2"
