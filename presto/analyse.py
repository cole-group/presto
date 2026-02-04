"""Functionality for analysing the results of a presto run."""

import io
from pathlib import Path
from typing import Any, Literal

import h5py
import loguru
import mdtraj
import numpy as np
import numpy.typing as npt
import openff.units as units
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from openff.toolkit import ForceField, Molecule
from PIL import Image
from rdkit.Chem import Draw
from rich.progress import track

from .find_torsions import (
    _TORSIONS_TO_EXCLUDE_SMARTS,
    _TORSIONS_TO_INCLUDE_SMARTS,
    get_rot_torsions_by_rot_bond,
)
from .loss import LossRecord
from .outputs import OutputStage, OutputType, StageKind, get_mol_path
from .settings import WorkflowSettings

logger = loguru.logger

PLT_STYLE = "ggplot"


def _add_legend_if_labels(ax: Axes, **kwargs: Any) -> None:
    """Add a legend to the axes only if there are labeled artists."""
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)


POTENTIAL_KEYS = Literal[
    "Bonds", "Angles", "ProperTorsions", "ImproperTorsions", "vdW", "Electrostatics"
]


def read_errors(
    paths_by_iter: dict[int, Path],
) -> dict[str, dict[int, npt.NDArray[np.float64]]]:
    """Read all energy and force data from the HDF5 files.

    Returns:
        Dictionary with keys: 'energy_reference', 'energy_predicted', 'energy_differences',
        'forces_reference', 'forces_predicted', 'forces_differences'.
        Each value is a dict mapping iteration number to numpy array.
    """

    results: dict[str, dict[int, npt.NDArray[np.float64]]] = {
        "energy_reference": {},
        "energy_predicted": {},
        "energy_differences": {},
        "forces_reference": {},
        "forces_predicted": {},
        "forces_differences": {},
    }

    for i, filepath in paths_by_iter.items():
        with h5py.File(filepath, "r") as f:
            results["energy_reference"][i] = f["energy_reference"][:]
            results["energy_predicted"][i] = f["energy_predicted"][:]
            results["energy_differences"][i] = f["energy_differences"][:]
            results["forces_reference"][i] = f["forces_reference"][:]
            results["forces_predicted"][i] = f["forces_predicted"][:]
            results["forces_differences"][i] = f["forces_differences"][:]
            results["n_atoms"] = f.attrs["n_atoms"]
            results["n_conformers"] = f.attrs["n_conformers"]

    return results


def read_losses(paths_by_iter: dict[int, Path]) -> pd.DataFrame:
    df_rows = []
    names = []
    for loss_type in ["train", "test"]:
        for field in LossRecord._fields:
            names.append(f"loss_{loss_type}_{field}")

    for i, loss_datafile in paths_by_iter.items():
        df = pd.read_csv(
            loss_datafile,
            sep=r"\s+",
            header=None,
            names=names,
        )
        df_rows.append(df.assign(iteration=i))

    return pd.concat(df_rows, ignore_index=True)


def load_force_fields(paths_by_iter: dict[int, Path]) -> dict[int, str]:
    """Load the .offxml files from the given paths."""
    return {i: ForceField(p) for i, p in paths_by_iter.items()}


def plot_loss(fig: Figure, ax: Axes, losses: pd.DataFrame) -> None:
    # Colour by iteration - full line for train, dotted for test
    iterations = losses["iteration"].unique()
    first_iteration = iterations[0]
    for i in iterations:
        loss_names = [name for name in losses.columns if name.startswith("loss_")]
        for loss_name in loss_names:
            if loss_name == "loss_test_regularisation":
                continue  # Skip regularisation loss as not meaningful
            linestyle = "-" if "train" in loss_name else "--"
            color_idx = (
                0 if "energy" in loss_name else 1 if "forces" in loss_name else 2
            )
            label = loss_name.split("_")[1] + " " + loss_name.split("_")[-1]
            ax.plot(
                losses[losses["iteration"] == i].index,
                losses[losses["iteration"] == i][loss_name],
                label=(
                    label if i == first_iteration else None
                ),  # Don't repeat labels for new iterations
                color=f"C{color_idx}",
                linestyle=linestyle,
            )

        # If this isn't the last iteration, add a vertical line to separate iterations # and label it
        if i != iterations[-1]:
            ax.axvline(
                x=losses[losses["iteration"] == i].index[-1] + 0.5,
                color="black",
                linestyle=":",
                alpha=0.5,
            )
            ax.text(
                losses[losses["iteration"] == i].index[-1] + 0.5,
                ax.get_ylim()[1] * 0.9,
                f"Iteration {i}",
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    _add_legend_if_labels(ax, bbox_to_anchor=(1.05, 1), loc="upper left")


def plot_energy_correlation(
    fig: Figure,
    ax: Axes,
    reference: dict[int, npt.NDArray[np.float64]],
    predicted: dict[int, npt.NDArray[np.float64]],
) -> None:
    """Plot the correlation between reference and predicted values. For
    forces, convert to the magnitude of the forces."""

    for i in reference.keys():
        ax.scatter(reference[i], predicted[i], alpha=0.5, label=f"Iteration {i}")
    all_values = np.concatenate(list(reference.values()) + list(predicted.values()))
    min_val = all_values.min()
    max_val = all_values.max()
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Reference Energy / kcal mol$^{-1}$")
    ax.set_ylabel("Predicted Energy / kcal mol$^{-1}$")
    ax.set_title("Energy Correlation Plot")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def get_mol_image_with_atom_idxs(
    molecule: Molecule, width: int = 300, height: int = 300
) -> Image.Image:
    """Generate a PIL Image of the molecule with atom indices labeled."""
    molecule_copy = Molecule(molecule)
    molecule_copy._conformers = None

    rdmol = molecule_copy.to_rdkit()

    # Build labels like "C:0", "C:1", "C:2", ...
    atom_labels = {
        atom.GetIdx(): f"{atom.GetSymbol()}:{atom.GetIdx()}"
        for atom in rdmol.GetAtoms()
    }

    drawer = Draw.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    for idx, label in atom_labels.items():
        opts.atomLabels[idx] = label

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, rdmol)
    drawer.FinishDrawing()

    # Convert PNG bytes to PIL Image
    png_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png_data))

    return img


def plot_force_error_by_atom_idx(
    fig: Figure,
    ax: Axes,
    errors: dict[int, npt.NDArray[np.float64]],
    mol: Molecule,
) -> None:
    """Plot a seaborn swarmplot of the force errors by atom index."""
    import seaborn as sns

    for iteration, force_errors in errors.items():
        # Create an array of atom indices
        atom_indices = np.arange(len(force_errors)) % mol.n_atoms
        df = pd.DataFrame(
            {
                "atom_index": atom_indices,
                "force_error": np.linalg.norm(force_errors, axis=1),
                "iteration": np.ones_like(atom_indices) * iteration,
            }
        )
        sns.stripplot(
            x="atom_index",
            y="force_error",
            data=df,
            ax=ax,
            label=f"Iteration {iteration}",
            alpha=0.4,
        )

    # Get molecule image
    mol_image = get_mol_image_with_atom_idxs(mol, width=1800, height=600)

    # Create an inset axes above the main plot for the molecule
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax_inset = inset_axes(
        ax,
        width="400%",
        height="120%",
        loc="upper center",
        bbox_to_anchor=(0, 1.15, 1, 0.3),
        bbox_transform=ax.transAxes,
    )
    ax_inset.imshow(mol_image)
    ax_inset.axis("off")

    ax.set_xlabel("Atom Index")
    ax.set_ylabel("Force Error / kcal mol$^{-1}$ Å$^{-1}$")

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(
        by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left"
    )


def plot_distributions_of_errors(
    fig: Figure,
    ax: Axes,
    errors: dict[int, npt.NDArray[np.float64]],
    error_type: Literal["energy", "force"],
) -> None:
    # Colour by iteration
    # Use continuous colourmap for the iterations
    iterations = errors.keys()
    colours = plt.colormaps["viridis"](np.linspace(0, 1, len(iterations) + 1))

    for i in iterations:
        ax.hist(
            errors[i].flat if error_type == "force" else errors[i],
            label=f"Iteration {i}",
            alpha=0.8,
            color=colours[i],
            edgecolor="black",
        )

    ax.set_xlabel(
        f"Relative Energy Error / kcal mol$^{-1}$"
        if error_type == "energy"
        else "Relative Force Error / kcal mol$^{-1}$ Å$^{-1}$"
    )
    ax.set_ylabel("Frequency")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel(
        f"Relative Energy Error / kcal mol$^{-1}$"
        if error_type == "energy"
        else "Relative Force Error / kcal mol$^{-1}$ Å$^{-1}$"
    )
    ax.set_ylabel("Frequency")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def plot_mean_errors(
    fig: Figure,
    ax: Axes,
    errors: dict[int, npt.NDArray[np.float64]],
    error_type: Literal["energy", "force"],
) -> None:
    mean_errors = {i: np.mean(errors[i]) for i in errors.keys()}

    ax.plot(
        list(mean_errors.keys()),
        list(mean_errors.values()),
        marker="o",
        color="black",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        "Mean Relative Energy Error / kcal mol$^{-1}$"
        if error_type == "energy"
        else "Mean Relative Force Error / kcal mol$^{-1}$ Å$^{-1}$"
    )


def plot_rmse_of_errors(
    fig: Figure,
    ax: Axes,
    errors: dict[int, npt.NDArray[np.float64]],
    error_type: Literal["energy", "force"],
) -> None:
    rmsd_errors = [np.sqrt(np.mean(errors[i] ** 2)) for i in errors.keys()]

    ax.plot(
        list(errors.keys()),
        rmsd_errors,
        marker="o",
        color="black",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        "Root Mean Squared Relative Energy Error / kcal mol$^{-1}$"
        if error_type == "energy"
        else "Root Mean Squared Relative Force Error / kcal mol$^{-1}$ Å$^{-1}$"
    )


def plot_error_statistics(
    fig: Figure,
    axs: npt.NDArray[Any],
    errors: dict[
        Literal["energy_differences", "forces_differences"],
        dict[int, npt.NDArray[np.float64]],
    ],
) -> None:
    """Plot the error statistics for the energy and force errors."""

    axs = axs.flatten()
    plot_distributions_of_errors(fig, axs[0], errors["energy_differences"], "energy")
    plot_distributions_of_errors(fig, axs[1], errors["forces_differences"], "force")
    # Hide the legend in the first plot
    axs[0].legend().set_visible(False)

    # Plot the rmsds of the errors
    plot_rmse_of_errors(fig, axs[2], errors["energy_differences"], "energy")
    plot_rmse_of_errors(fig, axs[3], errors["forces_differences"], "force")

    # Plot the mean errors
    # plot_mean_errors(fig, axs[4], errors, "energy")
    # plot_mean_errors(fig, axs[5], errors, "force")


def plot_ff_differences(
    fig: Figure,
    ax: Axes,
    force_fields: dict[int, ForceField],
    molecule: Molecule,
    potential_type: POTENTIAL_KEYS,
    parameter_key: str,
) -> dict[str, float]:
    # Get the initial and final potentials
    iterations = list(force_fields)
    topology = molecule.to_topology()
    labeled_start = force_fields[iterations[0]].label_molecules(topology)[0]
    labeled_end = force_fields[iterations[-1]].label_molecules(topology)[0]

    objects_start = set(labeled_start[potential_type].values())
    objects_end = set(labeled_end[potential_type].values())

    if not objects_start and not objects_end:
        return {}

    # Convert to dicts with the id as the key
    potentials_start = {p.id: p.to_dict() for p in objects_start}
    potentials_end = {p.id: p.to_dict() for p in objects_end}

    # Param ids
    if set(potentials_start.keys()) != set(potentials_end.keys()):
        raise ValueError(
            f"Force field has different {potential_type} ids at start and end: {set(potentials_start.keys())} vs {set(potentials_end.keys())}"
        )
    param_ids = sorted(potentials_start.keys())

    # Get the differences for each key id
    differences = {
        param_id: potentials_end[param_id][parameter_key]
        - potentials_start[param_id][parameter_key]
        for param_id in param_ids
        if parameter_key
        in potentials_start[param_id]  # Skip missing keys, e.g. periodicity in torsions
    }

    if not differences:
        return {}

    differences_first_key = list(differences.keys())[0]

    # Plot the differences
    q_units = (
        units.unit.degrees
        if potential_type == "Angles" and parameter_key == "angle"
        else differences[differences_first_key].units
    )
    # Use numeric x positions to avoid matplotlib categorical warning
    x_positions = list(range(len(differences)))
    ax.bar(
        x_positions,
        [float(differences[k] / q_units) for k in differences.keys()],
    )

    ax.set_ylabel(f"{parameter_key} difference / {q_units}")
    ax.set_xlabel("Key ID")
    ax.set_title(f"{potential_type} {parameter_key} differences")

    # Rotate tick labels 90 - set ticks first to avoid warning
    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(differences.keys()), rotation=90)

    return differences


def plot_ff_values(
    fig: Figure,
    ax: Axes,
    force_fields: dict[int, ForceField],
    molecule: Molecule,
    potential_type: POTENTIAL_KEYS,
    parameter_key: str,
) -> None:
    # nice colour map for the iterations
    colours = plt.colormaps["viridis"](np.linspace(0, 1, len(force_fields) + 1))

    # Get the desired ids
    first_ff = force_fields[list(force_fields.keys())[0]]
    labeled = first_ff.label_molecules(molecule.to_topology())[0]
    potentials_set = set(labeled[potential_type].values())
    param_ids = sorted([p.id for p in potentials_set])

    for i, ff in force_fields.items():
        # Get the initial and final potentials
        labeled = ff.label_molecules(molecule.to_topology())[0]
        objects = set(labeled[potential_type].values())

        if not objects:
            return

        # Convert to dicts with the id as the key
        potentials = {p.id: p.to_dict() for p in objects}
        # Make sure that we have exactly the same ids as the first ff
        if set(potentials.keys()) != set(param_ids):
            raise ValueError(
                f"Force field at iteration {i} has different {potential_type} ids: {set(potentials.keys())} vs {set(param_ids)}"
            )

        vals = {
            param_id: potentials[param_id][parameter_key]
            for param_id in param_ids
            if parameter_key
            in potentials[param_id]  # Skip missing keys, e.g. periodicity in torsions
        }

        if not vals:
            return

        vals_first_key = list(vals.keys())[0]

        # Plot the differences
        q_units = (
            units.unit.degrees
            if potential_type == "Angles" and parameter_key == "angle"
            else vals[vals_first_key].units
        )
        # Plot as circles with correct colour, not bars
        x_vals = np.arange(len(vals.keys()))
        ax.scatter(
            x_vals,
            [float(vals[k] / q_units) for k in vals.keys()],
            color=colours[i],
            label=f"Iteration {i}",
            # small
            s=10,
        )

    # Set x labels to be the key ids
    ax.set_xticks(np.arange(len(param_ids)))
    ax.set_xticklabels(param_ids, rotation=90)
    ax.set_ylabel(f"{parameter_key} / {q_units}")
    ax.set_xlabel("Key ID")
    ax.set_title(f"{potential_type} {parameter_key}")


pot_types_and_param_keys: dict[POTENTIAL_KEYS, list[str]] = {
    "Bonds": ["length", "k"],
    "Angles": ["angle", "k"],
    "ProperTorsions": [
        "k1",
        "k2",
        "k3",
        "k4",
    ],  # "phase1", "phase2", "phase3", "phase4"],
    "ImproperTorsions": ["k1"],  # "phase1"],
}


def plot_all_ffs(
    force_fields: dict[int, ForceField],
    molecule: Molecule,
    plot_type: Literal["values", "differences"],
) -> tuple[Figure, Axes]:
    plt_fn = plot_ff_values if plot_type == "values" else plot_ff_differences

    # 1 column per potential type
    ncols = len(pot_types_and_param_keys)
    # 1 row for each of the greatest number of parameters
    nrows = max([len(v) for v in pot_types_and_param_keys.values()])
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))

    logger.info(
        f"Plotting force field {plot_type} with {nrows} rows and {ncols} columns"
    )
    for i, (potential_type, param_keys) in track(
        enumerate(pot_types_and_param_keys.items()),
        total=len(pot_types_and_param_keys),
        description=f"Plotting {plot_type}",
        transient=True,
    ):
        for j, param_key in enumerate(param_keys):
            plt_fn(fig, axs[j, i], force_fields, molecule, potential_type, param_key)

        # If this is the last potential type, add legend
        if i == ncols - 1:
            _add_legend_if_labels(axs[j, i], bbox_to_anchor=(1.05, 1), loc="upper left")

        # Hide the remaining axes
        for k in range(j + 1, nrows):
            axs[k, i].axis("off")

    _add_legend_if_labels(axs[2, 3], bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()

    return fig, axs


def calculate_dihedrals_for_trajectory(
    pdb_path: Path,
    torsions: dict[tuple[int, int], tuple[int, int, int, int]],
) -> dict[tuple[int, int, int, int], npt.NDArray[np.float64]]:
    """Calculate dihedral angles for all torsions across all frames using MDTraj.

    Parameters
    ----------
    pdb_path : Path
        Path to the PDB trajectory file.
    torsions : dict[tuple[int, int], tuple[int, int, int, int]]
        Dictionary mapping rotatable bonds to torsion atom indices.

    Returns
    -------
    dict[tuple[int, int, int, int], npt.NDArray[np.float64]]
        Dictionary mapping torsion atom indices to array of dihedral angles (in degrees)
        for each frame.
    """

    trajectory = mdtraj.load(str(pdb_path))
    dihedrals = {}

    for torsion_atoms in torsions.values():
        # MDTraj expects indices as a 2D array with shape (n_dihedrals, 4)
        indices = np.array([torsion_atoms])
        # compute_dihedrals returns angles in radians
        angles_rad = mdtraj.compute_dihedrals(trajectory, indices)
        # Convert to degrees and flatten (we only have one dihedral)
        dihedrals[torsion_atoms] = np.degrees(angles_rad.flatten())

    return dihedrals


def plot_torsion_dihedrals(
    fig: Figure,
    axs: npt.NDArray[Any],
    dihedrals_by_iteration: dict[
        int, dict[tuple[int, int, int, int], npt.NDArray[np.float64]]
    ],
    mol: Molecule,
) -> None:
    """Plot dihedral angles for all rotatable torsions during trajectories.

    Each torsion gets its own subplot.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.
    axs : npt.NDArray[Any]
        Array of matplotlib axes (one for each torsion).
    dihedrals_by_iteration : dict
        Dictionary mapping iteration to dictionary of torsion dihedrals.
        Inner dict maps torsion atom indices to array of dihedral angles.
    mol : Molecule
        The molecule being analyzed.
    """
    # Get molecule image with atom indices
    mol_image = get_mol_image_with_atom_idxs(mol, width=1800, height=600)

    # Create an inset axes above the main plot for the molecule
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Place molecule image above the first subplot
    ax_inset = inset_axes(
        axs.flat[0],
        width="400%",
        height="120%",
        loc="upper center",
        bbox_to_anchor=(0, 1.15, 1, 0.3),
        bbox_transform=axs.flat[0].transAxes,
    )
    ax_inset.imshow(mol_image)
    ax_inset.axis("off")

    # Get all unique torsions across all iterations
    all_torsions: set[tuple[int, int, int, int]] = set()
    for dihedrals in dihedrals_by_iteration.values():
        all_torsions.update(dihedrals.keys())
    all_torsions_list = sorted(all_torsions)

    # Plot dihedrals for each iteration
    colours = plt.colormaps["viridis"](
        np.linspace(0, 1, len(dihedrals_by_iteration) + 1)
    )

    # Create one subplot per torsion
    for torsion_idx, torsion_atoms in enumerate(all_torsions_list):
        ax = axs.flat[torsion_idx]

        # Collect angles by iteration
        angles_by_iteration = {}

        for iteration_idx, (iteration, dihedrals) in enumerate(
            dihedrals_by_iteration.items()
        ):
            if torsion_atoms in dihedrals:
                angles = dihedrals[torsion_atoms]
                angles_by_iteration[iteration] = (angles, iteration_idx)

                # Create frame numbers as x-axis
                frames = np.arange(len(angles))

                # Label with iteration
                label = f"Iteration {iteration}"

                ax.plot(
                    frames,
                    angles,
                    label=label,
                    alpha=0.5,
                    color=colours[iteration_idx],
                    linewidth=1.5,
                )

        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Dihedral Angle / degrees")
        ax.set_title(
            f"Torsion [{torsion_atoms[0]}-{torsion_atoms[1]}-{torsion_atoms[2]}-{torsion_atoms[3]}]"
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.axhline(y=180, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.axhline(y=-180, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_ylim(-180, 180)

        # Add legend
        _add_legend_if_labels(ax, loc="best", fontsize="small")

        # Add histogram as inset on the right side
        if angles_by_iteration:
            ax_hist = inset_axes(
                ax,
                width="20%",
                height="100%",
                loc="center right",
                bbox_to_anchor=(0.15, 0, 1, 1),
                bbox_transform=ax.transAxes,
            )

            # Create histogram for each iteration with matching colors
            for _iteration, (angles, iteration_idx) in angles_by_iteration.items():
                ax_hist.hist(
                    angles,
                    bins=36,
                    orientation="horizontal",
                    range=(-180, 180),
                    alpha=0.5,
                    color=colours[iteration_idx],
                    edgecolor="black",
                    linewidth=0.3,
                )

            ax_hist.set_ylim(-180, 180)
            ax_hist.set_xlabel("Count", fontsize=8)
            ax_hist.tick_params(axis="both", labelsize=7)
            ax_hist.yaxis.set_visible(False)

    # Hide unused subplots
    for idx in range(len(all_torsions), len(axs.flat)):
        axs.flat[idx].axis("off")


def analyse_workflow(workflow_settings: WorkflowSettings) -> None:
    """Analyse the results of a presto workflow."""

    mols = workflow_settings.parameterisation_settings.molecules

    # Suppress matplotlib categorical units warning by setting logger level
    import logging

    logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

    with plt.style.context(PLT_STYLE):
        # Plot the losses
        path_manager = workflow_settings.get_path_manager()
        stage = OutputStage(StageKind.PLOTS)
        path_manager.mk_stage_dir(stage)

        output_paths_by_type = path_manager.get_all_output_paths_by_output_type()
        output_paths_by_type_by_mol = (
            path_manager.get_all_output_paths_by_output_type_by_molecule()
        )

        training_metric_paths = dict(
            enumerate(output_paths_by_type[OutputType.TRAINING_METRICS])
        )
        losses = read_losses(training_metric_paths)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_loss(fig, ax, losses)
        fig.savefig(
            str(path_manager.get_output_path(stage, OutputType.LOSS_PLOT)),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Get scatter paths organized by molecule
        scatter_paths_by_mol = output_paths_by_type_by_mol.get(OutputType.SCATTER, {})
        assert isinstance(scatter_paths_by_mol, dict)

        # Plot for each molecule
        for mol_idx, mol in enumerate(mols):
            if mol_idx not in scatter_paths_by_mol:
                logger.warning(f"No scatter paths found for molecule {mol_idx}")
                continue

            # Convert list of paths to dict indexed by iteration
            scatter_paths_for_mol = dict(enumerate(scatter_paths_by_mol[mol_idx]))
            errors = read_errors(scatter_paths_for_mol)

            # Plot the errors
            fig, axs = plt.subplots(2, 2, figsize=(13, 12))
            plot_error_statistics(fig, axs, errors)  # type: ignore[arg-type]
            error_plot_path = path_manager.get_output_path(stage, OutputType.ERROR_PLOT)
            error_plot_path_mol = get_mol_path(error_plot_path, mol_idx)
            fig.savefig(str(error_plot_path_mol), dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Plot the correlation plots
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
            plot_energy_correlation(
                fig,
                ax,
                errors["energy_reference"],
                errors["energy_predicted"],
            )
            corr_plot_path = path_manager.get_output_path(
                stage, OutputType.CORRELATION_PLOT
            )
            corr_plot_path_mol = get_mol_path(corr_plot_path, mol_idx)
            fig.savefig(str(corr_plot_path_mol), dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Plot the force error by atom index
            fig, ax = plt.subplots(1, 1, figsize=(0.5 * mol.n_atoms, 6))
            plot_force_error_by_atom_idx(fig, ax, errors["forces_differences"], mol)
            force_error_plot_path = path_manager.get_output_path(
                stage, OutputType.FORCE_ERROR_BY_ATOM_INDEX_PLOT
            )
            force_error_plot_path_mol = get_mol_path(force_error_plot_path, mol_idx)
            fig.savefig(str(force_error_plot_path_mol), dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Plot torsion dihedrals if trajectory files exist
            pdb_traj_paths_by_mol = output_paths_by_type_by_mol.get(
                OutputType.PDB_TRAJECTORY, {}
            )
            if mol_idx in pdb_traj_paths_by_mol:
                # Get rotatable torsions for this molecule
                torsions = get_rot_torsions_by_rot_bond(
                    mol,
                    include_smarts=_TORSIONS_TO_INCLUDE_SMARTS,
                    exclude_smarts=_TORSIONS_TO_EXCLUDE_SMARTS,
                )

                if torsions:
                    # Read trajectories and calculate dihedrals for each iteration
                    dihedrals_by_iteration = {}
                    pdb_paths_for_mol = pdb_traj_paths_by_mol[mol_idx]
                    if not isinstance(pdb_paths_for_mol, list):
                        pdb_paths_for_mol = [pdb_paths_for_mol]

                    # Include initial statistics (iteration 0) and training iterations
                    for iter_idx, pdb_path in enumerate(pdb_paths_for_mol):
                        if pdb_path.exists():
                            try:
                                dihedrals = calculate_dihedrals_for_trajectory(
                                    pdb_path, torsions
                                )
                                dihedrals_by_iteration[iter_idx] = dihedrals
                            except Exception as e:
                                logger.warning(
                                    f"Failed to read trajectory at {pdb_path}: {e}"
                                )

                    if dihedrals_by_iteration:
                        # Determine figure layout based on number of torsions
                        n_torsions = len(torsions)
                        # Create a grid layout - 2 columns for better layout
                        ncols = min(2, n_torsions)
                        nrows = (n_torsions + ncols - 1) // ncols
                        fig, axs = plt.subplots(
                            nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False
                        )
                        plot_torsion_dihedrals(fig, axs, dihedrals_by_iteration, mol)
                        torsion_plot_path = path_manager.get_output_path(
                            stage, OutputType.TORSION_DIHEDRALS_PLOT
                        )
                        torsion_plot_path_mol = get_mol_path(torsion_plot_path, mol_idx)
                        fig.tight_layout()
                        fig.savefig(
                            str(torsion_plot_path_mol), dpi=300, bbox_inches="tight"
                        )
                        plt.close(fig)

        # Plot the force field changes for each molecule
        offxml_paths = output_paths_by_type.get(OutputType.OFFXML, [])
        assert isinstance(offxml_paths, list)
        ff_paths = load_force_fields(dict(enumerate(offxml_paths)))

        for mol_idx, mol in enumerate(mols):
            fig, axs = plot_all_ffs(ff_paths, mol, "values")
            param_values_plot_path = path_manager.get_output_path(
                stage, OutputType.PARAMETER_VALUES_PLOT
            )
            param_values_plot_path_mol = get_mol_path(param_values_plot_path, mol_idx)
            fig.savefig(str(param_values_plot_path_mol), dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig, axs = plot_all_ffs(ff_paths, mol, "differences")
            param_diff_plot_path = path_manager.get_output_path(
                stage, OutputType.PARAMETER_DIFFERENCES_PLOT
            )
            param_diff_plot_path_mol = get_mol_path(param_diff_plot_path, mol_idx)
            fig.savefig(str(param_diff_plot_path_mol), dpi=300, bbox_inches="tight")
            plt.close(fig)
