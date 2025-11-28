"""Functionality for analysing the results of a BespokeFitSMEE run."""

import io
from pathlib import Path
from typing import Any, Literal

import h5py
import loguru
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
from tqdm import tqdm

from .outputs import OutputStage, OutputType, StageKind
from .loss import LossRecord
from .settings import WorkflowSettings

logger = loguru.logger

PLT_STYLE = "ggplot"

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
    for i in losses["iteration"].unique():
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
                    label if i == 0 else None
                ),  # Don't repeat labels for new iterations
                color=f"C{color_idx}",
                linestyle=linestyle,
            )

        # If this isn't the last iteration, add a vertical line to separate iterations # and label it
        if i != losses["iteration"].unique()[-1]:
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
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


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
    colours = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(iterations) + 1))

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


def plot_sd_of_errors(
    fig: Figure,
    ax: Axes,
    errors: dict[int, npt.NDArray[np.float64]],
    error_type: Literal["energy", "force"],
) -> None:
    sd_errors = {i: np.std(errors[i]) for i in errors.keys()}

    ax.plot(
        list(sd_errors.keys()),
        list(sd_errors.values()),
        marker="o",
        color="black",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        "Standard Deviation of Relative Energy Error / kcal mol$^{-1}$"
        if error_type == "energy"
        else "Standard Deviation of Relative Force Error / kcal mol$^{-1}$ Å$^{-1}$"
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

    # # Plot the standard deviation of the errors
    plot_sd_of_errors(fig, axs[4], errors["energy_differences"], "energy")
    plot_sd_of_errors(fig, axs[5], errors["forces_differences"], "force")


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

    parameter_keys = [
        k
        for k in potentials_start[list(potentials_start.keys())[0]].keys()
        if k not in ["smirks", "id"]
    ]
    if parameter_key not in parameter_keys:
        raise ValueError(f"Parameter key {parameter_key} not found in {parameter_keys}")

    # Get the differences for each key id
    differences = {
        param_id: potentials_end[param_id][parameter_key]
        - potentials_start[param_id][parameter_key]
        for param_id in param_ids
    }
    differences_first_key = list(differences.keys())[0]

    # Plot the differences
    q_units = (
        units.unit.degrees
        if potential_type == "Angles" and parameter_key == "angle"
        else differences[differences_first_key].units
    )
    ax.bar(
        list(differences.keys()),
        [float(differences[k] / q_units) for k in differences.keys()],
    )

    ax.set_ylabel(f"{parameter_key} difference / {q_units}")
    ax.set_xlabel("Key ID")
    ax.set_title(f"{potential_type} {parameter_key} differences")

    # Rotate tick labels 90
    ax.set_xticklabels(differences.keys(), rotation=90)

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
    colours = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(force_fields) + 1))

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

        parameter_keys = [
            k
            for k in potentials[list(potentials.keys())[0]].keys()
            if k not in ["smirks", "id"]
        ]
        if parameter_key not in parameter_keys:
            raise ValueError(
                f"Parameter key {parameter_key} not found in {parameter_keys}"
            )

        # Get the differences for each key id
        vals = {param_id: potentials[param_id][parameter_key] for param_id in param_ids}
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
    "ProperTorsions": ["k1", "k2", "k3", "k4", "phase1", "phase2", "phase3", "phase4"],
    "ImproperTorsions": ["k1", "phase1"],
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
    for i, (potential_type, param_keys) in tqdm(
        enumerate(pot_types_and_param_keys.items()), total=len(pot_types_and_param_keys)
    ):
        for j, param_key in enumerate(param_keys):
            plt_fn(fig, axs[j, i], force_fields, molecule, potential_type, param_key)

        # If this is the last potential type, add legend
        if i == ncols - 1:
            axs[j, i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Hide the remaining axes
        for k in range(j + 1, nrows):
            axs[k, i].axis("off")

    axs[2, 3].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()

    return fig, axs


def analyse_workflow(workflow_settings: WorkflowSettings) -> None:
    """Analyse the results of a BespokeFitSMEE workflow."""

    with plt.style.context(PLT_STYLE):
        # Plot the losses
        path_manager = workflow_settings.get_path_manager()
        stage = OutputStage(StageKind.PLOTS)
        path_manager.mk_stage_dir(stage)
        mol = Molecule.from_smiles(
            workflow_settings.parameterisation_settings.smiles,
            allow_undefined_stereo=True,
        )

        output_paths_by_output_type = path_manager.get_all_output_paths_by_output_type()
        training_metric_paths = dict(
            enumerate(output_paths_by_output_type[OutputType.TRAINING_METRICS])
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

        # Plot the errors
        scatter_paths = dict(enumerate(output_paths_by_output_type[OutputType.SCATTER]))
        errors = read_errors(scatter_paths)
        fig, axs = plt.subplots(3, 2, figsize=(13, 18))
        # TODO: typing below which is ignored
        plot_error_statistics(fig, axs, errors)  # type: ignore[arg-type]
        fig.savefig(
            str(path_manager.get_output_path(stage, OutputType.ERROR_PLOT)),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Plot the correlation plots
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
        plot_energy_correlation(
            fig,
            ax,
            errors["energy_reference"],
            errors["energy_predicted"],
        )
        fig.savefig(
            str(path_manager.get_output_path(stage, OutputType.CORRELATION_PLOT)),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Plot the force error by atom index
        fig, ax = plt.subplots(1, 1, figsize=(0.5 * mol.n_atoms, 6))
        plot_force_error_by_atom_idx(fig, ax, errors["forces_differences"], mol)
        fig.savefig(
            str(
                path_manager.get_output_path(
                    stage, OutputType.FORCE_ERROR_BY_ATOM_INDEX_PLOT
                )
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Plot the force field changes
        ff_paths = load_force_fields(
            dict(enumerate(output_paths_by_output_type[OutputType.OFFXML]))
        )

        fig, axs = plot_all_ffs(ff_paths, mol, "values")
        fig.savefig(
            str(path_manager.get_output_path(stage, OutputType.PARAMETER_VALUES_PLOT)),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, axs = plot_all_ffs(ff_paths, mol, "differences")
        fig.savefig(
            str(
                path_manager.get_output_path(
                    stage, OutputType.PARAMETER_DIFFERENCES_PLOT
                )
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
