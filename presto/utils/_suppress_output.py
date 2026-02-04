"""Utility to suppress unwanted output from dependencies."""

import logging
import warnings


def suppress_unwanted_output() -> None:
    """Suppress known nuisance warnings from dependencies."""

    # Suppress pkg_resources deprecation warning from smirnoff99frosst
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API",
        category=UserWarning,
        module="smirnoff99frosst.smirnoff99frosst",
    )

    # Suppress torch.load FutureWarning from e3nn
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning,
        module="e3nn.o3._wigner",
    )

    # Suppress torch.load FutureWarning from openmmml
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning,
        module="openmmml.models.macepotential",
    )

    # Suppress simtk.openmm deprecation warning
    warnings.filterwarnings(
        "ignore",
        message=".*simtk.openmm.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*simtk.openmm.*",
        category=UserWarning,
    )

    # Suppress torch.tensor() warning from descent
    warnings.filterwarnings(
        "ignore",
        message="To copy construct from a tensor.*",
        category=UserWarning,
        module="descent.targets.energy",
    )

    # Suppress warnings from openff
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="openff.toolkit",
    )

    # Suppress INFO logs from openff.interchange.smirnoff._nonbonded
    logging.getLogger("openff.interchange.smirnoff._nonbonded").setLevel(
        logging.WARNING
    )
    # Suppress INFO logs from openff.nagl
    logging.getLogger("openff.nagl").setLevel(logging.WARNING)

    # Suppress httpx logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Suppress descent logs
    logging.getLogger("descent").setLevel(logging.WARNING)

    # Suppress presto.msm debug logs
    logging.getLogger("presto.msm").setLevel(logging.INFO)

    # Suppress huggingface_hub warnings
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
