"""Pydantic models which control/validate the settings."""

import warnings
from abc import ABC
from pathlib import Path
from typing import Literal, TypeVar, Union

import numpy as np
import torch
import yaml
from descent.train import AttributeConfig, ParameterConfig
from loguru import logger
from openff.toolkit import Molecule
from openmm import unit
from packaging.version import Version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_units import OpenMMQuantity
from rdkit import Chem
from typing_extensions import Self

from . import __version__, mlp
from ._exceptions import InvalidSettingsError
from .outputs import OutputType, WorkflowPathManager
from .utils.typing import (
    AllowedAttributeType,
    NonLinearValenceType,
    OptimiserName,
    PathLike,
    TorchDevice,
    ValenceType,
)

_DEFAULT_SMILES_PLACEHOLDER = "CHANGEME"

_DEFAULT_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    validate_assignment=True,
)


def _model_to_yaml(model: BaseModel, yaml_path: PathLike) -> None:
    """Save the settings to a YAML file"""
    data = model.model_dump(mode="json")
    with open(yaml_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False, indent=4)


_T = TypeVar("_T", bound=BaseModel)


def _model_from_yaml(cls: type[_T], yaml_path: PathLike) -> _T:
    """Load settings from a YAML file"""
    with open(yaml_path, "r") as file:
        settings_data = yaml.safe_load(file)
    return cls(**settings_data)


class _DefaultSettings(BaseModel, ABC):
    """Default configuration for all models."""

    model_config = _DEFAULT_MODEL_CONFIG

    def to_yaml(self, yaml_path: PathLike) -> None:
        """Save the settings to a YAML file"""
        _model_to_yaml(self, yaml_path)

    @classmethod
    def from_yaml(cls, yaml_path: PathLike) -> Self:
        """Load settings from a YAML file"""
        return _model_from_yaml(cls, yaml_path)

    @property
    def output_types(self) -> set[OutputType]:
        """Return a set of expected output types for the function which
        implements this settings object. Subclasses should override this method."""
        return set()

    # @property
    # def output_types(self) -> set[OutputType]:
    #     """Return a set of expected output types for this settings object
    #     and all _DefaultSettings it contains."""
    #     outputs = set(self._output_types)
    #     for name, field in self.model_fields.items():
    #         if issubclass(field.annotation, _DefaultSettings):
    #             nested_settings = getattr(self, name)
    #             outputs.update(nested_settings.output_types)
    #     return outputs


class _SamplingSettingsBase(_DefaultSettings, ABC):
    """Settings for sampling (usually molecular dynamics)."""

    sampling_protocol: str = Field(
        ...,
        description="Type of sampling protocol. Each sampling settings subclass "
        "should set this to a unique value. This is used as a discriminator when "
        "loading from YAML.",
    )

    ml_potential: Literal[mlp.AvailableModels] = Field(
        "egret-1",
        description="The machine learning potential to use for calculating energies and forces of "
        " the snapshots. Note that this is not generally the potential used for sampling.",
    )

    timestep: OpenMMQuantity[unit.femtoseconds] = Field(  # type: ignore[type-arg]
        default=1 * unit.femtoseconds,
        description="MD timestep",
    )

    temperature: OpenMMQuantity[unit.kelvin] = Field(  # type: ignore[type-arg]
        default=500 * unit.kelvin,
        description="Temperature to run MD at",
    )

    snapshot_interval: OpenMMQuantity[unit.femtoseconds] = Field(  # type: ignore[type-arg]
        default=400 * unit.femtoseconds,
        description="Interval between saving snapshots during production sampling",
    )

    n_conformers: int = Field(
        10,
        description="The number of conformers to generate, from which sampling is started",
    )

    equilibration_sampling_time_per_conformer: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        default=0.1 * unit.picoseconds,
        description="Equilibration sampling time per conformer. No snapshots are saved during "
        "equilibration sampling. The total sampling time per conformer will be this plus "
        "the production_sampling_time_per_conformer.",
    )

    production_sampling_time_per_conformer: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        default=20 * unit.picoseconds,
        description="Production sampling time per conformer. The total sampling time per conformer "
        "will be this plus the equilibration_sampling_time_per_conformer.",
    )

    loss_energy_weight: float = Field(
        1000.0,
        description="Scaling factor for the energy loss term for samples from this protocol.",
    )

    loss_force_weight: float = Field(
        0.1,
        description="Scaling factor for the force loss term for samples from this protocol.",
    )

    @property
    def equilibration_n_steps_per_conformer(self) -> int:
        return int(self.equilibration_sampling_time_per_conformer / self.timestep)

    @property
    def production_n_snapshots_per_conformer(self) -> int:
        return int(self.production_sampling_time_per_conformer / self.snapshot_interval)

    @property
    def production_n_steps_per_snapshot_per_conformer(self) -> int:
        return int(self.snapshot_interval / self.timestep)

    @property
    def output_types(self) -> set[OutputType]:
        return {OutputType.PDB_TRAJECTORY}

    @model_validator(mode="after")
    def validate_sampling_times(self) -> Self:
        """Ensure that the sampling times divide exactly by the timestep and (for production) the snapshot interval."""
        for time, name in [
            (
                self.equilibration_sampling_time_per_conformer,
                "equilibration_sampling_time_per_conformer",
            ),
            (
                self.production_sampling_time_per_conformer,
                "production_sampling_time_per_conformer",
            ),
        ]:
            n_steps = time / self.timestep
            if not n_steps.is_integer():
                raise InvalidSettingsError(
                    f"{name} ({time}) must be divisible by the timestep ({self.timestep})."
                )

        # Additionally check that production sampling time divides by snapshot interval
        time = self.production_sampling_time_per_conformer / self.snapshot_interval
        if not n_steps.is_integer():
            raise InvalidSettingsError(
                f"production_sampling_time_per_conformer ({time}) must be divisible by the snapshot_interval ({self.snapshot_interval})."
            )

        return self


class MMMDSamplingSettings(_SamplingSettingsBase):
    """Settings for molecular dynamics sampling using a molecular mechanics
    force field. This is initally the force field supplined in the parameterisation
    settings, but is updated as the bespoke force field is trained."""

    sampling_protocol: Literal["mm_md"] = Field(
        "mm_md", description="Sampling protocol to use."
    )


class MLMDSamplingSettings(_SamplingSettingsBase):
    """Settings for molecular dynamics sampling using a machine learning
    potential. This protocol uses the ML reference potential for sampling as
    well as for energy and force calculations."""

    sampling_protocol: Literal["ml_md"] = Field(
        "ml_md", description="Sampling protocol to use."
    )


class MMMDMetadynamicsSamplingSettings(_SamplingSettingsBase):
    """Settings for molecular dynamics sampling using a molecular mechanics
    force field with metadynamics. This is initally the force field supplined in the parameterisation
    settings, but is updated as the bespoke force field is trained."""

    sampling_protocol: Literal["mm_md_metadynamics"] = Field(
        "mm_md_metadynamics", description="Sampling protocol to use."
    )

    metadynamics_bias_factor: float = Field(
        10.0, description="Bias factor for well-tempered metadynamics"
    )

    bias_width: float = Field(np.pi / 10, description="Width of the bias (in radians)")

    bias_factor: float = Field(
        10.0,
        description="Bias factor for well-tempered metadynamics. Typical range: 5-20",
    )

    bias_height: OpenMMQuantity[unit.kilojoules_per_mole] = Field(  # type: ignore[type-arg]
        2.0 * unit.kilojoules_per_mole,
        description="Initial height of the bias",
    )

    bias_frequency: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        0.5 * unit.picoseconds,
        description="Frequency at which to add bias",
    )

    bias_save_frequency: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        1.0 * unit.picoseconds,
        description="Frequency at which to save the bias",
    )

    # Make sure that the frequency and save_frequency are multiples of the timestep
    @model_validator(mode="after")
    def validate_frequencies(self) -> Self:
        for freq, name in [
            (self.bias_frequency, "frequency"),
            (self.bias_save_frequency, "save_frequency"),
        ]:
            n_steps = freq / self.timestep
            if not n_steps.is_integer():
                raise InvalidSettingsError(
                    f"{name} ({freq}) must be divisible by the timestep ({self.timestep})."
                )

            # Make sure that the sampling time per conformer is a multiple of the save frequency
            n_saves = self.production_sampling_time_per_conformer / freq
            if not n_saves.is_integer():
                raise InvalidSettingsError(
                    f"production_sampling_time_per_conformer ({self.production_sampling_time_per_conformer}) must be divisible by the {name} ({freq})."
                )
        return self

    @property
    def n_steps_per_bias(self) -> int:
        return int(self.bias_frequency / self.timestep)

    @property
    def n_steps_per_bias_save(self) -> int:
        return int(self.bias_save_frequency / self.timestep)

    @property
    def output_types(self) -> set[OutputType]:
        return {OutputType.METADYNAMICS_BIAS, OutputType.PDB_TRAJECTORY}


class MMMDMetadynamicsTorsionMinimisationSamplingSettings(
    MMMDMetadynamicsSamplingSettings
):
    """Settings for MM MD metadynamics sampling with additional torsion-restrained
    minimisation structures. This extends MMMDMetadynamicsSamplingSettings by generating
    additional training data from torsion-restrained minimisations."""

    sampling_protocol: Literal["mm_md_metadynamics_torsion_minimisation"] = Field(  # type: ignore[assignment]
        "mm_md_metadynamics_torsion_minimisation",
        description="Sampling protocol to use.",
    )

    # Settings for torsion-restrained minimisation
    ml_minimisation_steps: int = Field(
        10,
        description="Number of MLP minimisation steps with restrained torsions.",
    )

    mm_minimisation_steps: int = Field(
        10,
        description="Number of MM minimisation steps with restrained torsions.",
    )

    torsion_restraint_force_constant: OpenMMQuantity[  # type: ignore[type-arg, valid-type]
        unit.kilojoules_per_mole / unit.radian**2
    ] = Field(
        0.0 * unit.kilojoules_per_mole / unit.radian**2,
        description="Force constant for torsion restraints.",
    )

    # Loss weights for the MMMD metadynamics samples
    loss_energy_weight_mmmd: float = Field(
        1000.0,
        description="Scaling factor for the energy loss term for MMMD metadynamics samples.",
    )

    loss_force_weight_mmmd: float = Field(
        0.1,
        description="Scaling factor for the force loss term for MMMD metadynamics samples.",
    )

    # Loss weights for the torsion-minimised samples
    map_ml_coords_energy_to_mm_coords_energy: bool = Field(
        True,
        description="Whether to substitute the MLP energy for the MM-minimised coordinates with the "
        "MLP energy for the corresponding MLP-minimised coordinates.",
    )

    loss_energy_weight_mm_torsion_min: float = Field(
        1000.0,
        description="Scaling factor for the energy loss term for torsion-minimised samples, using "
        "MM minimisation.",
    )

    loss_force_weight_mm_torsion_min: float = Field(
        0.1,
        description="Scaling factor for the force loss term for torsion-minimised samples. ",
    )

    loss_energy_weight_ml_torsion_min: float = Field(
        1000.0,
        description="Scaling factor for the energy loss term for torsion-minimised samples, using "
        "MLP minimisation.",
    )

    loss_force_weight_ml_torsion_min: float = Field(
        0.1,
        description="Scaling factor for the force loss term for torsion-minimised samples. ",
    )

    @property
    def output_types(self) -> set[OutputType]:
        return {
            OutputType.METADYNAMICS_BIAS,
            OutputType.PDB_TRAJECTORY,
            OutputType.ML_MINIMISED_PDB,
            OutputType.MM_MINIMISED_PDB,
        }


SamplingSettings = Union[
    MMMDSamplingSettings,
    MLMDSamplingSettings,
    MMMDMetadynamicsSamplingSettings,
    MMMDMetadynamicsTorsionMinimisationSamplingSettings,
]


def _get_default_regularised_parameters() -> dict[ValenceType, list[str]]:
    return {
        "ProperTorsions": ["k"],
        "ImproperTorsions": ["k"],
    }


class TrainingSettings(_DefaultSettings):
    """Settings for the training process."""

    optimiser: OptimiserName = Field(
        "adam",
        description="Optimiser to use for the training. 'adam' is Adam, 'lm' is Levenberg-Marquardt",
    )
    # Use AttributeConfigs to prevent the user passing exclude or include keys,
    # which should be set in the parameterisation settings because they decide
    # which tagged SMARTS are generated
    parameter_configs: dict[ValenceType, ParameterConfig] = Field(
        default_factory=lambda: {  # type: ignore[arg-type]
            "LinearBonds": ParameterConfig(
                cols=["k1", "k2"],
                scales={"k1": 0.0028, "k2": 0.0028},
                limits={"k1": (1e-8, None), "k2": (1e-8, None)},
                include=None,
                exclude=None,
            ),
            "LinearAngles": ParameterConfig(
                cols=["k1", "k2"],
                scales={"k1": 0.012, "k2": 0.011},
                limits={"k1": (1e-8, None), "k2": (1e-8, None)},
                include=None,
                exclude=None,
            ),
            "ProperTorsions": ParameterConfig(
                cols=["k"],
                scales={"k": 1.3},
                limits={"k": (None, None)},
                regularize={"k": 1.0},
                include=None,
                # Exclude linear torsions to avoid non-zero force constants which can
                # cause instabilities. Taken from https://github.com/openforcefield/openff-forcefields/blob/05f7ad0daad1ccdefdf931846fd13df863ab5c7d/openforcefields/offxml/openff-2.2.1.offxml#L326-L328
                exclude=[
                    {
                        "id": "[*:1]-[*:2]#[*:3]-[*:4]",
                        "multiplicity": 1,
                        "parameter_handler": "ProperTorsions",
                    },
                    {
                        "id": "[*:1]~[*:2]-[*:3]#[*:4]",
                        "multiplicity": 1,
                        "parameter_handler": "ProperTorsions",
                    },
                    {
                        "id": "[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]",
                        "multiplicity": 1,
                        "parameter_handler": "ProperTorsions",
                    },
                ],
            ),
            "ImproperTorsions": ParameterConfig(
                cols=["k"],
                scales={"k": 0.12},
                limits={"k": (0, None)},
                regularize={"k": 1.0},
                include=None,
                exclude=None,
            ),
        },
        description="Configuration for the force field parameters to be trained.",
    )

    attribute_configs: dict[AllowedAttributeType, AttributeConfig] = Field(
        {},
        description="Configuration for the force field attributes to be trained. "
        "This allows 1-4 scaling for 'vdW' and 'Electrostatics' to be trained.",
    )

    n_epochs: int = Field(1000, description="Number of epochs in the ML fit")
    learning_rate: float = Field(0.01, description="Learning Rate in the ML fit")
    learning_rate_decay: float = Field(
        1.00, description="Learning Rate Decay. 0.99 is 1%, and 1.0 is no decay."
    )
    learning_rate_decay_step: int = Field(10, description="Learning Rate Decay Step")
    regularisation_target: Literal["initial", "zero"] = Field(
        "initial",
        description="Target value to regularise parameters towards. 'initial' is the initial parameter value, "
        "'zero' is zero.",
    )

    @property
    def output_types(self) -> set[OutputType]:
        return {
            OutputType.TENSORBOARD,
            OutputType.TRAINING_METRICS,
        }


class TypeGenerationSettings(_DefaultSettings):
    """Settings for generating tagged SMARTS types for a given potential type."""

    max_extend_distance: int = Field(
        -1,
        description="Maximum number of bonds to extend from the atoms to which the potential is applied "
        "when generating tagged SMARTS patterns. A value of -1 means no limit.",
    )
    include: list[str] = Field(
        [],
        description="List of SMARTS present in the initial force field for which to generate new SMARTS "
        " patterns. This allows you to split specific types for reparameterisation. This is mutually exclusive "
        "with the exclude field.",
    )

    exclude: list[str] = Field(
        [],
        description="List of SMARTS patterns to exclude when generating tagged SMARTS types. If present, "
        " these patterns will remain the same as in the initial force field. This is mutually exclusive "
        "with the include field.",
    )

    @model_validator(mode="after")
    def validate_include_exclude(self) -> Self:
        """Ensure that only one of include or exclude is set."""
        if self.include and self.exclude:
            raise InvalidSettingsError(
                "Only one of include or exclude can be set in TypeGenerationSettings."
            )
        return self


class ParameterisationSettings(_DefaultSettings):
    """Settings for the starting parameterisation."""

    smiles: Union[str, list[str]] = Field(
        ...,
        description="SMILES string or list of SMILES for molecules to fit",
    )

    initial_force_field: str = Field(
        "openff_unconstrained-2.3.0-rc2.offxml",
        description="The force field from which to start. This can be any"
        " OpenFF force field, or your own .offxml file.",
    )

    expand_torsions: bool = Field(
        True,
        description="Whether to expand the torsion periodicities up to 4.",
    )

    linearise_harmonics: bool = Field(
        True,
        description="Linearise the harmonic potentials in the Force Field (Default)",
    )

    type_generation_settings: dict[NonLinearValenceType, TypeGenerationSettings] = (
        Field(
            default_factory=lambda: {  # type: ignore[arg-type]
                "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
                "Angles": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
                "ProperTorsions": TypeGenerationSettings(
                    max_extend_distance=-1,
                    exclude=[
                        "[*:1]-[*:2]#[*:3]-[*:4]",  # Linear torsions should be kept linear
                        "[*:1]~[*:2]-[*:3]#[*:4]",  # Linear torsions should be kept linear
                        "[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]",  # Linear torsions should be kept linear
                    ],
                ),
                "ImproperTorsions": TypeGenerationSettings(
                    max_extend_distance=-1, exclude=[]
                ),
            },
            description="Settings for generating tagged SMARTS types for each valence type.",
        )
    )

    # Validate that all SMILES strings are valid
    @field_validator("smiles", mode="before")
    def validate_smiles(cls, value: str | list[str]) -> list[str]:
        """Validate all SMILES are valid, unique. Accepts string or list."""
        # Convert single string to list for backward compatibility
        if isinstance(value, str):
            value = [value]

        if not value:
            raise ValueError("smiles list cannot be empty")

        # Check for duplicates
        if len(value) != len(set(value)):
            duplicates = [s for s in value if value.count(s) > 1]
            unique_duplicates = list(set(duplicates))
            raise ValueError(f"Duplicate SMILES found: {unique_duplicates}")

        # Validate each SMILES string
        for smiles in value:
            if Chem.MolFromSmiles(smiles) is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
        return value

    @property
    def molecules(self) -> list[Molecule]:
        """Return the list of OpenFF Molecule objects for the SMILES strings."""
        return [
            Molecule.from_smiles(smiles, allow_undefined_stereo=True)
            for smiles in self.smiles
        ]


class WorkflowSettings(_DefaultSettings):
    """Overall settings for the full fitting workflow."""

    version: str = Field(
        __version__,
        description="Version of bespokefit_smee used to create these settings",
    )

    output_dir: Path = Field(
        Path("."),
        description="Directory where the output files will be saved",
    )

    device_type: TorchDevice = Field(
        "cuda", description="Device type for training, either 'cpu' or 'cuda'"
    )

    n_iterations: int = Field(
        2,
        description="Number of iterations of sampling, then training the FF to run",
    )

    memory: bool = Field(
        False,
        description="Whether to append new training data to training data from the previous iterations,"
        " or overwrite it (False).",
    )

    parameterisation_settings: ParameterisationSettings = Field(
        description="Settings for the starting parameterisation",
    )

    training_sampling_settings: SamplingSettings = Field(
        default_factory=lambda: MMMDMetadynamicsTorsionMinimisationSamplingSettings(),
        description="Settings for sampling for generating the training data (usually molecular dynamics)",
        discriminator="sampling_protocol",
    )

    testing_sampling_settings: SamplingSettings = Field(
        default_factory=lambda: MMMDMetadynamicsTorsionMinimisationSamplingSettings(),
        description="Settings for sampling for generating the testing data (usually molecular dynamics)",
        discriminator="sampling_protocol",
    )

    training_settings: TrainingSettings = Field(
        default_factory=lambda: TrainingSettings(),
        description="Settings for the training process",
    )

    # Raise an error if the major and minor versions do not match
    # (don't care about patch version)
    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """Validate version format and check compatibility."""
        try:
            parsed = Version(value)
        except Exception as e:
            raise ValueError(f"Invalid version format: {value}") from e

        actual_version = Version(__version__)

        # Warn the user if major or minor versions do not match
        if parsed.major != actual_version.major or parsed.minor != actual_version.minor:
            logger.warning(
                f"Version mismatch: settings version {value} may not be compatible with current version {__version__}."
            )

        return value

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, value: TorchDevice) -> TorchDevice:
        """Ensure that the requested device type is available."""
        if value == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this system.")

        if value == "cpu":
            warnings.warn(
                "Using CPU for training and sampling. This may be slow. Consider using CUDA if available.",
                UserWarning,
                stacklevel=2,
            )

        return value

    # Validate that linearise_harmonics argument in parameterisation settings is consistent with the valence types
    # in the training settings
    @model_validator(mode="after")
    def validate_parameterisation_training_consistency(self) -> Self:
        """Validate that linearise_harmonics argument in parameterisation settings is consistent with the valence types
        in the training settings."""

        harmonics_linearised = self.parameterisation_settings.linearise_harmonics
        excluded_valence_types = (
            ("Bonds", "Angles")
            if harmonics_linearised
            else ("LinearBonds", "LinearAngles")
        )
        if any(
            valence_type in self.training_settings.parameter_configs
            for valence_type in excluded_valence_types
        ):
            raise InvalidSettingsError(
                f"ParameterisationSettings.linearise_harmonics is {harmonics_linearised}, but TrainingSettings.parameter_configs "
                f"contains valence types that are inconsistent with this setting: {excluded_valence_types}. "
            )

        return self

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type)

    def get_path_manager(self) -> WorkflowPathManager:
        """Get the output paths manager for this workflow settings object."""
        # Get the number of molecules from the smiles list
        smiles = self.parameterisation_settings.smiles
        n_mols = len(smiles) if isinstance(smiles, list) else 1
        return WorkflowPathManager(
            output_dir=self.output_dir,
            n_iterations=self.n_iterations,
            n_mols=n_mols,
            training_settings=self.training_settings,
            training_sampling_settings=self.training_sampling_settings,
            testing_sampling_settings=self.testing_sampling_settings,
        )
