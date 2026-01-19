"""Apply OpenFF parameters to molecule, cluster conformers by RMSD and train"""

import functools
import logging
from pathlib import Path
from typing import Protocol, TypedDict, Unpack

import datasets
import datasets.combine
import descent
import descent.optim
import loguru
import smee
import tensorboardX
import torch
from tqdm import tqdm

# from .sample import get_data_MLMD, get_data_MMMD
from .loss import get_loss_closure_fn, prediction_loss
from .outputs import OutputType
from .settings import (
    TrainingSettings,
)
from .utils.register import get_registry_decorator
from .utils.typing import OptimiserName, PathLike
from .writers import (
    open_writer,
    report,
    write_metrics,
)

logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG for more detail
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("descent").setLevel(logging.DEBUG)

logger = loguru.logger


class TrainingFnArgs(TypedDict):
    """Arguments for training functions."""

    trainable_parameters: torch.Tensor
    initial_parameters: torch.Tensor
    trainable: descent.train.Trainable
    topologies: list[smee.TensorTopology]
    datasets: list[datasets.Dataset]
    datasets_test: list[datasets.Dataset]
    settings: TrainingSettings
    output_paths: dict[OutputType, Path]
    device: torch.device


class TrainFn(Protocol):
    """A protocol for training functions."""

    def __call__(
        self, **kwargs: Unpack[TrainingFnArgs]
    ) -> tuple[torch.Tensor, descent.train.Trainable]: ...


_TRAINING_FNS_REGISTRY: dict[OptimiserName, TrainFn] = {}
"""Registry of training functions for different optimiser names."""

_register_training_fn = get_registry_decorator(_TRAINING_FNS_REGISTRY)


@_register_training_fn("lm")
def train_levenberg_marquardt(
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    trainable: descent.train.Trainable,
    topologies: list[smee.TensorTopology],
    datasets: list[datasets.Dataset],
    datasets_test: list[datasets.Dataset],
    settings: TrainingSettings,
    output_paths: dict[OutputType, PathLike],
    device: torch.device,
) -> tuple[torch.Tensor, descent.train.Trainable]:
    """
    Iterate the training process using the Levenberg-Marquardt algorithm.

    Parameters
    ----------
        trainable_parameters: torch.Tensor
            The parameters to be optimized.
        initial_parameters: torch.Tensor
            The initial parameters before training.
        trainable: descent.train.Trainable
            The trainable object containing the parameters.
        topologies: list[smee.TensorTopology]
            The topologies of the systems.
        datasets: list[datasets.Dataset]
            The datasets to be used for training.
        datasets_test: list[datasets.Dataset]
            The datasets to be used for testing.
        settings: TrainingSettings
            The settings object containing training parameters.
        output_paths: dict[OutputType, PathLike]
            A mapping of output types to filesystem paths. The following keys are
            expected:
                - OutputType.TENSORBOARD
                - OutputType.TRAINING_METRICS
        device: torch.device
            The device to perform training on.

    Returns
    -------
        tuple[torch.Tensor, descent.train.Trainable]
            The updated parameters and the trainable object.
    """
    # Make sure we have all the required output paths and no others
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    # Run the training with the LM optimiser
    lm_config = descent.optim.LevenbergMarquardtConfig(
        mode="adaptive", n_convergence_criteria=2, max_steps=settings.n_epochs
    )

    # Get loss weights - using default values
    # TODO: Support getting these from protocol settings when available
    loss_energy_weight = 1000.0
    loss_force_weight = 0.1

    closure_fn = get_loss_closure_fn(
        datasets,
        trainable,
        trainable_parameters,
        initial_parameters,
        topologies,
        loss_energy_weight,
        loss_force_weight,
        settings.regularisation_target,
    )

    correct_fn = trainable.clamp

    # Create report function that computes metrics consistently with train_adam
    report_fn = functools.partial(
        report,
        trainable=trainable,
        topologies=topologies,
        datasets_train=datasets,
        datasets_test=datasets_test,
        initial_parameters=initial_parameters,
        regularisation_target=settings.regularisation_target,
        metrics_file=output_paths[OutputType.TRAINING_METRICS],
        experiment_dir=Path(output_paths[OutputType.TENSORBOARD]),
    )

    trainable_parameters = descent.optim.levenberg_marquardt(
        trainable_parameters, lm_config, closure_fn, correct_fn, report_fn
    )
    trainable_parameters.requires_grad_(True)

    return trainable_parameters, trainable


@_register_training_fn("adam")
def train_adam(
    trainable_parameters: torch.Tensor,
    initial_parameters: torch.Tensor,
    trainable: descent.train.Trainable,
    topologies: list[smee.TensorTopology],
    datasets: list[datasets.Dataset],
    datasets_test: list[datasets.Dataset],
    settings: TrainingSettings,
    output_paths: dict[OutputType, PathLike],
    device: torch.device,
) -> tuple[torch.Tensor, descent.train.Trainable]:
    """
    Iterate the training process using the Adam optimizer.

    Parameters
    ----------
        trainable_parameters: torch.Tensor
            The parameters to be optimized.
        initial_parameters: torch.Tensor
            The initial parameters before training.
        trainable: descent.train.Trainable
            The trainable object containing the parameters.
        topologies: list[smee.TensorTopology]
            The topologies of the systems.
        datasets: list[datasets.Dataset]
            The datasets to be used for training.
        datasets_test: list[datasets.Dataset]
            The datasets to be used for testing.
        settings: TrainingSettings
            The settings object containing training parameters.
        output_paths: dict[OutputType, PathLike]
            A mapping of output types to filesystem paths. The following keys are
            expected:
                - OutputType.TENSORBOARD
                - OutputType.TRAINING_METRICS
        device: torch.device
            The device to perform training on.

    Returns
    -------
        tuple[torch.Tensor, descent.train.Trainable]
            The updated parameters and the trainable object.
    """
    # Make sure we have all the required output paths and no others
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    # run the ML training
    with open(output_paths[OutputType.TRAINING_METRICS], "w") as metrics_file:
        with open_writer(Path(output_paths[OutputType.TENSORBOARD])) as writer:
            optimizer = torch.optim.Adam(
                [trainable_parameters], lr=settings.learning_rate, amsgrad=True
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=settings.learning_rate_decay
            )
            for v in tensorboardX.writer.hparams(
                {"optimizer": "Adam", "lr": settings.learning_rate}, {}
            ):
                writer.file_writer.add_summary(v)
            for i in tqdm(
                range(settings.n_epochs),
                leave=False,
                colour="blue",
                desc="Optimising MM parameters",
            ):
                losses_train = prediction_loss(
                    datasets,
                    trainable,
                    trainable_parameters,
                    initial_parameters,
                    topologies,
                    settings.regularisation_target,
                    str(device),
                )
                tot_loss_train = sum(losses_train)

                logger.info(f"Epoch {i}: Training Weighted Loss: {losses_train} ")
                if i % 10 == 0:
                    losses_test = prediction_loss(
                        datasets_test,
                        trainable,
                        trainable_parameters,
                        initial_parameters,
                        topologies,
                        settings.regularisation_target,
                        str(device),
                    )

                    write_metrics(
                        i,
                        losses_train,
                        losses_test,
                        writer,
                        metrics_file,
                    )

                tot_loss_train.backward(retain_graph=False)  # type: ignore[union-attr]
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                trainable.clamp(trainable_parameters)

                if i % settings.learning_rate_decay_step == 0:
                    scheduler.step()

        # Required to avoid filling up the GPU memory between iterations
        # TODO: Find a better way to do this.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # some book-keeping and outputting
        losses_train = prediction_loss(
            datasets,
            trainable,
            trainable_parameters,
            initial_parameters,
            topologies,
            settings.regularisation_target,
            str(device),
        )
        losses_test = prediction_loss(
            datasets_test,
            trainable,
            trainable_parameters,
            initial_parameters,
            topologies,
            settings.regularisation_target,
            str(device),
        )

        write_metrics(
            settings.n_epochs, losses_train, losses_test, writer, metrics_file
        )

        return trainable_parameters, trainable
