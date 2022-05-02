from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from graphnet.data.utils import get_desired_event_numbers, get_even_track_cascade_indicies
from graphnet.models.training.utils import get_predictions as _get_predictions, make_train_validation_dataloader, make_dataloader
from graphnet.components.loss_functions import BinaryCrossEntropyLoss, LogCoshLoss, VonMisesFisher2DLoss
from graphnet.models.task.reconstruction import PassOutput1, BinaryClassificationTask, EnergyReconstruction, ZenithReconstruction, ZenithReconstructionWithKappa

from sklearn.model_selection import train_test_split


Target = Literal["track", "energy", "zenith"]


@dataclass
class Archive:
    path: Path

    @property
    def root_str(self):
        return str(self.path.absolute())

    @property
    def state_dict_str(self):
        return str(self.path.joinpath('state_dict.pth').absolute())

    @property
    def model_str(self):
        return str(self.path.joinpath('model.pth').absolute())

    @property
    def results_str(self):
        return str(self.path.joinpath('results.csv').absolute())

    @property
    def roc_auc_str(self):
        return str(self.path.joinpath('roc_auc.png').absolute())

    @property
    def resolution_str(self):
        return str(self.path.joinpath('resolution.png').absolute())


@dataclass
class Args:
    run_name: str  # nnb-8, nnb-4, ...
    target: Target  # track, energy, zenith

    database: Path
    pulsemap: str
    features: list[str]
    truth: list[str]

    batch_size: int
    num_workers: int
    gpus: list[int]

    max_epochs: int
    patience: int

    archive: Archive

    @property
    def database_str(self):
        return str(self.database.absolute())


@dataclass
class Values:
    detector: ...
    gnn: ...
    task: ...
    model: ...


def get_predictions(*, target: Target, trainer, model, test_dataloader):
    if target == 'track':
        return _get_predictions(
            trainer,
            model,
            test_dataloader,
            prediction_columns=[target + '_pred'],
            additional_attributes=[target, 'event_no', 'energy'],
        )

    elif target == 'energy':
        return _get_predictions(
            trainer,
            model,
            test_dataloader,
            prediction_columns=[target + '_pred'],
            additional_attributes=[target, 'event_no'],
        )

    elif target == 'zenith':
        return _get_predictions(
            trainer,
            model,
            test_dataloader,
            prediction_columns=[target + '_pred', target + '_kappa'],
            additional_attributes=[target, 'event_no', 'energy'],
        )

    else:
        raise Exception('target does not match')


def get_task(*, target: Target, gnn):
    if target == 'track':
        return BinaryClassificationTask(
            hidden_size=gnn.nb_outputs,
            target_labels=target,
            loss_function=BinaryCrossEntropyLoss(),
        )

    elif target == 'energy':
        # task = EnergyReconstruction(
        #     hidden_size=gnn.nb_outputs,
        #     target_labels=target,
        #     loss_function=LogCoshLoss(),
        #     transform_prediction_and_target=torch.log10,
        # )
        return PassOutput1(
            hidden_size=gnn.nb_outputs,
            target_labels=target,
            loss_function=LogCoshLoss(),
            transform_target=torch.log10,
            transform_inference=lambda x: torch.pow(10, x)
        )

    elif target == 'zenith':
        return ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=target,
            loss_function=VonMisesFisher2DLoss(),
        )

    else:
        raise Exception('target does not match')


def get_selections(args: Args):
    if args.target == 'track':
        train_valid_selection, test_selection = get_even_track_cascade_indicies(args.database)

    elif args.target == 'energy' or args.target == 'zenith':
        selection = get_desired_event_numbers(
            args.database,
            10000000000,
            fraction_muon=0, fraction_nu_e=0.34, fraction_nu_mu=0.33, fraction_nu_tau=0.33  # type: ignore
        )
        train_valid_selection, test_selection = train_test_split(selection, test_size=0.25, random_state=42)

    else:
        raise Exception('target does not match')

    return train_valid_selection, test_selection


def get_dataloaders(args: Args):
    train_valid_selection, test_selection = get_selections(args)

    training_dataloader, validation_dataloader = make_train_validation_dataloader(  # type: ignore
        db=args.database_str,
        selection=train_valid_selection,
        pulsemaps=args.pulsemap,
        features=args.features,
        truth=args.truth,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.33,
    )
    test_dataloader = make_dataloader(
        db=args.database_str,
        pulsemaps=args.pulsemap,
        features=args.features,
        truth=args.truth,
        selection=test_selection,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return training_dataloader, validation_dataloader, test_dataloader
