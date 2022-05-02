from itertools import groupby
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.components.loss_functions import BinaryCrossEntropyLoss, LogCoshLoss, VonMisesFisher2DLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.pipeline import InSQLitePipeline
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import PassOutput1, BinaryClassificationTask, EnergyReconstruction, ZenithReconstruction, ZenithReconstructionWithKappa
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse

import common
import nnb
from train_test import train_test
from metrics import test_results


# Configurations
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
_features = FEATURES.DEEPCORE
_truth = TRUTH.DEEPCORE

try:
    del _truth[_truth.index('interaction_time')]
except ValueError:
    # not found in list
    pass


def convert_model(args: common.Args):
    print()
    print(f'===== convert_model({args.target=}) =====')

    # Building model
    _, get_parts = get_funcs(args)
    model = get_parts(args, train_dataloader=[]).model

    model.load_state_dict(args.archive.state_dict_str)
    model.save(args.archive.model_str)


def run_pipeline(args_list: list[common.Args], args_settings: common.Args):
    print(f'===== run_pipeline() =====')

    def get_output_column_names(target: common.Target):
        if target in ['azimuth', 'zenith']:
            return [target + '_pred', target + '_kappa']
        elif target in ['track', 'neutrino', 'energy']:
            return [target + '_pred']
        else:
            raise Exception('target not found')

    def build_module_dictionary(args_list: list[common.Args]):
        module_dict = {}
        for args in args_list:
            module_dict[args.target] = {}
            module_dict[args.target]['path'] = args.archive.model_str
            module_dict[args.target]['output_column_names'] = get_output_column_names(args.target)
        return module_dict

    # Build Pipeline
    pipeline = InSQLitePipeline(
        module_dict=build_module_dictionary(args_list),
        features=args_settings.features,
        truth=args_settings.truth,
        device=f'cuda:{args_settings.gpus[0]}',
        batch_size=args_settings.batch_size,
        n_workers=args_settings.num_workers,
        pipeline_name='pipeline',
        outdir='pipeline_results'
    )

    # Run Pipeline
    pipeline(args_settings.database_str, args_settings.pulsemap)


def main():
    # Config
    targets_all: list[common.Target] = ['track', 'energy', 'zenith']
    run_names_all = ['8nn', '4nn', '3nn', '2nn']

    pipeline_name = 'pipeline_tim_0'

    # Parser
    parser = argparse.ArgumentParser(
        description='A script to train, test, generate metrics and run pipelines for variations of graphnet.')

    parser.add_argument('-f', dest='functions', nargs='+',
                        default=['train_test', 'metrics', 'pipeline'],
                        help='what functions to run on targets')
    parser.add_argument('-t', dest='targets', nargs='+',
                        default=targets_all,
                        help='what targets to run functions on')
    parser.add_argument('-n', dest='run_names', nargs='+',
                        required=True,
                        help='what targets to run functions on')

    args = parser.parse_args()
    print(f'{args=}')

    targets = args.targets
    functions = args.functions
    run_names = args.run_names

    # Run
    args_dict = {}
    for run_name in run_names_all:
        args_dict[run_name] = {}
        for target in targets_all:
            args_dict[run_name][target] = common.Args(
                run_name=run_name,
                target=target,

                database=Path('/remote/ceph/user/t/timg/dev_lvl7_robustness_muon_neutrino_0000.db'),
                pulsemap='SRTTWOfflinePulsesDC',
                features=_features,
                truth=_truth,

                batch_size=512,
                num_workers=30,
                gpus=[3],

                max_epochs=50,
                patience=5,

                archive=common.Archive(Path(f'/remote/ceph/user/t/timg/archive/{run_name}/{target}'),)
            )

    for run_name in run_names:
        for target in targets:
            args = args_dict[run_name][args]
            get_dataloaders, get_parts = get_funcs(args)

            for target in targets:
                if 'convert_model' in functions:
                    convert_model(args)
                if 'train_test' in functions:
                    train_test(args, get_dataloaders, get_parts)
                if 'metrics' in functions:
                    test_results(args)

        if 'pipeline' in functions:
            for args_list in args_dict[run_name]:
                # Use first item as settings
                args_settings = args_list[0]
                run_pipeline(args_list, args_settings)


def get_funcs(args: common.Args):
    if args.run_name == '8nn':
        get_dataloaders = nnb.get_dataloaders
        get_parts = lambda *args, **kwargs: nnb.get_parts(*args, **kwargs, nb_nearest_neighbours=8)
    elif args.run_name == '5nn':
        get_dataloaders = nnb.get_dataloaders
        get_parts = lambda *args, **kwargs: nnb.get_parts(*args, **kwargs, nb_nearest_neighbours=5)
    elif args.run_name == '4nn':
        get_dataloaders = nnb.get_dataloaders
        get_parts = lambda *args, **kwargs: nnb.get_parts(*args, **kwargs, nb_nearest_neighbours=4)
    elif args.run_name == '3nn':
        get_dataloaders = nnb.get_dataloaders
        get_parts = lambda *args, **kwargs: nnb.get_parts(*args, **kwargs, nb_nearest_neighbours=3)
    elif args.run_name == '2nn':
        get_dataloaders = nnb.get_dataloaders
        get_parts = lambda *args, **kwargs: nnb.get_parts(*args, **kwargs, nb_nearest_neighbours=2)
    else:
        raise Exception('run_name not found')

    return get_dataloaders, get_parts


if __name__ == '__main__':
    main()
