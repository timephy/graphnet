import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

import common


def roc_auc(y_pred, y_true) -> Tuple[pd.DataFrame, float]:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    df = pd.DataFrame(list(zip(fpr, tpr)), columns=['fpr', 'tpr'])
    return df, auc


def resolution_data(target, results):
    results['energy_log10'] = np.log10(results['energy'])

    x = []
    y = []
    ranges = np.arange(0, 3.1, 0.1)
    for i in range(0, len(ranges)-1):
        min = ranges[i]
        max = ranges[i+1]
        idx = (results['energy_log10'] > min) & (results['energy_log10'] < max)
        data_sliced = results.loc[idx, :].reset_index(drop=True)

        x.append(np.mean(data_sliced['energy_log10']))
        y.append(resolution_fn(data_sliced['R']))

    # return x, y
    return pd.DataFrame(list(zip(x, y)), columns=['energy', 'resolution'])


def resolution_fn(r):
    return (np.percentile(r, 84) - np.percentile(r, 16)) / 2.


def plot_roc_auc(
    dfs: List[pd.DataFrame],
    aucs: List[float],
    labels: List[str],
    *,
    path_out: str,
    title: str = 'Classification'
):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1])  # , 'k–'

    for df, auc, label in zip(dfs, aucs, labels):
        plt.plot(df['fpr'], df['tpr'], label=f'{label} (AUC={auc:.3f})')

    plt.xlabel('False positive rate', fontsize=16)
    plt.ylabel('True positive rate', fontsize=16)
    plt.legend(loc='lower right')
    plt.title(title, fontsize=16)
    plt.savefig(path_out)


def plot_resolution(
    dfs: List[pd.DataFrame],
    labels: List[str],
    *,
    path_out: str,
    title: str = 'Resolution',
    xlabel: str = 'Energy log10 GeV',
    ylabel: str = 'Resolution',
):
    plt.figure(figsize=(8, 6))

    for df, label in zip(dfs, labels):
        plt.plot(df['energy'], df['resolution'], linestyle='solid', label=label)

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)
    plt.legend(loc='lower right')
    plt.savefig(path_out)


def plot_scatter(args: common.Args, y_pred, y_true, scale='linear'):
    max = np.max([y_pred.max(), y_true.max()])

    plt.figure(figsize=(8, 6))
    plt.plot([0, max], [0, max])  # , 'k–'

    plt.scatter(y_pred, y_true, marker='.')  # type: ignore

    plt.xlabel('Predicted Value', fontsize=16)
    plt.ylabel('Actual Value', fontsize=16)
    plt.xscale(scale)
    plt.yscale(scale)
    plt.legend(loc='lower right')
    plt.title(f'{args.target.capitalize()} Regression', fontsize=16)
    plt.savefig(str(args.archive.root.joinpath('scatter.png')))


def plot_metrics_combined(args_list: List[common.Args], *, path_out: str):
    assert len(args_list) > 0

    if args_list[0].target == 'track':
        dfs = [pd.read_csv(args.archive.roc_csv_str) for args in args_list]
        aucs = [np.load(args.archive.auc_file_str) for args in args_list]
        labels = [args.run_name for args in args_list]
        plot_roc_auc(
            dfs,
            aucs,
            labels,
            path_out=path_out
        )

    elif args_list[0].target == 'energy':
        dfs = [pd.read_csv(args.archive.resolution_csv_str) for args in args_list]
        labels = [args.run_name for args in args_list]
        plot_resolution(
            dfs,
            labels,
            title='Energy Regression',
            ylabel='Energy Resolution (%)',
            path_out=path_out
        )

    elif args_list[0].target == 'zenith':
        dfs = [pd.read_csv(args.archive.resolution_csv_str) for args in args_list]
        labels = [args.run_name for args in args_list]
        plot_resolution(
            dfs,
            labels,
            title='Zenith Regression',
            ylabel='Zenith Resolution (deg)',
            path_out=path_out
        )


def plot_metrics(args: common.Args):
    if args.target == 'track':
        plot_metrics_combined(
            [args],
            path_out=args.archive.roc_auc_plot_str
        )

    elif args.target == 'energy':
        plot_metrics_combined(
            [args],
            path_out=args.archive.resolution_plot_str
        )

    elif args.target == 'zenith':
        plot_metrics_combined(
            [args],
            path_out=args.archive.resolution_plot_str
        )


def generate_metrics(args: common.Args):
    print()
    print(f'test_results({args.target=})')

    results = pd.read_csv(args.archive.results_str)

    if args.target == 'track':
        y_pred = results[args.target + '_pred']
        y_pred_tag = y_pred.round()
        y_true = results[args.target]

        df, auc = roc_auc(y_pred, y_true)
        df.to_csv(args.archive.roc_csv_str)
        np.save(args.archive.auc_file_str, auc)
        # plot_roc_auc(args, df, auc)

        print(confusion_matrix(y_true, y_pred_tag))
        print(classification_report(y_true, y_pred_tag))

    elif args.target == 'energy':
        results['R'] = results[args.target + '_pred'] - results[args.target]
        results['R'] *= 100 / results[args.target]

        df = resolution_data(args.target, results)
        df.to_csv(args.archive.resolution_csv_str)
        # plot_resolution(args, df)

    elif args.target == 'zenith':
        results['R'] = results[args.target + '_pred'] - results[args.target]
        results['R'] *= 360 / (2*np.pi)

        df = resolution_data(args.target, results)
        df.to_csv(args.archive.resolution_csv_str)
        # plot_resolution(args, df)
