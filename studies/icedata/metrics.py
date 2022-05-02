import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

import common


def roc_auc(y_pred, y_true):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return fpr, tpr, auc


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

    return x, y


def resolution_fn(r):
    return (np.percentile(r, 84) - np.percentile(r, 16)) / 2.


def plot_roc_auc(target, path_out, run_name, fpr, tpr, auc):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1])  # , 'k–'
    plt.plot(fpr, tpr, label='Testing (AUC = {:.3f})'.format(auc))
    plt.xlabel('False positive rate', fontsize=16)
    plt.ylabel('True positive rate', fontsize=16)
    plt.legend(loc='lower right')
    plt.title(f'{target.capitalize()} Classification', fontsize=16)
    plt.savefig(f'{path_out}/{run_name}_roc-auc.jpg')


def plot_resolution(target, path_out, x, y, run_name):
    plt.figure(figsize=(8, 6))

    plt.plot(x, y, linestyle='solid', lw=0.5, color='black', alpha=1)

    plt.xlabel('Energy log10 GeV', fontsize=16)
    plt.ylabel(f'{target.capitalize()} Resolution', fontsize=16)
    plt.legend(loc='lower right')
    plt.title(f'{target.capitalize()} Regression', fontsize=16)
    plt.savefig(f'{path_out}/{run_name}_resolution.jpg')


def plot_scatter(target, path_out, y_pred, y_true, run_name, scale='linear'):
    max = np.max([y_pred.max(), y_true.max()])

    plt.figure(figsize=(8, 6))
    plt.plot([0, max], [0, max])  # , 'k–'

    plt.scatter(y_pred, y_true, marker='.')  # type: ignore

    plt.xlabel('Predicted Value', fontsize=16)
    plt.ylabel('Actual Value', fontsize=16)
    plt.xscale(scale)
    plt.yscale(scale)
    plt.legend(loc='lower right')
    plt.title(f'{target.capitalize()} Regression', fontsize=16)
    plt.savefig(f'{path_out}/{run_name}_scatter.jpg')


def test_results(args: common.Args):
    print()
    print(f'===== test_results({args.target=}) =====')

    results = pd.read_csv(args.archive.results_str)

    path_out = 'test_results'
    os.makedirs(path_out, exist_ok=True)

    if args.target == 'track':
        y_pred = results[args.target + '_pred']
        y_pred_tag = y_pred.round()
        y_true = results[args.target]

        fpr, tpr, auc = roc_auc(y_pred, y_true)
        plot_roc_auc(args.target, path_out, args.run_name, fpr, tpr, auc)

        print(confusion_matrix(y_true, y_pred_tag))
        print(classification_report(y_true, y_pred_tag))

    elif args.target == 'energy':
        results['R'] = results[args.target + '_pred'] - results[args.target]
        results['R'] *= 100 / results[args.target]

        x, y = resolution_data(args.target, results)
        plot_resolution(args.target, path_out, x, y, args.run_name)

    elif args.target == 'zenith':
        results['R'] = results[args.target + '_pred'] - results[args.target]
        results['R'] *= 360 / (2*np.pi)

        x, y = resolution_data(args.target, results)
        plot_resolution(args.target, path_out, x, y, args.run_name)

    else:
        raise Exception('target does not match')
