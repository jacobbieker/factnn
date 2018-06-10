from sklearn.metrics import roc_auc_score, r2_score, roc_curve
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def plot_energy_confusion(prediction, truth, log_xy=True, log_z=True, ax=None):
    ax = ax or plt.gca()

    if log_xy is False:
        truth = np.log10(truth)
        prediction = np.log10(prediction)

    min_label = np.min(truth)
    min_pred = np.min(prediction)
    max_pred = np.max(prediction)
    max_label = np.max(truth)

    if min_label < min_pred:
        min_ax = min_label
    else:
        min_ax = min_pred

    if max_label > max_pred:
        max_ax = max_label
    else:
        max_ax = max_pred

    limits = [
        min_ax,
        max_ax
    ]
    print(limits)
    print("Max, min Label")
    print([min_label, max_label])

    counts, x_edges, y_edges, img = ax.hist2d(
        truth,
        prediction,
        bins=[100, 100],
        range=[limits, limits],
        norm=LogNorm() if log_z is True else None
    )
    ax.set_aspect(1)
    ax.figure.colorbar(img, ax=ax)

    if log_xy is True:
        ax.set_xlabel(r'$\log_{10}(E_{\mathrm{MC}} \,\, / \,\, \mathrm{GeV})$')
        ax.set_ylabel(r'$\log_{10}(E_{\mathrm{Est}} \,\, / \,\, \mathrm{GeV})$')
    else:
        ax.set_xlabel(r'$E_{\mathrm{MC}} \,\, / \,\, \mathrm{GeV}$')
        ax.set_ylabel(r'$E_{\mathrm{Est}} \,\, / \,\, \mathrm{GeV}$')

    return ax


def plot_disp_confusion(prediction, truth, log_xy=True, log_z=True, ax=None):
    ax = ax or plt.gca()

    if log_xy is True:
        truth = np.log10(truth)
        prediction = np.log10(prediction)

    min_label = np.min(truth)
    min_pred = np.min(prediction)
    max_pred = np.max(prediction)
    max_label = np.max(truth)

    if min_label < min_pred:
        min_ax = min_label
    else:
        min_ax = min_pred

    if max_label > max_pred:
        max_ax = max_label
    else:
        max_ax = max_pred

    limits = [
        min_ax,
        max_ax
    ]
    print(limits)
    print("Max, min Label")
    print([min_label, max_label])

    counts, x_edges, y_edges, img = ax.hist2d(
        truth,
        prediction,
        bins=[100, 100],
        range=[limits, limits],
        norm=LogNorm() if log_z is True else None
    )
    ax.set_aspect(1)
    ax.figure.colorbar(img, ax=ax)

    if log_xy is True:
        ax.set_xlabel(r'$log_{10}(Disp_{MC}) (mm)$')
        ax.set_ylabel(r'$log_{10}(Disp_{EST}) (mm)$')
    else:
        ax.set_xlabel(r'$Disp_{MC} (mm)$')
        ax.set_ylabel(r'$Disp_{EST} (mm)$')

    return ax


def plot_roc(truth, predictions, ax=None):
    ax = ax or plt.gca()

    ax.axvline(0, color='lightgray')
    ax.axvline(1, color='lightgray')
    ax.axhline(0, color='lightgray')
    ax.axhline(1, color='lightgray')

    mean_fpr, mean_tpr, _ = roc_curve(truth, predictions)

    ax.set_title('Area Under Curve: {:.4f}'.format(
        roc_auc_score(truth, predictions)
    ))

    ax.plot(mean_fpr, mean_tpr, label='ROC curve')
    ax.legend()
    ax.set_aspect(1)

    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.figure.tight_layout()

    return ax


def plot_probabilities(performace_df, model=None, ax=None, classnames=('Proton', 'Gamma')):
    ax = ax or plt.gca()

    bin_edges = np.linspace(0, 1, 100 + 2)
    ax.hist(
        performace_df,
        bins=bin_edges, label="Proton", histtype='step',
    )
    if model is not None:
        ax.hist(
            model,
            bins=bin_edges, label="Gamma", histtype='step',
        )

    ax.legend()
    ax.set_xlabel('Gamma confidence'.format(classnames[1]))
    ax.figure.tight_layout()

    return ax
