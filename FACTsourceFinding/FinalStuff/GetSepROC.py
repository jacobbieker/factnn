import pickle
from keras import backend as K
import h5py
from fact.io import read_h5py
import yaml
import tensorflow as tf
import os
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy import savetxt, loadtxt, round, zeros, sin, cos, arctan2, clip, pi, tanh, exp, arange, dot, outer, array, shape, zeros_like, reshape, mean, median, max, min
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera
from sklearn.metrics import roc_auc_score,  roc_curve, r2_score
from sklearn import metrics


rf = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/separator_performance.hdf5", key='data')

labels = rf['label']
predictions = rf['label_prediction']

labels = labels.reshape(-1,1)
y_test = labels
y_score = predictions
print(labels.shape)
print(y_score.shape)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_score)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve AUC: {:.4f}'.format(roc_auc_score(y_test, y_score)))
plt.show()


rf = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/cv_disp.hdf5", key='data')

labels = rf['disp']
predictions = rf['disp_prediction']

labels = labels.reshape(-1,1)
y_label = labels.reshape(-1,)
predictions_x = predictions.reshape((-1,))
print(labels.shape)
print(y_score.shape)
score = r2_score(y_label, predictions_x)

def plot_sourceX_Y_confusion(performace_df, label, log_xy=True, log_z=True, ax=None):

    ax = ax or plt.gca()

    #label = performace_df.label.copy()
    prediction = performace_df.copy()

    if log_xy is False:
        label = np.log10(label)
        prediction = np.log10(prediction)

    min_label = np.min(label)
    min_pred = np.min(prediction)
    max_pred = np.max(prediction)
    max_label = np.max(label)

    if min_label < min_pred:
        min_ax = min_label
    else:
        min_ax = min_pred

    if max_label > max_pred:
        max_ax = max_label
    else:
        max_ax = max_pred

    if log_z:
        min_ax = min_label
        max_ax = max_label
    limits = [
        min_ax,
        max_ax
    ]
    print(limits)
    print("Max, min Label")
    print([min_label, max_label])

    counts, x_edges, y_edges, img = ax.hist2d(
        label,
        prediction,
        bins=[100, 100],
        range=[limits, limits],
        norm=LogNorm() if log_xy is False else None
    )
    ax.set_aspect(1)
    ax.figure.colorbar(img, ax=ax)

    if log_xy is True:
        ax.set_xlabel(r'$X$')
        ax.set_ylabel(r'$Y$')
    else:
        ax.set_xlabel(r'$X_{\mathrm{MC}}$')
        ax.set_ylabel(r'$Y_{\mathrm{Est}}$')

    return ax


fig1 = plt.figure()
ax = fig1.add_subplot(1, 1, 1)
ax.set_title(" R^2: " + str(score))
plot_sourceX_Y_confusion(predictions_x, y_label, ax=ax)
fig1.show()

fig1 = plt.figure()
ax = fig1.add_subplot(1, 1, 1)
ax.set_title(" R^2: " + str(score))
plot_sourceX_Y_confusion(predictions_x, y_label, log_xy=False, ax=ax)
fig1.show()