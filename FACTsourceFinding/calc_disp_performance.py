import os
# to force on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# y in radians and second column is az, first is zd, adds the errors together, seems to work?
def rmse_360_2(y_true, y_pred):
    az_error = tf.reduce_mean(K.abs(tf.atan2(K.sin(y_true[:,1] - y_pred[:,1]), K.cos(y_true[:,1] - y_pred[:,1]))))
    zd_error = K.mean(K.square(y_pred[:,0] - y_true[:,0]), axis=-1)
    return az_error + zd_error

import keras.losses
keras.losses.rmse_360_2 = rmse_360_2

# load

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'


#All files for calculating ROC and AUC on, for phi and theta for simulated, Az, Zd for all, Energy for simulated, Ra, Dec for observation
path_diffuse_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_diffuse_BothSource_Images.h5"
path_mc_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_proton_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Proton_BothTracking_Images.h5"
path_crab_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_diffuse_BothSource_Images.h5"
path_mrk501_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_mrk501_preprocessed_images.h5"

disp_models = "/run/media/jacob/WDRed8Tb1/Models/Disp/"
energy_models = "/run/media/jacob/WDRed8Tb1/Models/Energy/"
sep_models = "/run/media/jacob/WDRed8Tb1/Models/Sep/"

phiTheta_pickle = "phiTheta_auc.p"
sourceXY_pickle = "sourceXY_auc.p"
sep_pickle = "sep.p"
azzd_pickle = "azzd.p"
energy_pickle = "energy_pickle"

best_auc_sep = []
best_auc_sep_auc = []

best_energy = []
best_energy_auc = []

best_azzd = []
best_azzd_auc = []

best_xy = []
best_xy_auc = []

def plot_sourceX_Y_confusion(performace_df, label, log_xy=True, log_z=True, ax=None):

    ax = ax or plt.gca()

    #label = performace_df.label.copy()
    prediction = performace_df.copy()

    if log_xy is True:
        label = np.log10(label)
        prediction = np.log10(prediction)

    min_label = np.floor(np.min(label))
    min_pred = np.floor(np.min(prediction))
    max_pred = np.ceil(np.max(prediction))
    max_label = np.ceil(np.max(label))

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

    counts, x_edges, y_edges, img = ax.hist2d(
        label,
        prediction,
        bins=[100, 100],
        range=[limits, limits],
        norm=LogNorm() if log_z is True else None,
    )
    ax.set_aspect(1)
    ax.figure.colorbar(img, ax=ax)

    if log_xy is True:
        ax.set_xlabel(r'$\log_{10}(StupidX_{\mathrm{MC}} \,\, / \,\, \mathrm{GeV})$')
        ax.set_ylabel(r'$\log_{10}(StupidY_{\mathrm{Est}} \,\, / \,\, \mathrm{GeV})$')
    else:
        ax.set_xlabel(r'$StupidX_{\mathrm{MC}} \,\, / \,\, \mathrm{GeV}$')
        ax.set_ylabel(r'$StupidY_{\mathrm{Est}} \,\, / \,\, \mathrm{GeV}$')

    return ax

def plot_regressor_confusion(performace_df, label, log_xy=True, log_z=True, ax=None):

    ax = ax or plt.gca()

    #label = performace_df.label.copy()
    prediction = performace_df.copy()

    if log_xy is True:
        label = np.log10(label)
        prediction = np.log10(prediction)

    min_label = np.floor(np.min(label))
    min_pred = np.floor(np.min(prediction))
    max_pred = np.ceil(np.max(prediction))
    max_label = np.ceil(np.max(label))

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

    counts, x_edges, y_edges, img = ax.hist2d(
        label,
        prediction,
        bins=[100, 100],
        range=[limits, limits],
        norm=LogNorm() if log_z is True else None,
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

def plot_azZd_confusion(performace_df, label, log_xy=False, log_z=True, ax=None):

    ax = ax or plt.gca()

    #label = performace_df.label.copy()
    prediction = performace_df.copy()
    print(prediction)
    print(label)

    if log_xy is True:
        label = label[:,0]
        prediction = prediction[:,0]
    else:
        label = label[:,1]
        prediction = prediction[:,1]

    min_label = np.floor(np.min(label))
    min_pred = np.floor(np.min(prediction))
    max_pred = np.ceil(np.max(prediction))
    max_label = np.ceil(np.max(label))

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

    counts, x_edges, y_edges, img = ax.hist2d(
        label,
        prediction,
        bins=[100, 100],
        range=[limits, limits],
        norm=LogNorm() if log_z is True else None
    )
    ax.set_aspect(1)
    ax.figure.colorbar(img, ax=ax)

    if log_xy is True:
        ax.set_xlabel(r'$Zd_{\mathrm{MC}} \,\, / \,\, \mathrm{Deg}$')
        ax.set_ylabel(r'$Zd_{\mathrm{Est}} \,\, / \,\, \mathrm{Deg}$')
    else:
        ax.set_xlabel(r'$Az_{\mathrm{MC}} \,\, / \,\, \mathrm{Deg}$')
        ax.set_ylabel(r'$Az_{\mathrm{Est}} \,\, / \,\, \mathrm{Deg}$')

    return ax

def plot_phiTheta_confusion(performace_df, label, log_xy=False, log_z=True, ax=None):

    ax = ax or plt.gca()

    #label = performace_df.label.copy()
    prediction = performace_df.copy()
    print(prediction)
    print(label)

    if log_xy is True:
        label = label[:,0]
        prediction = prediction[:,0]
    else:
        label = label[:,1]
        prediction = prediction[:,1]

    min_label = np.floor(np.min(label))
    min_pred = np.floor(np.min(prediction))
    max_pred = np.ceil(np.max(prediction))
    max_label = np.ceil(np.max(label))

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

    counts, x_edges, y_edges, img = ax.hist2d(
        label,
        prediction,
        bins=[100, 100],
        range=[limits, limits],
        norm=LogNorm() if log_z is True else None
    )
    ax.set_aspect(1)
    ax.figure.colorbar(img, ax=ax)

    if log_xy is True:
        ax.set_xlabel(r'$Theta_{\mathrm{MC}} \,\, / \,\, \mathrm{Deg}$')
        ax.set_ylabel(r'$Theta_{\mathrm{Est}} \,\, / \,\, \mathrm{Deg}$')
    else:
        ax.set_xlabel(r'$Phi_{\mathrm{MC}} \,\, / \,\, \mathrm{Deg}$')
        ax.set_ylabel(r'$Phi_{\mathrm{Est}} \,\, / \,\, \mathrm{Deg}$')

    return ax

def plot_sourceXY_confusion(performace_df, label, log_xy=False, log_z=True, ax=None):

    ax = ax or plt.gca()

    #label = performace_df.label.copy()
    prediction = performace_df.copy()
    print("Prediction 0: ")
    print(prediction[0])
    print("Prediction 1: ")
    print(prediction[1])
    print("Label 0: ")
    print(label[0])
    print("Label 1: ")
    print(label[1])
    print("Prediction [:,0]")
    print(prediction[:,0])
    print("Prediction[:,1]")
    print(prediction[:,1])
    print("Label [:,0]")
    print(label[:,0])
    print("Label[:,1]")
    print(label[:,1])
    if log_xy is True:
        label = label[:,0]
        prediction = prediction[:,0]
    else:
        label = label[:,1]
        prediction = prediction[:,1]

    min_label = np.floor(np.min(label))
    min_pred = np.floor(np.min(prediction))
    max_pred = np.ceil(np.max(prediction))
    max_label = np.ceil(np.max(label))

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

    counts, x_edges, y_edges, img = ax.hist2d(
        label,
        prediction,
        bins=[100, 100],
        range=[limits, limits],
        norm=LogNorm() if log_z is True else None
    )
    ax.set_aspect(1)
    ax.figure.colorbar(img, ax=ax)

    if log_xy is True:
        ax.set_xlabel(r'$X_{\mathrm{MC}} \,\, / \,\, \mathrm{Pixels}$')
        ax.set_ylabel(r'$X_{\mathrm{Est}} \,\, / \,\, \mathrm{Pixels}$')
    else:
        ax.set_xlabel(r'$Y_{\mathrm{MC}} \,\, / \,\, \mathrm{Pixels}$')
        ax.set_ylabel(r'$Y_{\mathrm{Est}} \,\, / \,\, \mathrm{Pixels}$')

    return ax

def calc_roc_gammaHad(path_image, proton_image, path_keras_model):
    '''
    Returns ROC for gamma/hadron separation, also builds migration matrix for it and the plots of confidence
    :param path_image:
    :param path_keras_model:
    :return:
    '''

    model = load_model(path_keras_model)

    with h5py.File(path_image, 'r') as f:
        with h5py.File(proton_image, 'r') as f2:
            # stream the data in to predict on it, safe region for all is the last 40%, except for the latest sep models
            items = len(f['Image'])
            items_proton = len(f2['Image'])
            test_images = f['Image'][-int(items*.4):-1]
            test_images_false = f2['Image'][-int(items*.4):-1]
            # Since the mc AzZd ones are in radians, need these to be to, could convert back if need be later
            validating_dataset = np.concatenate([test_images, test_images_false], axis=0)
            labels = np.array([True] * (len(test_images)) + [False] * len(test_images_false))
            validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
            print(labels.shape)

            predictions = model.predict(validating_dataset, batch_size=64)
            print(predictions.shape)

            # get ROC and AUC score
            print(roc_auc_score(validation_labels, predictions))
            best_auc_sep.append(path_keras_model)
            best_auc_sep.append(roc_auc_score(validation_labels, predictions))
            with open(sep_pickle, "wb") as f:
                pickle.dump([best_auc_sep, best_auc_sep_auc], f)

    # TODO Change this to work with my simulated ones, this uses both hadron and gamma truths
    '''
    plt.style.use('ggplot')
    bins = np.arange(0,1.01,0.01)
    ax = h_df.hist(['Gamma'], bins=bins, alpha=0.75, color=(238/255, 129/255, 10/255), figsize=(5, 3))
    g_df.hist(['Gamma'], bins=bins, ax=ax, alpha=0.75, color=(118/255, 157/255, 6/255))
    plt.yscale('log')
    plt.legend(['Hadron', 'Gamma'], loc='lower center')
    plt.title('Prediction of simulated events with AUC {:.3f}'.format(stop_auc))
    plt.xlabel('Gamma ray probability')
    plt.ylabel('Event count')
    plt.tight_layout()
    plt.savefig(path_build+'CNN_MC_Evaluation.pdf')

    # TODO Change this to work with my predicitons df.hist['Gamma'] is the gamma confidence for real observation
    plt.style.use('ggplot')
    df.hist(['Gamma'], bins=100, color=(118/255, 157/255, 6/255), figsize=(5, 3))
    plt.yscale('log')
    plt.title('Prediction of real events')
    plt.xlabel('Gamma ray probability')
    plt.ylabel('Event count')
    plt.tight_layout()
    plt.savefig(path_build+'CNN_Real_Evaluation.pdf')
    '''
    return NotImplemented

def calc_roc_azzd(path_image, path_keras_model):
    '''
    Returns the Number of events in the threshold for the Az, Zd estimation for simulated events, for observation this means that it is in comparsion to current classifier
    :param path_image:
    :param path_keras_model:
    :return:
    '''


    model = load_model(path_keras_model)

    with h5py.File(path_image, 'r') as f:
        # stream the data in to predict on it, safe region for all is the last 40%, except for the latest sep models
        items = len(f['Image'])
        test_images = f['Image'][-int(items*.4):-1]
        labels_az = f['Source_Zd'][-int(items*.4):-1]
        labels_zd = f['Source_Az'][-int(items*.4):-1]
        # Since the mc AzZd ones are in radians, need these to be to, could convert back if need be later
        labels_az = np.deg2rad(labels_az)
        labels_zd = np.deg2rad(labels_zd)
        labels = np.column_stack((labels_zd, labels_az))
        print(labels.shape)

        predictions = model.predict(test_images, batch_size=64)
        print(predictions.shape)
        predictions = predictions
    # Now make the confusion matrix

    #Loss Score so can tell which one it is
    filename = path_keras_model.split("/")[-1]
    filename = filename.split("_")[0]
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(filename + ' Reconstructed vs. True Az')
    plot_azZd_confusion(predictions, labels, ax=ax)
    fig1.show()

    # Plot confusion
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.set_title(filename + 'Reconstructed vs. True Zd')
    plot_azZd_confusion(predictions, labels, log_xy=True, ax=ax)
    fig2.show()

    return NotImplemented

def calc_roc_thetaphi(path_image, path_keras_model):
    '''
    Returns the Number of events in the threshold for the Az, Zd estimation for simulated events, for observation this means that it is in comparsion to current classifier
    :param path_image:
    :param path_keras_model:
    :return:
    '''


    model = load_model(path_keras_model)

    with h5py.File(path_image, 'r') as f:
        # stream the data in to predict on it, safe region for all is the last 40%, except for the latest sep models
        items = len(f['Image'])
        test_images = f['Image'][-int(items*.4):-1]
        labels_az = f['Phi'][-int(items*.4):-1]
        labels_zd = f['Theta'][-int(items*.4):-1]
        # Since the mc AzZd ones are in radians, need these to be to, could convert back if need be later
        labels_az = np.deg2rad(labels_az)
        labels_zd = np.deg2rad(labels_zd)
        labels = np.column_stack((labels_zd, labels_az))
        print(labels.shape)

        predictions = model.predict(test_images, batch_size=64)
        print(predictions.shape)
        predictions = predictions
    # Now make the confusion matrix

    #Loss Score so can tell which one it is
    filename = path_keras_model.split("/")[-1]
    filename = filename.split("_")[0]
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(filename + ' Reconstructed vs. True Theta')
    plot_phiTheta_confusion(predictions, labels, ax=ax)
    fig1.show()

    # Plot confusion
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.set_title(filename + 'Reconstructed vs. True Phi')
    plot_phiTheta_confusion(predictions, labels, log_xy=True, ax=ax)
    fig2.show()

    return NotImplemented

def calc_roc_sourceXY(path_image, path_keras_model):
    '''
    Returns the Number of events in the threshold for the Az, Zd estimation for simulated events, for observation this means that it is in comparsion to current classifier
    :param path_image:
    :param path_keras_model:
    :return:
    '''


    model = load_model(path_keras_model)

    with h5py.File(path_image, 'r') as f:
        # stream the data in to predict on it, safe region for all is the last 40%, except for the latest sep models
        items = len(f['Image'])
        test_images = f['Image'][-int(items*.4):-1]
        source_x = f['Source_X'][-int(items*.4):-1]
        source_y = f['Source_Y'][-int(items*.4):-1]
        source_x += 180.975 # shifts everything to positive
        source_y += 185.25 # shifts everything to positive
        source_x = source_x / 4.94 # Ratio between the places
        source_y = source_y / 4.826 # Ratio between y in original and y here
        # Since the mc AzZd ones are in radians, need these to be to, could convert back if need be later
        labels = np.column_stack((source_x, source_y))
        print(labels.shape)
        predictions = model.predict(test_images, batch_size=64)
        print(predictions.shape)
        predictions = predictions
    # Now make the confusion matrix

    #Loss Score so can tell which one it is
    filename = path_keras_model.split("/")[-1]
    filename = filename.split("_")[0]
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(filename + ' Reconstructed vs. True X')
    plot_sourceX_Y_confusion(predictions[:,0], predictions[:,1], ax=ax)
    fig1.show()

    # Plot confusion
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.set_title(filename + 'Reconstructed vs. True Y')
    plot_sourceXY_confusion(labels[:,0], labels[:,0], log_xy=True, ax=ax)
    fig2.show()

    error = predictions - labels
    best_xy.append(path_keras_model)
    best_xy_auc.append([np.mean(error, axis=1), np.mean(error)])

    with open(sourceXY_pickle, "wb") as f:
        pickle.dump([best_xy, best_xy_auc], f)

    return NotImplemented

def calc_roc_energy(path_image, path_keras_model):
    '''
    Returns the ROC for the energy estimation
    :param path_image:
    :param path_keras_model:
    :return:
    '''

    model = load_model(path_keras_model)

    with h5py.File(path_image, 'r') as f:
        # stream the data in to predict on it, safe region for all is the last 40%, except for the latest sep models
        items = len(f['Image'])
        test_images = f['Image'][-int(items*.4):-1]
        labels = f['Energy'][-int(items*.4):-1]
        print(labels.shape)

        predictions = model.predict(test_images, batch_size=64)
        print(predictions.shape)
        predictions = predictions.reshape(-1,)
    # Now make the confusion matrix

    #Loss Score so can tell which one it is
    try:
        filename = path_keras_model.split("/")[-1]
        filename = filename.split("_")[0]
        fig1 = plt.figure()
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_title(filename + ' Reconstructed vs. True Energy (log color scale)')
        plot_regressor_confusion(predictions, labels, ax=ax)
        fig1.show()

        # Plot confusion
        fig2 = plt.figure()
        ax = fig2.add_subplot(1, 1, 1)
        ax.set_title(filename + 'Reconstructed vs. True Energy (linear color scale)')
        plot_regressor_confusion(predictions, labels, log_z=False, ax=ax)
        fig2.show()

        error = predictions - labels
        best_energy.append(path_keras_model)
        best_energy_auc.append(np.mean(error))

        with open(energy_pickle, "wb") as f:
            pickle.dump([best_energy, best_energy_auc], f)

    except Exception as e:
        print(e)
        print(path_keras_model)

    return NotImplemented

def calc_roc_radec(path_image, path_keras_model):
    '''
    This one calcs ra and dec, makes Skymap of images, outputs that, as well as Li and Ma significance
    :param path_image:
    :param path_keras_model:
    :return: Li and Ma Significance for the model
    '''
    return NotImplemented


import os

#calc_roc_sourceXY(path_mc_images, "/run/media/jacob/WDRed8Tb1/Models/Disp2/585.735_MC_holchTrueSource_b30_p_(5, 5)_drop_0.24_conv_4_pool_0_denseN_216_numDense_4_convN_13_opt_same.h5")
#calc_roc_azzd(path_diffuse_images, "/run/media/jacob/WDRed8Tb1/Models/Disp/0.036_MC_ZdAz_b54_p_(5, 5)_drop_0.571_conv_9_pool_1_denseN_349_numDense_2_convN_10_opt_adam.h5")
#calc_roc_thetaphi(path_diffuse_images, "/run/media/jacob/WDRed8Tb1/Models/Disp/0.003_MC_ThetaPhiCustomError_b46_p_(2, 2)_drop_0.956_conv_5_pool_1_denseN_372_numDense_0_convN_241_opt_adam.h5")
#calc_roc_gammaHad(path_mc_images, path_proton_images, "/run/media/jacob/WDRed8Tb1/Models/Sep/0.172_MC_SepAll_b20_p_(3, 3)_drop_0.0_numDense_2_conv_5_pool_1_denseN_112_convN_37.h5")

sep_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(sep_models)
             for file in fileName if '.h5' in file]

energy_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(energy_models)
              for file in fileName if '.h5' in file]

source_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(disp_models)
             for file in fileName if '.h5' in file]

for path in sep_paths:
    calc_roc_gammaHad(path_mc_images, path_proton_images, path)

for path in energy_paths:
    calc_roc_energy(path_mc_images, path)
