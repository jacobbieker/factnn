import os
# to force on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

architecture = 'manjar'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'


#All files for calculating ROC and AUC on, for phi and theta for simulated, Az, Zd for all, Energy for simulated, Ra, Dec for observation
path_diffuse_images = base_dir + "/Rebinned_5_MC_diffuse_BothSource_Images.h5"
path_mc_images = base_dir + "/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_proton_images = base_dir + "/Rebinned_5_MC_Proton_BothTracking_Images.h5"

disp_models = "/run/media/jacob/WDRed8Tb1/Models/Disp/"
energy_models = base_dir + "/Models/RealFinalEnergy/"
sep_models = base_dir + "/Models/FinalSep/"

phiTheta_pickle = "phiTheta_auc.p"
sourceXY_pickle = "sourceXY_auc.p"
sep_pickle = "sep_all.p"
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

        error = predictions - labels
        best_energy.append(path_keras_model)
        best_energy_auc.append(np.mean(error))

        with open(energy_pickle, "wb") as f:
            pickle.dump([best_energy, best_energy_auc], f)

    except Exception as e:
        print(e)
        print(path_keras_model)

    return NotImplemented



import os

sep_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(sep_models)
             for file in fileName if '.h5' in file]

energy_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(energy_models)
                for file in fileName if '.h5' in file]

for path in sep_paths:
    try:
        calc_roc_gammaHad(path_mc_images, path_proton_images, path)
    except:
        pass

for path in energy_paths:
    calc_roc_energy(path_mc_images, path)
