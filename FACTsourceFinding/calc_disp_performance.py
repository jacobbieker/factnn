import os
# to force on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
import h5py
from fact.io import read_h5py
import yaml
import tensorflow as tf
import os
import keras
import numpy as np
from numpy import savetxt, loadtxt, round, zeros, sin, cos, arctan2, clip, pi, tanh, exp, arange, dot, outer, array, shape, zeros_like, reshape, mean, median, max, min
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera
from sklearn.metrics import roc_auc_score
from sklearn import metrics

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
path_proton_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Proton_BothSource_Images.h5"
path_crab_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_diffuse_BothSource_Images.h5"
path_mrk501_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_mrk501_preprocessed_images.h5"

def calc_roc_gammaHad(path_image, path_keras_model):
    '''
    Returns ROC for gamma/hadron separation, also builds migration matrix for it and the plots of confidence
    :param path_image:
    :param path_keras_model:
    :return:
    '''

    # TODO Change this to work with my simulated ones, this uses both hadron and gamma truths

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
    return NotImplemented

def calc_roc_phitheta(path_image, path_keras_model, threshold):
    '''
    Returns Number of events in the threshold as ROC for Phi and Theta, as well as building migration matrix for them
    :param path_image:
    :param path_keras_model:
    :return:
    '''
    return NotImplemented

def calc_roc_azzd(path_image, path_keras_model, threshold):
    '''
    Returns the Number of events in the threshold for the Az, Zd estimation for simulated events, for observation this means that it is in comparsion to current classifier
    :param path_image:
    :param path_keras_model:
    :return:
    '''
    return NotImplemented

def calc_roc_energy(path_image, path_keras_model, threshold):
    '''
    Returns the ROC for the energy estimation
    :param path_image:
    :param path_keras_model:
    :return:
    '''
    return NotImplemented

def calc_roc_radec(path_image, path_keras_model):
    '''
    This one calcs ra and dec, makes Skymap of images, outputs that, as well as Li and Ma significance
    :param path_image:
    :param path_keras_model:
    :return: Li and Ma Significance for the model
    '''
    return NotImplemented

