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

def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))
# Hyperparameters

import keras.losses
keras.losses.rmse_360_2 = rmse_360_2
keras.losses.euc_dist_keras = euc_dist_keras

# load

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'


#All files for calculating ROC and AUC on, for phi and theta for simulated, Az, Zd for all, Energy for simulated, Ra, Dec for observation
path_diffuse_images = "/run/media/jacob/SSD/Rebinned_5_MC_diffuse_BothSource_Images.h5"
path_mc_images = "/run/media/jacob/SSD/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_proton_images = "/run/media/jacob/SSD/Rebinned_5_MC_Proton_BothTracking_Images.h5"
path_crab_images = "/run/media/jacob/SSD/Rebinned_5_MC_diffuse_BothSource_Images.h5"
path_mrk501_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_mrk501_preprocessed_images.h5"
path_crab_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_Crab1314_1_Images.h5"

disp_models = "/run/media/jacob/WDRed8Tb1/Models/FinalDisp/"
energy_models = "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/Energy/"
sep_models = "/run/media/jacob/WDRed8Tb2/Sep/"
xy_models = "/run/media/jacob/WDRed8Tb1/Models/FinalSourceXY/"

phiTheta_pickle = "phiTheta_auc.p"
sourceXY_pickle = "sourceXY_auc.p"
sep_pickle = "sep_talapas.p"
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

def plot_roc(performace_df, model, ax=None):

    ax = ax or plt.gca()

    ax.axvline(0, color='lightgray')
    ax.axvline(1, color='lightgray')
    ax.axhline(0, color='lightgray')
    ax.axhline(1, color='lightgray')

    roc_aucs = []

    mean_fpr, mean_tpr, _ = metrics.roc_curve(performace_df, model)

    ax.set_title('Area Under Curve: {:.4f}'.format(
        metrics.roc_auc_score(performace_df, model)
    ))

    ax.plot(mean_fpr, mean_tpr, label='ROC curve')
    ax.legend()
    ax.set_aspect(1)

    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.figure.tight_layout()

    return ax


def plot_sourceX_Y_confusion(performace_df, label, log_xy=True, log_z=True, ax=None):

    ax = ax or plt.gca()

    #label = performace_df.label.copy()
    prediction = performace_df.copy()

    if log_xy is False:
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
    print(limits)
    print("Max, min Label")
    print([min_label, max_label])

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
        label = label
        prediction = prediction
    else:
        label = label
        prediction = prediction

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
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle

def plot_probabilities(performace_df, model=None, ax=None, classnames=('Proton', 'Gamma')):

    ax = ax or plt.gca()

    bin_edges = np.linspace(0, 1, 100+2)
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

crabdf = read_h5py("/run/media/jacob/WDRed8Tb1/testing/open_mrk501_sample_analysis/build/proton_test.hdf5", key='events')
mrk501 = read_h5py("/run/media/jacob/WDRed8Tb1/testing/open_mrk501_sample_analysis/build/gamma_test.hdf5", key="events")

plt.clf()
fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
plot_probabilities(crabdf['gamma_prediction'].dropna().values, mrk501['gamma_prediction'], ax)
plt.show()
exit()

plt.clf()
fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
plot_probabilities(mrk501['gamma_prediction'].dropna().values, None, ax)
plt.show()



def calc_roc_gammaHad(path_image, proton_image, path_keras_model):
    '''
    Returns ROC for gamma/hadron separation, also builds migration matrix for it and the plots of confidence
    :param path_image:
    :param path_keras_model:
    :return:
    '''

    #model = load_model("/run/media/jacob/WDRed8Tb1/Models/FinalSep/0.503_MC_vggSepNoGen_b58_p_(5, 5)_drop_0.93_conv_3_pool_0_denseN_109_numDense_2_convN_98_opt_same.h5")
    model = load_model(path_keras_model)
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file="gamma_HadBest.png")

    with h5py.File(path_image, 'r') as f:
        with h5py.File(proton_image, 'r') as f2:
            # stream the data in to predict on it, safe region for all is the last 40%, except for the latest sep models
            items = len(f['Image'])
            items_proton = len(f2['Image'])
            test_images = f['Image'][-int(items*.5):]
            test_images_false = f2['Image'][-int(items_proton*.8):]
            # Since the mc AzZd ones are in radians, need these to be to, could convert back if need be later
            validating_dataset = test_images#np.concatenate([test_images, test_images_false], axis=0)
            #labels = np.array([True] * (len(test_images)))
            #labels = np.array([True] * (len(test_images)) + [False] * len(test_images_false))
            #validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
            #print(labels.shape)

            predictions = model.predict(test_images, batch_size=64)

            high_pred = crabdf.index[(crabdf["gamma_prediction"] > 0.95)]
            pred_high = predictions[:,1].reshape((-1,))
            print(high_pred)
            print(pred_high)
            temp_high = []
            for index, pred in enumerate(pred_high):
                if pred > 0.5:
                    temp_high.append(index)
            pred_high = temp_high
            print(high_pred)
            print(pred_high)

            high_vale = crabdf.iloc[high_pred,:]
            high_pred = crabdf.iloc[pred_high,:]

            high_vale.describe().to_csv("RF_mrk501.csv")
            high_pred.describe().to_csv("NN_mrk501.csv")


            # Compare the two
            high_pred.summary()



            predictions2 = model.predict(test_images_false, batch_size=64)
            print(predictions.shape)
            print(predictions)
            from sklearn.preprocessing import label_binarize
            from itertools import cycle
            n_classes = 1
            #predictions = label_binarize(predictions, [0,1]) #predictions.reshape(-1,1)
            #validation_labels = label_binarize(validation_labels, [0,1])
            #labels = labels.reshape(-1,1)
            predictions = predictions[:,1].reshape((-1,))
            prediction2 = predictions2[:,1].reshape((-1,))
            #crabdf = read_h5py("/run/media/jacob/WDRed8Tb1/testing/open_crab_sample_analysis/build/crab_dl3.hdf5", key='events', last=647000)
            #mrk501 = read_h5py("/run/media/jacob/WDRed8Tb1/testing/open_mrk501_sample_analysis/build/crab_dl3.hdf5", key="events", last=448000)
            plt.clf()
            fig1 = plt.figure()
            ax = fig1.add_subplot(1,1,1)
            plot_probabilities(prediction2, predictions, ax)
            plt.show()

            K.clear_session()
            tf.reset_default_graph()


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
        test_images = f['Image'][0:int(items*.2):]
        labels_az = f['Source_Zd'][0:int(items*.2):]
        labels_zd = f['Source_Az'][0:int(items*.2):]
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
        with h5py.File(path_proton_images, 'r') as f2:
            # Get some truth data for now, just use Crab images
            images = f['Image'][0:-1]
            #images2 = f2['Image'][0:2000]
            images_source_zd = f['Theta'][0:-1]
            #images_source_az = f2['Theta'][0:2000]
            #images_source_az = np.deg2rad(images_source_az)
            #images_source_zd = np.deg2rad(images_source_zd)
            # stream the data in to predict on it, safe region for all is the last 40%, except for the latest sep models
            np.random.seed(0)
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            #np.random.shuffle(images2)
            np.random.set_state(rng_state)
            #np.random.shuffle(images_source_az)
            np.random.set_state(rng_state)
            np.random.shuffle(images_source_zd)
            images = images[0:int(0.2*len(images))]
            #images2 = images2[0:int(0.02*len(images2))]
            #images_source_az = images_source_az[0:int(0.02*len(images_source_az))]
            images_source_zd = images_source_zd[0:int(0.2*len(images_source_zd))]
            validating_dataset = images # np.concatenate([images, images2], axis=0)
            #print(validating_dataset.shape)
            labels = images_source_zd #np.concatenate([images_source_zd, images_source_az], axis=0)
            y = validating_dataset
            print(labels.shape)

            transformed_images = []
            for image_one in validating_dataset:
                #print(image_one.shape)
                image_one = image_one/np.sum(image_one)
                #print(np.sum(image_one))
                transformed_images.append(image_one)
                #print(np.max(image_one))
            validating_dataset = np.asarray(transformed_images)

            predictions = model.predict(validating_dataset, batch_size=64)
            print(predictions.shape)
            predictions = predictions.reshape(-1,)
            #predictions = np.deg2rad(predictions)
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

from sklearn.metrics import r2_score
def calc_roc_sourceXY(path_image, path_keras_model):
    '''
    Returns the Number of events in the threshold for the Az, Zd estimation for simulated events, for observation this means that it is in comparsion to current classifier
    :param path_image:
    :param path_keras_model:
    :return:
    '''


    model = load_model(path_keras_model)
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file="sourceXY_model_OneOut.png")
    return 0
    with h5py.File(path_image, 'r') as f:
        # stream the data in to predict on it, safe region for all is the last 40%, except for the latest sep models
        items = len(f['Image'])
        images = f['Image'][0:-1]
        source_x = f['Source_X'][0:-1]
        source_y = f['Source_Y'][0:-1]
        np.random.seed(0)
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(source_x)
        np.random.set_state(rng_state)
        np.random.shuffle(source_y)
        images = images[-int(0.5*len(images)):]
        source_x = source_x[-int(0.5*len(source_x)):]
        source_y = source_y[-int(0.5*len(source_y)):]

        # Since the mc AzZd ones are in radians, need these to be to, could convert back if need be later
        labels = np.column_stack((source_x, source_y))
        print(labels.shape)
        #predictions = model.predict(images, batch_size=64)
        #print(predictions.shape)
        #predictions = predictions

    # Now make the confusion matrix
    predictions = model.predict(images, batch_size=64)
    #test_pred = model.predict(images_test_y, batch_size=64)
    #print(roc_auc_score(x_label, predictions))
    # print(roc_auc_score(sign_test, test_pred))
    predictions_x = predictions[0].reshape(-1,)
    predictions_y = predictions[1].reshape(-1,)
    test_pred_y = source_x.reshape(-1,)
    test_pred_x = source_y.reshape(-1,)
    score_y = r2_score(test_pred_y, predictions_y)
    score_x = r2_score(test_pred_x, predictions_x)

    #Loss Score so can tell which one it is

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(str(score_x) + ' Reconstructed Train X vs. True X')
    plot_sourceX_Y_confusion(predictions_x, test_pred_x, log_z=True, ax=ax)
    fig1.show()


    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(' Reconstructed Train X vs. Rec Train Y')
    plot_sourceX_Y_confusion(predictions_x, predictions_y, ax=ax)
    fig1.show()

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(' True Train X vs. True Train Y')
    plot_sourceX_Y_confusion(test_pred_x, test_pred_y, ax=ax)
    fig1.show()

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(str(score_y) + ' Reconstructed Train Y vs. True Y')
    plot_sourceX_Y_confusion(predictions_y, test_pred_y, log_xy=True, log_z=True, ax=ax)
    fig1.show()

    #exit(1)
    K.clear_session()
    tf.reset_default_graph()

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
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file="Energy_Best_Holch.png")
    return 0

    with h5py.File("/run/media/jacob/SSD/Rebinned_5_MC_Gamma_BothSource_Images.h5", 'r') as f:
        # stream the data in to predict on it, safe region for all is the last 40%, except for the latest sep models
        items = len(f['Image'])
        test_images = f['Image'][-int(items*1.0):]
        labels = f['Energy'][-int(items*1.0):]

        # Get low, med, high
        mean_energy = np.mean(labels)
        std_dev = np.std(labels)

        lower_limit = 1000
        higher_limit = 10000

        smaller_labels = np.where(labels < lower_limit)[0]
        higher_labels = np.where(labels > higher_limit)[0]
        mid_labels = np.where(labels < higher_limit)[0]
        tmp_med = []
        for index in mid_labels:
            tmp_med.append(labels[index])
        tmp_med = np.asarray(tmp_med)
        mid_labels = np.where(tmp_med > lower_limit)[0]
        small_test_images = []
        large_test_images = []
        middle_test_images = []
        small_labels = []
        large_labels = []
        middle_labels = []
        for index in smaller_labels:
            small_test_images.append(test_images[index])
            small_labels.append(labels[index])
        for index in higher_labels:
            large_test_images.append(test_images[index])
            large_labels.append(labels[index])
        for index in mid_labels:
            middle_test_images.append(test_images[index])
            middle_labels.append(labels[index])

        small_test_images = np.asarray(small_test_images)
        middle_test_images = np.asarray(middle_test_images)
        large_test_images = np.asarray(large_test_images)

        print(small_test_images.shape)
        print(middle_test_images.shape)
        print(large_test_images.shape)

    predictions = model.predict(test_images, batch_size=64)
    print(predictions.shape)
    predictions = predictions.reshape(-1,)
    score = r2_score(labels, predictions)
    print("Total R2:")
    print(score)
    if score > 0.65:
        fig1 = plt.figure()
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_title(str(score) + ' Reconstructed vs. True Energy (log color scale)')
        plot_regressor_confusion(predictions, labels, ax=ax)
        fig1.show()

        # Plot confusion
        fig2 = plt.figure()
        ax = fig2.add_subplot(1, 1, 1)
        ax.set_title(str(score) + ' Reconstructed vs. True Energy (linear color scale)')
        plot_regressor_confusion(predictions, labels, log_z=False, ax=ax)
        fig2.show()
        predictions = model.predict(small_test_images, batch_size=64)
        print(predictions.shape)
        predictions = predictions.reshape(-1,)
        score = r2_score(small_labels, predictions)
        print("Lower R2:")
        print(score)
        fig1 = plt.figure()
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_title(str(score) + ' Reconstructed vs. True Energy (log color scale)')
        plot_regressor_confusion(predictions, small_labels, ax=ax)
        fig1.show()

        # Plot confusion
        fig2 = plt.figure()
        ax = fig2.add_subplot(1, 1, 1)
        ax.set_title(str(score) + ' Reconstructed vs. True Energy (linear color scale)')
        plot_regressor_confusion(predictions, small_labels, log_z=False, ax=ax)
        fig2.show()
        predictions = model.predict(middle_test_images, batch_size=64)
        print(predictions.shape)
        predictions = predictions.reshape(-1,)
        score = r2_score(middle_labels, predictions)
        print("Mid R2:")
        print(score)
        fig1 = plt.figure()
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_title(str(score) + ' Reconstructed vs. True Energy (log color scale)')
        plot_regressor_confusion(predictions, middle_labels, ax=ax)
        fig1.show()

        # Plot confusion
        fig2 = plt.figure()
        ax = fig2.add_subplot(1, 1, 1)
        ax.set_title(str(score) + ' Reconstructed vs. True Energy (linear color scale)')
        plot_regressor_confusion(predictions, middle_labels, log_z=False, ax=ax)
        fig2.show()
        predictions = model.predict(large_test_images, batch_size=64)
        print(predictions.shape)
        predictions = predictions.reshape(-1,)
        score = r2_score(large_labels, predictions)
        print("Upper R2:")
        print(score)
        fig1 = plt.figure()
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_title(str(score) + ' Reconstructed vs. True Energy (log color scale)')
        plot_regressor_confusion(predictions, large_labels, ax=ax)
        fig1.show()

        # Plot confusion
        fig2 = plt.figure()
        ax = fig2.add_subplot(1, 1, 1)
        ax.set_title(str(score) + ' Reconstructed vs. True Energy (linear color scale)')
        plot_regressor_confusion(predictions, large_labels, log_z=False, ax=ax)
        fig2.show()
    # Now make the confusion matrix

    #Loss Score so can tell which one it is
    K.clear_session()
    tf.reset_default_graph()
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

#calc_roc_energy(path_mc_images, "/run/media/jacob/WDRed8Tb1/Models/FinalEnergy/5210276.532_MC_energyNoGenDriver_b32_p_(2, 2)_drop_0.16_conv_4_pool_1_denseN_96_numDense_2_convN_17_opt_adam.h5")
#exit(1)
#calc_roc_thetaphi(path_mc_images, "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/0.000_MC_OnlyZdNoGenDriver_b36_p_(3, 3)_drop_0.944_conv_4_pool_1_denseN_169_numDense_3_convN_196_opt_%3Ckeras.optimizers.SGD object at 0x7f6319821668%3E.h5")
#calc_roc_thetaphi(path_diffuse_images, "/run/media/jacob/WDRed8Tb1/Models/FinalDisp/0.022_MC_OnlyThetaNoGen_b28_p_(3, 3)_drop_0.264_conv_8_pool_1_denseN_507_numDense_2_convN_60_opt_adam.h5")
#calc_roc_thetaphi(path_proton_images, "/run/media/jacob/WDRed8Tb1/Models/FinalDisp/0.022_MC_OnlyThetaNoGen_b28_p_(3, 3)_drop_0.264_conv_8_pool_1_denseN_507_numDense_2_convN_60_opt_adam.h5")


#calc_roc_sourceXY(path_mc_images, "/run/media/jacob/WDRed8Tb1/Models/FinalSourceXY/test/test/25.999_MC_holchSsourceXYFinalTrainZeroedOne_b28_p_(3, 3)_drop_0.26_conv_5_pool_1_denseN_251_numDense_2_convN_60_opt_same.h5")

#calc_roc_azzd(path_diffuse_images, "/run/media/jacob/WDRed8Tb1/Models/Disp/0.036_MC_ZdAz_b54_p_(5, 5)_drop_0.571_conv_9_pool_1_denseN_349_numDense_2_convN_10_opt_adam.h5")
#calc_roc_thetaphi(path_diffuse_images, "/run/media/jacob/WDRed8Tb1/Models/Disp/0.003_MC_ThetaPhiCustomError_b46_p_(2, 2)_drop_0.956_conv_5_pool_1_denseN_372_numDense_0_convN_241_opt_adam.h5")
#calc_roc_gammaHad(path_mc_images, path_proton_images, "/run/media/jacob/WDRed8Tb1/Models/Sep/0.172_MC_SepAll_b20_p_(3, 3)_drop_0.0_numDense_2_conv_5_pool_1_denseN_112_convN_37.h5")
sep_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk("/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/Sep/")
             for file in fileName if '.h5' in file]

energy_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk("/run/media/jacob/WDRed8Tb2/RealFinalEnergy/")
              for file in fileName if '.h5' in file]

source_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk("/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/finalSource/")
             for file in fileName if '.h5' in file]

disp_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk("/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/testing/")
                for file in fileName if '.h5' in file]

#calc_roc_energy(path_mc_images, "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/Energy/Y_ENERGYCONVMC_OneOutputPoolENERGYCONV_drop_0.27.h5")
#calc_roc_sourceXY(path_crab_images, "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/finalSource/Y_SeparateOutputs SOURCEMC_OneOutputPoolSOURCEXYSTDDEV_drop_0.09.h5")

#exit()

#calc_roc_gammaHad("/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_gamma_SOURCEXYALLSTDDEV_Images.h5", "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Proton_STDDEV_Images.h5", "/run/media/jacob/WDRed8Tb1/Models/FinalSep/3.459_MC_vggSepNoGen_b16_p_(3, 3)_drop_0.55_conv_3_pool_1_denseN_29_numDense_3_convN_122_opt_same.h5")
#exit()
#calc_roc_gammaHad("/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_gamma_SOURCEXYALLSTDDEV_Images.h5", "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Proton_STDDEV_Images.h5", "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/CrabOnes/SOURCE_MC_CrabSTDDEV_b64_p_(5, 5)_drop_0.99_numDense_3_conv_3_pool_1_denseN_69_convN_62.h5")

#exit()
#calc_roc_gammaHad("/run/media/jacob/WDRed8Tb1/Rebinned_5_Mrk501_HALFMILSTDDEV_Images.h5", "/run/media/jacob/WDRed8Tb1/Rebinned_5_Mrk501_HALFMILSTDDEV_Images.h5", "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/SOURCE_MC_SepSTDDEV_b143_p_(5, 5)_drop_0.22_numDense_3_conv_3_pool_1_denseN_219_convN_109.h5")
#calc_roc_gammaHad("/run/media/jacob/WDRed8Tb2/Rebinned_5_Crab1314_STDDEV_Images.h5", "/run/media/jacob/WDRed8Tb2/Rebinned_5_Crab1314_STDDEV_Images.h5", "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/SOURCE_MC_SepSTDDEV_b143_p_(5, 5)_drop_0.22_numDense_3_conv_3_pool_1_denseN_219_convN_109.h5")
calc_roc_gammaHad("/run/media/jacob/WDRed8Tb1/Rebinned_5_Mrk501_HALFMILSTDDEV_Images.h5", "/run/media/jacob/WDRed8Tb1/Rebinned_5_Mrk501_HALFMILSTDDEV_Images.h5", "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/CrabOnes/SOURCE_MC_CrabSTDDEV_b64_p_(5, 5)_drop_0.99_numDense_3_conv_3_pool_1_denseN_69_convN_62.h5")
calc_roc_gammaHad("/run/media/jacob/WDRed8Tb2/Rebinned_5_Crab1314_STDDEV_Images.h5", "/run/media/jacob/WDRed8Tb2/Rebinned_5_Crab1314_STDDEV_Images.h5", "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/CrabOnes/SOURCE_MC_CrabSTDDEV_b64_p_(5, 5)_drop_0.99_numDense_3_conv_3_pool_1_denseN_69_convN_62.h5")

calc_roc_energy(path_mc_images, "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/Energy/5210276.532_MC_energyNoGenDriver_b32_p_(2, 2)_drop_0.16_conv_4_pool_1_denseN_96_numDense_2_convN_17_opt_adam.h5")
calc_roc_sourceXY(path_crab_images, "/run/media/jacob/WDRed8Tb1/Models/FinalSourceXY/test/test/156.727_MC_holchSsourceXYFinal_b18_p_(2, 2)_drop_0.63_conv_2_pool_0_denseN_169_numDense_0_convN_22_opt_same.h5")

exit()

sep_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk("/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/FinalStuff/Sep/")
             for file in fileName if '.h5' in file]


for path in energy_paths:
    print(path)
    try:
        calc_roc_energy(path_mc_images, path)
        K.clear_session()
        tf.reset_default_graph()
    except Exception as e:
        print(e)
        K.clear_session()
        tf.reset_default_graph()
        pass

exit()

for path in source_paths:
    print(path)
    try:
        calc_roc_sourceXY("/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_gamma_SOURCEXYREALALLSTDDEV_Images.h5", path)
    except:
        pass

exit()
for path in sep_paths:
    print(path)
    #if path not in best_auc_sep:
    try:
        calc_roc_gammaHad("/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_gamma_SOURCEXYALLSTDDEV_Images.h5", "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Proton_STDDEV_Images.h5", path)
    except Exception as e:
        print(e)
        pass

for path in sep_paths:
    #print(path)
    #if path not in best_auc_sep:
    try:
        calc_roc_gammaHad(path_mc_images, path_proton_images, path)
    except Exception as e:
        print(e)
        pass
exit()
exit()
#for path in disp_paths:
#    try:
#        calc_roc_thetaphi(path_mc_images, path)
#    except:
#        pass
for path in source_paths:
    print(path)
    try:
        calc_roc_sourceXY(path_mc_images, path)
    except:
        pass
#
#exit(1)

if os.path.isfile(sep_pickle):
    with open(sep_pickle, "rb") as f:
        tmp = pickle.load(f)
        best_auc_sep = tmp[0]
        print(best_auc_sep)
        best_auc_sep_auc = tmp[0]

#
#

calc_roc_sourceXY(path_crab_images, "/run/media/jacob/WDRed8Tb1/Models/FinalSourceXY/test/test/156.727_MC_holchSsourceXYFinal_b18_p_(2, 2)_drop_0.63_conv_2_pool_0_denseN_169_numDense_0_convN_22_opt_same.h5")



for path in disp_paths:
    print(path)
    try:
        calc_roc_thetaphi(path_mc_images, path)
    except Exception as e:
        print(e)
        pass

