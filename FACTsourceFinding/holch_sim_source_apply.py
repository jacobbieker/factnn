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

from fact.io import read_h5py
from fact_plots.skymap import plot_skymap
from fact.coordinates import camera_to_equatorial
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord

from fact.io import read_h5py

from fact_plots.plotting import add_preliminary
from fact_plots.time import read_timestamp

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



path_diffuse_images = "/run/media/jacob/SSD/Rebinned_5_MC_diffuse_BothSource_Images.h5"
path_mc_images = "/run/media/jacob/SSD/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_proton_images = "/run/media/jacob/SSD/Rebinned_5_MC_Proton_BothTracking_Images.h5"
path_crab_images = "/run/media/jacob/SSD/Rebinned_5_MC_diffuse_BothSource_Images.h5"
path_mrk501_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_Mrk501_5000_Images.h5"
path_crab_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_Crab1314_1_Images.h5"

plot_config = {
    'xlabel': r'$(\theta \,\, / \,\, {}^\circ )^2$',
    'preliminary_position': 'lower center',
    'preliminary_size': 'xx-large',
    'preliminary_color': 'lightgray',
}

columns = [
    'ra_prediction',
    'dec_prediction'
]


def plot_skymap(df, width=90.0, bins=10000, center_ra=None, center_dec=None, ax=None):
    '''
    Plot a 2d histogram of the reconstructed positions of air showers

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of the reconstructed events containing the columns
        `reconstructed_source_position_0`, `reconstructed_source_position_0`,
        `zd_tracking`, `az_tracking`, `time`, where time is the
        observation time as datetime
    width: float
        Extent of the plot in degrees
    bins: int
        number of bins
    center_ra: float
        right ascension of the center in degrees
    center_dec: float
        declination of the center in degrees
    ax: matplotlib.axes.Axes
        axes to plot into
    '''
    ax = ax or plt.gca()

    ra = df['ra_prediction']
    dec = df['dec_prediction']

    if center_ra is None:
        center_ra = ra.mean()
        center_ra *= 15  # conversion from hourangle to degree

    if center_dec is None:
        center_dec = dec.mean()

    bins, x_edges, y_deges, img = ax.hist2d(
        ra * 15,  # conversion from hourangle to degree
        dec,
        bins=bins,
        range=[
            [center_ra - width / 2, center_ra + width / 2],
            [center_dec - width / 2, center_dec + width / 2]
        ],
        )

    ax.set_xlabel('right ascencion / degree')
    ax.set_ylabel('declination / degree')
    ax.set_aspect(1)

    return ax, img


def main(data_path, threshold=0.0, key='events', bins=100, width=8.0, preliminary=True, config=None, output=None,
         source=None):
    '''
    Plot a 2d histogram of the origin of the air showers in the
    given hdf5 file in ra, dec.
    '''
    if config:
        with open(config) as f:
            plot_config.update(yaml.safe_load(f))

    if threshold > 0.0:
        columns.append('gamma_prediction')

    events = data_path#read_h5py(data_path, key='events', columns=columns)

    if threshold > 0.0:
        events = events.query('gamma_prediction >= @threshold').copy()

    fig, ax = plt.subplots(1, 1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    if source:
        coord = SkyCoord.from_name(source)
        center_ra = coord.ra.deg
        center_dec = coord.dec.deg
    else:
        center_ra = center_dec = None

    ax, img = plot_skymap(
        events,
        width=width,
        bins=bins,
        center_ra=center_ra,
        center_dec=center_dec,
        ax=ax,
    )

    if source:
        ax.plot(
            center_ra,
            center_dec,
            label=source,
            color='r',
            marker='o',
            linestyle='',
            markersize=10,
            markerfacecolor='none',
        )
        ax.legend()

    fig.colorbar(img, cax=cax, label='Gamma-Like Events')

    if preliminary:
        add_preliminary(
            plot_config['preliminary_position'],
            size=plot_config['preliminary_size'],
            color=plot_config['preliminary_color'],
            ax=ax,
            zorder=3,
        )

    fig.tight_layout(pad=0)
    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()

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

import datetime
def calc_roc_sourceXY(path_image, path_keras_model, path_y_model):
    '''
    Returns the Number of events in the threshold for the Az, Zd estimation for simulated events, for observation this means that it is in comparsion to current classifier
    :param path_image:
    :param path_keras_model:
    :return:
    '''

    with h5py.File(path_image, 'r') as f:
        # stream the data in to predict on it, safe region for all is the last 40%, except for the latest sep models
        items = len(f['Image'])
        images = f['Image'][0:-1]
        source_x = f['Source_X'][0:-1]
        source_y = f['Source_Y'][0:-1]
        az_point = f['Pointing_Az'][0:-1]
        zd_point = f['Pointing_Zd'][0:-1]
        obstime = f['Time'][0:-1]
        #obstime = np.asarray(converted_time)
        np.random.seed(0)
        rng_state = np.random.get_state()
        #np.random.shuffle(images)
        #np.random.set_state(rng_state)
        #np.random.shuffle(source_x)
        #np.random.set_state(rng_state)
        #np.random.shuffle(source_y)
        #np.random.set_state(rng_state)
        #np.random.shuffle(az_point)
        #np.random.set_state(rng_state)
        #np.random.shuffle(zd_point)
        #np.random.set_state(rng_state)
        #np.random.shuffle(obstime)
        obstime = pd.to_datetime(obstime, unit='us')
        ra, dec = camera_to_equatorial(
            x = source_x, y=source_y,
            zd_pointing=zd_point, az_pointing=az_point,
            obstime=obstime
        )
        images = images#[0:int(0.2*len(images))]
        source_x = source_x#[0:int(0.2*len(source_x))]
        source_y = source_y#[0:int(0.2*len(source_y))]

        #source_x += 180.975/2 # shifts everything to positive
        #source_y += 185.25/2 # shifts everything to positive
        #source_x = source_x / 4.94 # Ratio between the places
        #source_y = source_y / 4.826 # Ratio between y in original and y here
        # Since the mc AzZd ones are in radians, need these to be to, could convert back if need be later
        #labels = np.column_stack((source_x, source_y))
        #print(labels.shape)
        model_x = load_model(path_keras_model)
        predictions = model_x.predict(images, batch_size=64)
        print(predictions.shape)
        predictions_x = predictions.reshape(-1,)

        # Now convert to model_y things
        K.clear_session()
        tf.reset_default_graph()
        model_y = load_model(path_y_model)

        transformed_images = []
        for image_one in images:
            #print(image_one.shape)
            #image_one = image_one/np.sum(image_one)
            #print(np.sum(image_one))
            transformed_images.append(np.rot90(image_one)) # Try rotation to see if helps
            #print(np.max(image_one))
        images = np.asarray(transformed_images)

        predictions = model_y.predict(images, batch_size=64)
        print(predictions.shape)
        predictions_y = predictions.reshape(-1,)
        #predictions[:,0] += 180.975/2 # shifts everything to positive
        #predictions[:,1] += 185.25/2 # shifts everything to positive
        #predictions[:,0] = predictions[:,0] / 4.94 # Ratio between the places
        #predictions[:,1] = predictions[:,1] / 4.826 # Ratio between y in original and y here
    # Now make the confusion matrix

    #Loss Score so can tell which one it is
    filename = path_keras_model.split("/")[-1]
    filename = filename.split("_")[0]

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(filename + ' Reconstructed vs. True X')
    plot_sourceX_Y_confusion(predictions_x, source_x, ax=ax)
    fig1.show()


    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(filename + ' Reconstructed X vs. Rec Y')
    plot_sourceX_Y_confusion(predictions_x, predictions_y, ax=ax)
    fig1.show()

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(filename + ' True X vs. True Y')
    plot_sourceX_Y_confusion(source_x, source_y, ax=ax)
    fig1.show()

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title(filename + ' Reconstructed vs. True Y')
    plot_sourceX_Y_confusion(predictions_y, source_y, log_xy=True, ax=ax)
    fig1.show()

    # Now plot on skymap comparison
    # Convert estimations to equatorial
    # Get actual time
    ra, dec = camera_to_equatorial(
        x = predictions_x, y=predictions_y,
        zd_pointing=zd_point, az_pointing=az_point,
        obstime=obstime
    )
    df = pd.DataFrame()
    df['ra_prediction'] = ra
    df['dec_prediction'] = dec
    main(df, source="MRK501")
    plot_skymap(df=df)

    df = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk421_precuts.hdf5", key='events', last=5000)

    print("Read in file")

    plot_skymap(df=df)

    return NotImplemented

calc_roc_sourceXY(path_mrk501_images, "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/X_MC_CombinedAllShort_b28_p_(3, 3)_drop_0.26_conv_5_pool_1_denseN_251_numDense_2_convN_60_opt_same.h5", "/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/1455.516Y_MC_CombinedAll_b18_p_(2, 2)_drop_0.63_conv_2_pool_0_denseN_169_numDense_0_convN_22_opt_same.h5")