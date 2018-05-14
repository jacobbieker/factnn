import os
# to force on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pickle
from keras import backend as K
import h5py
from fact.io import read_h5py, read_h5py_chunked
import yaml
import tensorflow as tf
import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, ELU, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera
import pandas as pd

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

def atan2(x, y, epsilon=1.0e-12):
    x = tf.where(tf.equal(x, 0.0), x+epsilon, x)
    y = tf.where(tf.equal(y, 0.0), y+epsilon, y)
    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
    return angle

# y in radians and second column is az, first is zd, adds the errors together, seems to work?
def rmse_360_2(y_true, y_pred):
    az_error = tf.reduce_mean(K.abs(tf.atan2(K.sin(y_true[:,1] - y_pred[:,1]), K.cos(y_true[:,1] - y_pred[:,1]))))
    zd_error = K.mean(K.square(y_pred[:,0] - y_true[:,0]), axis=-1)
    return az_error + zd_error

def euc_dist_keras(y_true, y_pred):
    taxicab_error_wrong = K.abs(y_true[:,0] - y_pred[:,1]) + K.abs(y_true[:,1] - y_pred[:,0])
    other_error = K.sqrt(K.sum(K.square(y_true[:,0] - y_pred[:,0]) + K.square(y_true[:,1] - y_pred[:,1])))
    return other_error
# Hyperparameters

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
        ax.set_xlabel(r'$X$')
        ax.set_ylabel(r'$Y$')
    else:
        ax.set_xlabel(r'$X_{\mathrm{MC}}$')
        ax.set_ylabel(r'$Y_{\mathrm{Est}}$')

    return ax


batch_sizes = [16, 64, 256]
patch_sizes = [(2, 2), (3, 3), (5, 5), (4, 4)]
dropout_layers = [0.0, 1.0]
num_conv_layers = [0, 6]
num_dense_layers = [0, 6]
num_conv_neurons = [32,128]
num_dense_neuron = [32,256]
num_pooling_layers = [0, 2]
num_runs = 500
number_of_training = 529000*(0.6)
number_of_testing = 529000*(0.2)
number_validate = 529000*(0.2)
optimizers = ['same']
epoch = 500

path_mc_images = "/run/media/jacob/SSD/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_proton_images = "/run/media/jacob/SSD/Rebinned_5_MC_Proton_BothTracking_Images.h5"
np.random.seed(0)

def metaYielder():
    with h5py.File(path_mc_images, 'r') as f:
        gam = len(f['Image'])
        with h5py.File(path_proton_images, 'r') as f2:
            had = len(f2['Image'])
            sumEvt = gam + had
            print(sumEvt)

    gamma_anteil = gam / sumEvt
    hadron_anteil = had / sumEvt

    gamma_count = int(round(number_of_training * gamma_anteil))
    hadron_count = int(round(number_of_training * hadron_anteil))

    gamma_anteil = 0.5
    hadron_anteil = 0.5

    return gamma_anteil, hadron_anteil, gamma_count, hadron_count

desc = ""
with h5py.File(path_mc_images, 'r') as f:
    with h5py.File(path_proton_images, 'r') as f2:
        gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder()
        # Get some truth data for now, just use Crab images
        items = len(f2["Image"])
        images = f['Image'][0:1000]
        images_false = f2['Image'][0:1000]
        validating_dataset = np.concatenate([images, images_false], axis=0)
        #print(validating_dataset.shape)
        labels = np.array([True] * (len(images)) + [False] * len(images_false))
        np.random.seed(0)
        rng_state = np.random.get_state()
        np.random.shuffle(validating_dataset)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        validating_dataset = validating_dataset[0:int(0.8*len(validating_dataset))]
        labels = labels[0:int(0.8*len(labels))]
        #print(ind)
        #print(counts)
        #rng_state = np.random.get_state()
        #np.random.set_state(rng_state)
        #np.random.shuffle(validating_dataset)
        #np.random.set_state(rng_state)
        #np.random.shuffle(labels)
        #del images
        #del images_false
        validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
        y = validating_dataset
        x_label = validation_labels
        print("Finished getting data")


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer):
    try:
        model_base = ""# base_dir +"/" # + "/Models/FinalSourceXY/test/test/"
        model_name = "MC_SourceSep" + "_Numconv_" + str(num_conv) + "_NumDense_" + str(num_dense)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            #reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=70, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "X_" + desc + model_name + ".h5", monitor='val_loss', verbose=0,
                                                               save_best_only=True, save_weights_only=False, mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=18, verbose=0, mode='auto')

            # Make the model
            inp = keras.models.Input((75,75,1))
            # Block - conv
            x = Conv2D(conv_neurons, 8, 8, border_mode='same', subsample=[1,1], activation='relu', name='Conv1')(inp)
            # Block - conv
            x = Conv2D(2*conv_neurons, 5, 5, border_mode='same', subsample=[1,1], activation='relu', name='Conv2')(x)
            x = MaxPooling2D(padding='same')(x)
            # Block - conv
            x = Conv2D(4*conv_neurons, 5, 5, border_mode='same', subsample=[2,2], activation='relu', name='Conv3')(x)
            x = MaxPooling2D(padding='same')(x)
            x = Dropout(dropout_layer)(x)
            # Block - conv
            x = Conv2D(2*conv_neurons, 5, 5, border_mode='same', subsample=[2,2], activation='relu', name='Conv5')(x)
            # Block - conv
            x = Conv2D(4*conv_neurons, 5, 5, border_mode='same', subsample=[2,2], activation='relu', name='Conv6')(x)

            x = Conv2D(2*conv_neurons, 3, 3, border_mode='same', subsample=[2,2], activation='relu', name='Conv6')(x)
            # Block - flatten
            x = Flatten()(x)
            x = Dropout(dropout_layer)(x)
            # Block - fully connected
            for i in range(num_dense+1):
                x = Dense(dense_neuron, activation='elu')(x)
                x = Dropout(0.5)(x)
                x = ELU()(x)

            y_out = Dense(2, activation='softmax', name="y_out")(x)

            model = keras.models.Model(inp, y_out)
            # Block - output
            model.summary()
            adam = keras.optimizers.adam(lr=0.0001)
            model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
            #model.fit_generator(generator=batchYielder(), steps_per_epoch=np.floor(((number_of_training / batch_size))), epochs=epoch,
            #                    verbose=2, validation_data=(y, y_label), callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])
            model.fit(x=y, y=x_label, batch_size=batch_size, epochs=500, verbose=2, validation_split=0.2, callbacks=[early_stop, model_checkpoint, csv_logger])

            K.clear_session()
            tf.reset_default_graph()

    except Exception as e:
        print(e)
        K.clear_session()
        tf.reset_default_graph()
        pass


for i in range(num_runs):
    dropout_layer = np.round(np.random.uniform(0.0, 1.0), 2)
    batch_size = np.random.randint(batch_sizes[0], batch_sizes[1])
    num_conv = np.random.randint(num_conv_layers[0], num_conv_layers[1])
    num_dense = np.random.randint(num_dense_layers[0], num_dense_layers[1])
    patch_size = patch_sizes[np.random.randint(0, 3)]
    num_pooling_layer = np.random.randint(num_pooling_layers[0], num_pooling_layers[1])
    dense_neuron = np.random.randint(num_dense_neuron[0], num_dense_neuron[1])
    conv_neurons = np.random.randint(num_conv_neurons[0], num_conv_neurons[1])
    optimizer = optimizers[np.random.randint(0,1)]
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer)
