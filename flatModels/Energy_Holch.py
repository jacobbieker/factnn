import os
# to force on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
from keras.layers import Dense, Dropout, Activation, Conv1D, ConvLSTM2D, ELU, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score

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
    az_error = K.mean(K.abs(tf.atan2(K.sin(y_true - y_pred), K.cos(y_true - y_pred))))
    #zd_error = K.mean(K.square(y_pred[:,0] - y_true[:,0]), axis=-1)
    return az_error

def euc_dist_keras(y_true, y_pred):
    taxicab_error_wrong = K.abs(y_true[:,0] - y_pred[:,1]) + K.abs(y_true[:,1] - y_pred[:,0])
    other_error = K.sqrt(K.sum(K.square(y_true[:,0] - y_pred[:,0]) + K.square(y_true[:,1] - y_pred[:,1])))
    return other_error
# Hyperparameters

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_regressor_confusion(performace_df, label, log_xy=True, log_z=True, ax=None):

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


batch_sizes = [16, 64, 256]
patch_sizes = [(2, 2), (3, 3), (5, 5), (4, 4)]
dropout_layers = [0.01, 0.6]
num_conv_layers = [0, 6]
num_dense_layers = [0, 6]
num_conv_neurons = [8,128]
num_dense_neuron = [8,256]
num_pooling_layers = [0, 2]
num_runs = 500
number_of_training = 529000*(0.6)
number_of_testing = 529000*(0.2)
number_validate = 529000*(0.2)
optimizers = ['same']
epoch = 500

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

path_mc_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5"
#path_mrk501 = "/run/media/jacob/WDRed8Tb1/dl2_theta/Mrk501_precuts.hdf5"

#mrk501 = read_h5py(path_mrk501, key="events", columns=["event_num", "night", "run_id", "source_x_prediction", "source_y_prediction"])
#mc_image = read_h5py_chunked(path_mc_images, key='events', columns=['Image', 'Event', 'Night', 'Run'])

with h5py.File(path_mc_images, 'r') as f2:
    length_items = len(f2['Image'])
    length_training = 0.6*length_items


def batchYielder(path_to_training_data, type_training, percent_training, num_events_per_epoch=1000, path_to_proton_data=None, time_slice=100, batch_size=64):
    with h5py.File(path_to_training_data, 'r') as f:
        items = list(f.items())[1][1].shape[0]
        items = int(items*percent_training)
        length_dataset = len(f['Image'])
        section = 0
        times_train_in_items = int(np.floor(items / num_events_per_epoch))
        image = f['Image']
        if type_training == 'Energy':
            energy = f['Energy']
            while True:
                # Now create the batches from labels and other things
                batch_num = 0
                section = section % times_train_in_items
                offset = int(section * num_events_per_epoch)
                while batch_size * (batch_num + 1) < items:
                    batch_images = image[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size),time_slice-35:time_slice,::]
                    #batch_images = batch_images[:,time_slice-35:time_slice,::]
                    batch_image_label = energy[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    batch_num += 1
                    yield (batch_images, batch_image_label)
                section += 1

# Need validation generator
def validationGenerator(validation_percentage, time_slice=100, batch_size=64, predicting=False):
    global predicting_labels
    with h5py.File(path_mc_images, 'r') as f:
            # Get some truth data for now, just use Crab images
            items = len(f["Image"])
            validation_test = validation_percentage * items
            num_batch_in_validate = int(validation_test / batch_size)
            section = 0
            images = f['Image']
            energy = f['Energy']
            while True:
                batch_num = 0
                section = section % num_batch_in_validate
                offset = int(section * num_batch_in_validate)
                while batch_size * (batch_num + 1) < items:
                    batch_images = images[int(length_training+ (offset + int((batch_num)*batch_size))):int(length_training + (offset + int((batch_num+1)*batch_size))),time_slice-35:time_slice,::]
                    # Now slice it to only take the first 40 frames of the trigger from Jan's analysis
                    #batch_images = batch_images[:,time_slice-35:time_slice,::]
                    labels = energy[int(length_training+ (offset + int((batch_num)*batch_size))):int(length_training + (offset + int((batch_num+1)*batch_size)))]
                    batch_image_label = labels
                    batch_num += 1
                    yield (batch_images, batch_image_label)
                section += 1

from sklearn.metrics import r2_score

time_slice = 40

def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, frac_per_epoch):
    #try:
    model_base = base_dir + "/Models/3DEnergy/"
    model_name = "MC_Seperation3D" + "_p_" + str(
        patch_size) + "_drop_" + str(dropout_layer) + "_numDense_" + str(num_dense) \
                 + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + \
                 "_denseN_" + str(dense_neuron) + "_convN_" + str(conv_neurons)
    if not os.path.isfile(model_base + model_name + ".csv"):
        csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
        reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001)
        model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "_" + model_name + ".h5",
                                                           monitor='val_loss',
                                                           verbose=0,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='auto', period=1)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                   patience=10 * frac_per_epoch,
                                                   verbose=0, mode='auto')

        # Make the model
        model = Sequential()

        # Base Conv layer
        model.add(ConvLSTM2D(32, kernel_size=3, strides=2,
                             padding='same',
                             input_shape=(35, 75, 75, 1), activation='relu', dropout=0.3, recurrent_dropout=0.4, recurrent_activation='hard_sigmoid'))
        #model.add(
        #    ConvLSTM2D(32, kernel_size=3, strides=2,
        #               padding='same', activation='relu', dropout=0.3, recurrent_dropout=0.4, recurrent_activation='hard_sigmoid'))

        #model.add(Dropout(dropout_layer))

        model.add(
            Conv2D(64, (3,3), strides=(1, 1),
                   padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(dropout_layer))
        model.add(
            Conv2D(128, (3,3), strides=(1, 1),
                   padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(dropout_layer))
        model.add(MaxPooling2D())
        model.add(Flatten())

        for i in range(1):
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(dropout_layer/2))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(dropout_layer/2))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout_layer/2))

        # Final Dense layer
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse',
                      metrics=['mae'])
        model.summary()
        # Makes it only use
        model.fit_generator(generator=batchYielder(path_to_training_data=path_mc_images, time_slice=time_slice,  type_training="Energy", batch_size=batch_size, percent_training=0.6),
                            steps_per_epoch=int(np.floor(0.2*length_items/batch_size))
                            , epochs=1600,
                            verbose=1, validation_data=validationGenerator(0.2, time_slice=time_slice, batch_size=batch_size), validation_steps=int(np.floor(0.2*length_items/batch_size)),
                            callbacks=[early_stop, reduceLR, model_checkpoint],
                            )
        predictions = model.predict_generator(validationGenerator(0.2, time_slice=time_slice, predicting=True, batch_size=batch_size), steps=int(np.floor(0.2*length_items/batch_size)))
        #test_pred = model.predict(test_dataset, batch_size=64)
        print(r2_score(predicting_labels, predictions))
        exit()
        #print(roc_auc_score(test_labels, test_pred))
        K.clear_session()
        tf.reset_default_graph()

    #except Exception as e:
    #    print(e)
    #    K.clear_session()
    #    tf.reset_default_graph()
    #    pass

#except Exception as e:
#    print(e)
#    K.clear_session()
#    tf.reset_default_graph()
#    pass


for i in range(num_runs):
    dropout_layer = np.round(np.random.uniform(0.0, 1.0), 2)
    batch_size = 32#np.random.randint(batch_sizes[0], batch_sizes[1])
    num_conv = np.random.randint(num_conv_layers[0], num_conv_layers[1])
    num_dense = np.random.randint(num_dense_layers[0], num_dense_layers[1])
    patch_size = patch_sizes[np.random.randint(0, 3)]
    num_pooling_layer = np.random.randint(num_pooling_layers[0], num_pooling_layers[1])
    dense_neuron = np.random.randint(num_dense_neuron[0], num_dense_neuron[1])
    conv_neurons = np.random.randint(num_conv_neurons[0], num_conv_neurons[1])
    optimizer = optimizers[np.random.randint(0,1)]
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer)
