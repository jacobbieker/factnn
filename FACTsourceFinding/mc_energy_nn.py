#!/bin/env /projects/sventeklab/jbieker/virtualenv/thesis


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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, ELU, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from fact.coordinates.utils import horizontal_to_camera
import pickle
architecture = 'manjar'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

# Hyperparameters

batch_sizes = [16, 64, 256]
patch_sizes = [(2, 2), (3, 3), (5, 5), (4,4)]
dropout_layers = [0.0, 1.0]
num_conv_layers = [3, 6]
num_dense_layers = [2, 6]
num_conv_neurons = [8,128]
num_dense_neuron = [8,256]
num_pooling_layers = [1, 2]
num_runs = 500
number_of_training = 497000*(0.6)
number_of_testing = 497000*(0.2)
number_validate = int(497000*(0.2))
optimizer = 'adam'
epoch = 900


path_mc_images = base_dir + "/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_mc_proton = base_dir + "/Rebinned_5_MC_Proton_BothTracking_Images.h5"



def metaYielder():
    gamma_anteil = 1
    gamma_count = int(round(number_of_training*gamma_anteil))

    return gamma_anteil, gamma_count


with h5py.File(path_mc_images, 'r') as f:
    with h5py.File(path_mc_proton, 'r') as f2:
        gamma_anteil, gamma_count = metaYielder()
        # Get some truth data for now, just use Crab images
        images = f['Image'][0:-1]
        images_energy = f['Energy'][0:-1]
        images2 = f2['Image'][0:-1]
        images_energy2 = f2['Energy'][0:-1]
        images = images[0:int(0.5*len(images))]
        images_energy = images_energy[0:int(0.5*len(images_energy))]
        images2 = images2[0:int(0.5*len(images2))]
        images_energy2 = images_energy2[0:int(0.5*len(images_energy2))]
        validating_dataset = np.concatenate([images, images2], axis=0)
        #print(validating_dataset.shape)
        labels = np.concatenate([images_energy, images_energy2], axis=0)
        np.random.seed(0)
        rng_state = np.random.get_state()
        np.random.shuffle(validating_dataset)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        y = validating_dataset
        y_label = labels
        print(y_label.shape)
        print("Finished getting data")


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons):
    try:
        model_base = base_dir + "/Models/RealFinalEnergy/"
        model_name = "MC_energyNoGenDriver_b" + str(batch_size) +"_p_" + str(patch_size) + "_drop_" + str(dropout_layer) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + "_denseN_" + str(dense_neuron) + "_numDense_" + str(num_dense) + "_convN_" + \
                     str(conv_neurons) + "_opt_" + str(optimizer)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "{val_loss:.3f}_" + model_name + ".h5", monitor='val_loss', verbose=0,
                                                               save_best_only=True, save_weights_only=False, mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='auto')

            gamma_anteil, gamma_count = metaYielder()
            # Make the model
            model = Sequential()
            # Preprocess incoming data, centered around zero with small standard deviation
            #model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(75, 75, 1)))
            # Block - conv
            model.add(Conv2D(conv_neurons, 8, 8, border_mode='same', subsample=[4,4], activation='elu', name='Conv1', input_shape=(75,75,1)))
            # Block - conv
            model.add(Conv2D(2*conv_neurons, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv2'))
            # Block - conv
            model.add(Conv2D(4*conv_neurons, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv3'))
            # Block - flatten
            model.add(Flatten())
            model.add(Dropout(dropout_layer))
            model.add(ELU())
            # Block - fully connected
            model.add(Dense(dense_neuron, activation='elu', name='FC1'))
            model.add(Dropout(0.5))
            model.add(ELU())
            # Block - output
            model.add(Dense(1, name='output'))
            model.summary()
            adam = keras.optimizers.adam(lr=0.001)
            model.compile(optimizer=adam, loss='mse', metrics=['mae'])

            model.fit(x=y, y=y_label, batch_size=batch_size, epochs=epoch, validation_split=0.2, callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])

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
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons)
