#import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os.path

from factnn.data.preprocess.eventfile_preprocessor import EventFilePreprocessor
from factnn.generator.keras.eventfile_generator import EventFileGenerator
from factnn.utils import kfold

import GPy, GPyOpt
from keras.layers import Flatten, ConvLSTM2D, MaxPooling2D, Dense, Activation, Dropout, Conv2D, MaxPooling3D, BatchNormalization, AveragePooling2D, AveragePooling3D
from keras.models import Sequential
import keras
import numpy as np
import keras.backend as K
import tensorflow as tf

base_dir = "/home/jacob/Documents/cleaned_event_files_test/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "gamma/"]
proton_dir = [base_dir + "proton/"]

# Get paths from the directories
gamma_paths = []
for directory in gamma_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            gamma_paths.append(os.path.join(root, file))

# Get paths from the directories
crab_paths = []
for directory in proton_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            crab_paths.append(os.path.join(root, file))

# Now do the Kfold Cross validation Part for both sets of paths
#gamma_paths = gamma_paths[0:100]
#crab_paths = crab_paths[0:100]

gamma_indexes = kfold.split_data(gamma_paths, kfolds=5)
proton_indexes = kfold.split_data(crab_paths, kfolds=5)


def data(start_slice, end_slice, final_slices, rebin_size, gamma_train, proton_train, batch_size=8, as_channels=True, kfold_index=0):
    shape = [start_slice, end_slice]

    gamma_configuration = {
        'rebin_size': rebin_size,
        'output_file': "../gamma.hdf5",
        'shape': shape,
        'paths': gamma_train[0][kfold_index],
        'as_channels': as_channels
    }

    proton_configuration = {
        'rebin_size': rebin_size,
        'output_file': "../proton.hdf5",
        'shape': shape,
        'paths': proton_train[0][kfold_index],
        'as_channels': as_channels
    }

    proton_train_preprocessor = EventFilePreprocessor(config=proton_configuration)
    gamma_train_preprocessor = EventFilePreprocessor(config=gamma_configuration)
    print(gamma_train_preprocessor.shape)

    gamma_configuration['paths'] = gamma_train[1][kfold_index]
    proton_configuration['paths'] = proton_train[1][kfold_index]

    gamma_validate_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    gamma_configuration['paths'] = gamma_train[1][kfold_index]
    proton_configuration['paths'] = proton_train[1][kfold_index]

    gamma_test_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    energy_train = EventFileGenerator(paths=gamma_train[0][kfold_index], batch_size=batch_size,
                                      preprocessor=gamma_train_preprocessor,
                                      as_channels=as_channels,
                                      final_slices=final_slices,
                                      slices=(start_slice, end_slice),
                                      augment=True,
                                      normalize=False,
                                      training_type='Energy')

    energy_validate = EventFileGenerator(paths=gamma_train[1][kfold_index], batch_size=batch_size,
                                         preprocessor=gamma_validate_preprocessor,
                                         as_channels=as_channels,
                                         final_slices=final_slices,
                                         slices=(start_slice, end_slice),
                                         augment=False,
                                         normalize=False,
                                         training_type='Energy')

    energy_test = EventFileGenerator(paths=gamma_train[2][kfold_index], batch_size=batch_size,
                                     preprocessor=gamma_test_preprocessor,
                                     as_channels=as_channels,
                                     final_slices=final_slices,
                                     slices=(start_slice, end_slice),
                                     augment=False,
                                     normalize=False,
                                     training_type='Energy')

    if as_channels:
        final_shape = (gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], final_slices)
    else:
        final_shape = (final_slices, gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], 1)

    return energy_train, energy_validate, energy_test, final_shape


def create_model(shape):
    separation_model = Sequential()

    separation_model.add(Conv2D(64, kernel_size=3, strides=1,
                                padding='same',
                                input_shape=shape))
    #separation_model.add(BatchNormalization())
    separation_model.add(Activation('relu'))
    separation_model.add(Dropout(0.1))
    separation_model.add(MaxPooling2D())
    separation_model.add(Conv2D(64, kernel_size=3, strides=1,
                                padding='same',
                                input_shape=shape))
    #separation_model.add(BatchNormalization())
    separation_model.add(Activation('relu'))
    separation_model.add(Dropout(0.1))
    separation_model.add(MaxPooling2D())

    separation_model.add(Conv2D(128, kernel_size=3, strides=1,
                                padding='same',
                                input_shape=shape))
    #separation_model.add(BatchNormalization())
    separation_model.add(Activation('relu'))
    separation_model.add(Dropout(0.1))
    separation_model.add(MaxPooling2D())
    separation_model.add(Conv2D(128, kernel_size=3, strides=1,
                                padding='same',
                                input_shape=shape))
    #separation_model.add(BatchNormalization())
    separation_model.add(Activation('relu'))
    separation_model.add(Dropout(0.1))
    separation_model.add(MaxPooling2D())

    separation_model.add(Flatten())
    separation_model.add(Dense(128))
    #separation_model.add(BatchNormalization())
    separation_model.add(Activation('relu'))
    separation_model.add(Dropout(0.1))
    separation_model.add(Dense(256))
    #separation_model.add(BatchNormalization())
    separation_model.add(Activation('relu'))
    separation_model.add(Dropout(0.1))
    def r2(y_true, y_pred):
        from keras import backend as K
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return -1. * (1 - SS_res / (SS_tot + K.epsilon()))

    separation_model.add(Dense(1, activation='linear'))
    separation_model.compile(optimizer='adam', loss='mse',
                             metrics=['mae', r2])

    separation_model.summary()

    return separation_model


def fit_model(separation_model, train_gen, val_gen):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0002,
                                               patience=5,
                                               verbose=0, mode='auto',
                                               restore_best_weights=False)
    separation_model.fit_generator(
        generator=train_gen,
        epochs=200,
        verbose=1,
        validation_data=val_gen,
        callbacks=[early_stop],
        use_multiprocessing=True,
        workers=12,
        max_queue_size=50,
    )
    return separation_model


def model_evaluate(separation_model, test_gen):
    evaluation, test, r2 = separation_model.evaluate_generator(
        generator=test_gen,
        verbose=0,
        use_multiprocessing=True,
        workers=10,
        max_queue_size=50,
    )
    return r2


def run_mnist(start_slice=30, end_slice=80, final_slices=1, rebin_size=50,
              batch_size=64):
    evaluations = []

    for i in range(4):
        train_gen, val_gen, test_gen, shape = data(start_slice=start_slice, end_slice=end_slice, final_slices=final_slices,
                                                   rebin_size=rebin_size, gamma_train=gamma_indexes,
                                                   proton_train=proton_indexes, batch_size=batch_size, kfold_index=i)
        separation_model = create_model(shape=shape)
        separation_model = fit_model(separation_model, train_gen, val_gen)
        evaluation = model_evaluate(separation_model, test_gen)
        evaluations.append(evaluation)
        print(evaluation)
        K.clear_session()
        tf.reset_default_graph()
    print(np.mean(evaluations))
    print(np.std(evaluations))
    return evaluation

run_mnist()