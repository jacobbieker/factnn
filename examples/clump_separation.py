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


def data(start_slice, end_slice, final_slices, rebin_size, gamma_train, proton_train, batch_size=8):
    shape = [start_slice, end_slice]

    gamma_configuration = {
        'rebin_size': rebin_size,
        'output_file': "../gamma.hdf5",
        'shape': shape,
        'paths': gamma_train[0][0],
        'as_channels': False
    }

    proton_configuration = {
        'rebin_size': rebin_size,
        'output_file': "../proton.hdf5",
        'shape': shape,
        'paths': proton_train[0][0],
        'as_channels': False
    }

    proton_train_preprocessor = EventFilePreprocessor(config=proton_configuration)
    gamma_train_preprocessor = EventFilePreprocessor(config=gamma_configuration)
    print(gamma_train_preprocessor.shape)

    gamma_configuration['paths'] = gamma_train[1][0]
    proton_configuration['paths'] = proton_train[1][0]

    proton_validate_preprocessor = EventFilePreprocessor(config=proton_configuration)
    gamma_validate_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    gamma_configuration['paths'] = gamma_train[1][0]
    proton_configuration['paths'] = proton_train[1][0]

    proton_test_preprocessor = EventFilePreprocessor(config=proton_configuration)
    gamma_test_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    energy_train = EventFileGenerator(paths=gamma_train[0][0], batch_size=batch_size,
                                      preprocessor=gamma_train_preprocessor,
                                      proton_paths=proton_train[0][0],
                                      proton_preprocessor=proton_train_preprocessor,
                                      as_channels=False,
                                      final_slices=final_slices,
                                      slices=(start_slice, end_slice),
                                      augment=True,
                                      normalize=True,
                                      training_type='Separation')

    energy_validate = EventFileGenerator(paths=gamma_train[1][0], batch_size=batch_size,
                                         proton_paths=proton_train[1][0],
                                         proton_preprocessor=proton_validate_preprocessor,
                                         preprocessor=gamma_validate_preprocessor,
                                         as_channels=False,
                                         final_slices=final_slices,
                                         slices=(start_slice, end_slice),
                                         augment=False,
                                         normalize=True,
                                         training_type='Separation')

    energy_test = EventFileGenerator(paths=gamma_train[2][0], batch_size=batch_size,
                                     proton_paths=proton_train[2][0],
                                     proton_preprocessor=proton_test_preprocessor,
                                     preprocessor=gamma_test_preprocessor,
                                     as_channels=False,
                                     final_slices=final_slices,
                                     slices=(start_slice, end_slice),
                                     augment=False,
                                     normalize=False,
                                     training_type='Separation')

    final_shape = (final_slices, gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], 1)

    return energy_train, energy_validate, energy_test, final_shape


def create_model(shape):
    separation_model = Sequential()

    separation_model.add(ConvLSTM2D(16, kernel_size=3, strides=1,
                                    padding='same',
                                    input_shape=shape,
                                    activation='tanh',
                                    dropout=0.1, recurrent_dropout=0.1,
                                    recurrent_activation='hard_sigmoid',
                                    return_sequences=True,
                                    stateful=False))
    separation_model.add(AveragePooling3D())
    separation_model.add(ConvLSTM2D(32, kernel_size=3, strides=1,
                                    padding='same',
                                    activation='tanh',
                                    dropout=0.1, recurrent_dropout=0.1,
                                    recurrent_activation='hard_sigmoid',
                                    return_sequences=True,
                                    stateful=False))
    separation_model.add(AveragePooling3D())
    separation_model.add(ConvLSTM2D(64, kernel_size=3, strides=1,
                                    padding='same',
                                    activation='tanh',
                                    dropout=0.1, recurrent_dropout=0.1,
                                    recurrent_activation='hard_sigmoid',
                                    return_sequences=False,
                                    stateful=False))
    separation_model.add(AveragePooling2D())
    separation_model.add(Conv2D(128, kernel_size=3, strides=1,
                                padding='same'))
    #separation_model.add(BatchNormalization())
    separation_model.add(Activation('relu'))
    separation_model.add(AveragePooling2D())
    separation_model.add(Dropout(0.2))

    separation_model.add(Flatten())
    separation_model.add(Dense(128))
    #separation_model.add(BatchNormalization())
    separation_model.add(Activation('relu'))
    separation_model.add(Dropout(0.3))
    separation_model.add(Dense(256))
    #separation_model.add(BatchNormalization())
    separation_model.add(Activation('relu'))
    separation_model.add(Dropout(0.3))

    separation_model.add(Dense(2, activation='softmax'))
    separation_model.compile(optimizer='adam', loss='categorical_crossentropy',
                             metrics=['acc'])

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
        workers=5,
        max_queue_size=50,
    )
    return separation_model


def model_evaluate(separation_model, test_gen):
    evaluation = separation_model.evaluate_generator(
        generator=test_gen,
        verbose=0,
        use_multiprocessing=True,
        workers=10,
        max_queue_size=50,
    )
    return evaluation


def run_mnist(start_slice=40, end_slice=60, final_slices=20, rebin_size=100,
              batch_size=8):
    train_gen, val_gen, test_gen, shape = data(start_slice=start_slice, end_slice=end_slice, final_slices=final_slices,
                                               rebin_size=rebin_size, gamma_train=gamma_indexes,
                                               proton_train=proton_indexes, batch_size=batch_size)
    separation_model = create_model(shape=shape)
    separation_model = fit_model(separation_model, train_gen, val_gen)
    evaluation = model_evaluate(separation_model, test_gen)
    print(evaluation)
    return evaluation


run_mnist()
