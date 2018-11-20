import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from factnn import GammaPreprocessor, ProtonPreprocessor
from factnn.generator.keras.eventfile_generator import EventFileGenerator
from factnn.data.preprocess.eventfile_preprocessor import EventFilePreprocessor
import os.path
from factnn.utils import kfold
from keras.models import load_model


def data():
    base_dir = "/home/jacob/Development/event_files/"
    obs_dir = [base_dir + "public/"]
    gamma_dir = [base_dir + "gamma/"]
    proton_dir = [base_dir + "proton/"]

    shape = [30, 70]
    rebin_size = 5

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
    gamma_indexes = kfold.split_data(gamma_paths, kfolds=5)
    proton_indexes = kfold.split_data(crab_paths, kfolds=5)

    gamma_configuration = {
        'rebin_size': rebin_size,
        'output_file': "../gamma.hdf5",
        'shape': shape,
        'paths': gamma_indexes[0][0],
        'as_channels': False
    }

    proton_configuration = {
        'rebin_size': rebin_size,
        'output_file': "../proton.hdf5",
        'shape': shape,
        'paths': proton_indexes[0][0],
        'as_channels': False
    }

    gamma_train_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    gamma_configuration['paths'] = gamma_indexes[1][0]
    proton_configuration['paths'] = proton_indexes[1][0]

    gamma_validate_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    energy_train = EventFileGenerator(paths=gamma_indexes[0][0], batch_size=1000,
                                      preprocessor=gamma_train_preprocessor,
                                      as_channels=False,
                                      final_slices=5,
                                      slices=(30, 70),
                                      augment=True,
                                      training_type='Energy')

    energy_validate = EventFileGenerator(paths=gamma_indexes[1][0], batch_size=200,
                                         preprocessor=gamma_validate_preprocessor,
                                         as_channels=False,
                                         final_slices=5,
                                         slices=(30, 70),
                                         augment=True,
                                         training_type='Energy')

    x_train, y_train = energy_train.__getitem__(0)
    x_test, y_test = energy_validate.__getitem__(0)

    return x_train, y_train, x_test, y_test


from keras.layers import Dense, Dropout, Flatten, ConvLSTM2D, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, PReLU, \
    BatchNormalization, ReLU
from keras.models import Sequential
import keras
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, randint, uniform


def create_model(x_train, y_train, x_test, y_test):
    def r2(y_true, y_pred):
        from keras import backend as K
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return -1. * (1 - SS_res / (SS_tot + K.epsilon()))

    # Make the model
    model = Sequential()

    # Base Conv layer
    model.add(ConvLSTM2D(16, kernel_size={{choice([1, 2, 3, 4, 5])}}, strides=1,
                         padding='same',
                         input_shape=(5, 75, 75, 1),
                         activation={{choice(['relu', 'tanh'])}},
                         dropout={{uniform(0, 0.75)}},
                         recurrent_dropout={{uniform(0, 0.75)}},
                         recurrent_activation='hard_sigmoid',
                         return_sequences=True))
    model.add(ConvLSTM2D(32, kernel_size={{choice([1, 2, 3, 4, 5])}},
                         strides=1,
                         padding='same', activation={{choice(['relu', 'tanh'])}},
                         dropout={{uniform(0, 0.75)}},
                         recurrent_dropout={{uniform(0, 0.75)}},
                         recurrent_activation='hard_sigmoid',
                         return_sequences=False))
    model.add(MaxPooling2D())
    model.add(
        Conv2D(32, kernel_size={{choice([1, 2, 3, 4, 5])}}, strides=1,
               padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(
        Conv2D(32, kernel_size={{choice([1, 2, 3, 4, 5])}}, strides=1,
               padding='same', activation='relu'))
    # model.add(MaxPooling3D())
    model.add(MaxPooling2D())
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout({{uniform(0, 0.75)}}))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout({{uniform(0, 0.75)}}))

    # Final Dense layer
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss='mse',
                  metrics=['mae', r2])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0002,
                                               patience=5,
                                               verbose=0, mode='auto')
    model_checkpoint = keras.callbacks.ModelCheckpoint("models/hyperas_thesis_energy_{val_loss:0.2}.hdf5",
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto', period=1)
    nan_term = keras.callbacks.TerminateOnNaN()


    result = model.fit(x_train, y_train,
                       batch_size=8,
                       epochs=100,
                       verbose=2,
                       validation_split=0.2,
                       callbacks=[early_stop, model_checkpoint, nan_term])
    # get the highest validation accuracy of the training epochs
    validation_acc = np.nanmin(result.history['val_loss'])
    if np.isnan(validation_acc):
        validation_acc = 2**32-1
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    best_model.summary()
    best_model.save("models/hyperas_thesis_energy_best.hdf5")
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
