#import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

    proton_train_preprocessor = EventFilePreprocessor(config=proton_configuration)
    gamma_train_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    gamma_configuration['paths'] = gamma_indexes[1][0]
    proton_configuration['paths'] = proton_indexes[1][0]

    proton_validate_preprocessor = EventFilePreprocessor(config=proton_configuration)
    gamma_validate_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    energy_gen_config = {
        'seed': 1337,
        'batch_size': 32,
        'start_slice': 0,
        'number_slices': shape[1] - shape[0],
        'mode': 'train',
        'chunked': False,
        'augment': True,
        'from_directory': True,
        'input_shape': [-1, gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2],
                        gamma_train_preprocessor.shape[1], 1],
        'as_channels': False,
    }

    energy_train = EventFileGenerator(paths=gamma_indexes[0][0], batch_size=3000,
                                      preprocessor=gamma_train_preprocessor,
                                      proton_paths=proton_indexes[0][0],
                                      proton_preprocessor=proton_train_preprocessor,
                                      as_channels=False,
                                      final_slices=10,
                                      slices=(30, 70),
                                      augment=True,
                                      training_type='Separation')

    energy_validate = EventFileGenerator(paths=gamma_indexes[1][0], batch_size=800,
                                         preprocessor=gamma_validate_preprocessor,
                                         proton_paths=proton_indexes[1][0],
                                         proton_preprocessor=proton_train_preprocessor,
                                         as_channels=False,
                                         final_slices=10,
                                         slices=(30, 70),
                                         augment=True,
                                         training_type='Separation')

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

    separation_model = Sequential()

    separation_model.add(ConvLSTM2D(32, kernel_size={{choice([1,2,3,4,5])}}, strides=1,
                                    padding='same',
                                    input_shape=[10,75,75,1],
                                    activation={{choice(['relu', 'tanh'])}},
                                    dropout={{uniform(0,0.5)}}, recurrent_dropout={{uniform(0,0.5)}},
                                    recurrent_activation={{choice(['relu', 'tanh', 'hard_sigmoid'])}},
                                    return_sequences=False,
                                    stateful=False))
    separation_model.add(MaxPooling2D())
    separation_model.add(Conv2D(64, kernel_size={{choice([1,3,5])}}, strides=1,
                                padding='same', activation={{choice(['relu', 'elu', 'sigmoid'])}}))
    separation_model.add(Dropout({{uniform(0, 0.75)}}))
    separation_model.add(Flatten())
    separation_model.add(Dense(64))
    separation_model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    separation_model.add(Dropout({{uniform(0, 0.75)}}))
    separation_model.add(Dense(32))
    separation_model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    separation_model.add(Dropout({{uniform(0, 0.75)}}))


    separation_model.add(Dense(2, activation='softmax'))
    separation_model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss='categorical_crossentropy',
                             metrics=['acc'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                               patience=10,
                                               verbose=0, mode='auto')

    result = separation_model.fit(x_train, y_train,
                                  batch_size=8,
                                  epochs=100,
                                  verbose=2,
                                  validation_split=0.2,
                                  callbacks=[early_stop])
    # get the highest validation accuracy of the training epochs
    validation_acc = np.amin(result.history['val_loss'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': validation_acc, 'status': STATUS_OK, 'model': separation_model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    best_model.summary()
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
