# import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os.path

from factnn.data.preprocess.eventfile_preprocessor import EventFilePreprocessor
from factnn.generator.keras.eventfile_generator import EventFileGenerator
from factnn.utils import kfold

import GPy, GPyOpt
from keras.layers import Flatten, ConvLSTM2D, MaxPooling2D, Dense, Activation, Dropout
from keras.models import Sequential
import keras
import numpy as np

def data():
    base_dir = "/home/jacob/Development/event_files/"
    obs_dir = [base_dir + "public/"]
    gamma_dir = [base_dir + "diffuse_gamma/"]
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

    energy_train = EventFileGenerator(paths=gamma_indexes[0][0], batch_size=2000,
                                      preprocessor=gamma_train_preprocessor,
                                      as_channels=False,
                                      final_slices=10,
                                      slices=(30, 70),
                                      augment=False,
                                      training_type='Disp')

    energy_validate = EventFileGenerator(paths=gamma_indexes[1][0], batch_size=800,
                                         preprocessor=gamma_validate_preprocessor,
                                         as_channels=False,
                                         final_slices=10,
                                         slices=(30, 70),
                                         augment=False,
                                         training_type='Disp')

    x_train, y_train = energy_train.__getitem__(0)
    x_test, y_test = energy_validate.__getitem__(0)

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = data()


def create_model(shape=(10, 75, 75, 1), neuron_1=64, kernel_1=3, strides_1=1, act_1=3, drop_1=0.3, rec_drop_1=0.3,
                 rec_act_1=2, dense_neuron_1=64, dense_neuron_2=128, optimizer=0,
                 pool=True, dense_act_1=0, dense_act_2=0, dense_drop_1=0.5, dense_drop_2=0.5):
    def r2(y_true, y_pred):
        from keras import backend as K
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return -1. * (1 - SS_res / (SS_tot + K.epsilon()))

    # Convert nums to strings
    activations = ['relu', 'sigmoid', 'hard_sigmoid', 'tanh']
    optimzers = ['adam', 'rmsprop', 'sgd']

    separation_model = Sequential()

    # separation_model.add(BatchNormalization())
    separation_model.add(ConvLSTM2D(neuron_1, kernel_size=kernel_1, strides=strides_1,
                                    padding='same',
                                    input_shape=shape,
                                    activation=activations[act_1],
                                    dropout=drop_1, recurrent_dropout=rec_drop_1,
                                    recurrent_activation=activations[rec_act_1],
                                    return_sequences=False,
                                    stateful=False))
    # separation_model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    if pool:
        separation_model.add(MaxPooling2D())
    # separation_model.add(Dropout({{uniform(0, 1)}}))
    separation_model.add(Flatten())
    separation_model.add(Dense(dense_neuron_1))
    separation_model.add(Activation(activations[dense_act_1]))
    separation_model.add(Dropout(dense_drop_1))
    separation_model.add(Dense(dense_neuron_2))
    separation_model.add(Activation(activations[dense_act_2]))
    separation_model.add(Dropout(dense_drop_2))

    # For energy

    separation_model.add(Dense(1, activation='linear'))
    separation_model.compile(optimizer=optimzers[optimizer], loss='mse',
                             metrics=['mae', r2])
    return separation_model


def fit_model(separation_model, x_train, y_train):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                               patience=5,
                                               verbose=0, mode='auto',
                                               restore_best_weights=True)
    model_checkpoint = keras.callbacks.ModelCheckpoint("models/hyperas_thesis_disp_{val_loss:0.4}.hdf5",
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto', period=1)
    nan_term = keras.callbacks.TerminateOnNaN()
    separation_model.fit(x_train, y_train,
                                  batch_size=8,
                                  epochs=100,
                                  verbose=2,
                                  validation_split=0.2,
                                  callbacks=[early_stop, model_checkpoint, nan_term])
    return separation_model


def model_evaluate(separation_model, x_test, y_test):
    evaluation = separation_model.evaluate(x_test, y_test, batch_size=8, verbose=2)
    return evaluation

# function to run mnist class

def run_mnist(x_train, y_train, x_test, y_test, shape=(10, 75, 75, 1), neuron_1=64, kernel_1=3, strides_1=1, act_1=3, drop_1=0.3, rec_drop_1=0.3,
              rec_act_1=2, dense_neuron_1=64, dense_neuron_2=128, optimizer=0,
              pool=True, dense_act_1=0, dense_act_2=0, dense_drop_1=0.5, dense_drop_2=0.5):
    separation_model = create_model(shape=shape, neuron_1=neuron_1, kernel_1=kernel_1, strides_1=strides_1, act_1=act_1, drop_1=drop_1, rec_drop_1=rec_drop_1,
                                    rec_act_1=rec_act_1, dense_neuron_1=dense_neuron_1, dense_neuron_2=dense_neuron_2, optimizer=optimizer,
                                    pool=pool, dense_act_1=dense_act_1, dense_act_2=dense_act_2, dense_drop_1=dense_drop_1, dense_drop_2=dense_drop_2)
    separation_model = fit_model(separation_model, x_train, y_train)
    evaluation = model_evaluate(separation_model, x_test, y_test)
    return evaluation


bounds = [{'name': 'drop_1', 'type': 'continuous', 'domain': (0.0, 0.75)},
          {'name': 'rec_drop_1', 'type': 'continuous', 'domain': (0.0, 0.75)},
          {'name': 'dense_drop_1', 'type': 'continuous', 'domain': (0.0, 0.75)},
          {'name': 'dense_drop_2', 'type': 'continuous', 'domain': (0.0, 0.75)},
          {'name': 'neuron_1', 'type': 'discrete', 'domain': (8, 16, 32, 64)},
          {'name': 'kernel_1', 'type': 'discrete', 'domain': (1, 2, 3, 4, 5)},
          {'name': 'strides_1', 'type': 'discrete', 'domain': (1, 2, 3)},
          {'name': 'dense_neuron_1', 'type': 'discrete', 'domain': (8, 16, 32, 64)},
          {'name': 'dense_neuron_2', 'type': 'discrete', 'domain': (8, 16, 32, 64, 128)},
          {'name': 'pool', 'type': 'discrete', 'domain': (True, False)},
          {'name': 'optimizer', 'type': 'discrete', 'domain': (0, 1, 2)},
          {'name': 'dense_act_1', 'type': 'discrete', 'domain': (0, 1)},
          {'name': 'dense_act_2', 'type': 'discrete', 'domain': (0, 1)},
          {'name': 'act_1', 'type': 'discrete', 'domain': (0, 1, 3)},
          {'name': 'rec_act_1', 'type': 'discrete', 'domain': (0, 2, 3)},
          ]


def f(x):
    print(x)
    evaluation = run_mnist(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        neuron_1=int(x[:, 4]),
        kernel_1=int(x[:, 5]),
        strides_1=int(x[:, 6]),
        act_1=int(x[:, 13]),
        drop_1=float(x[:, 0]),
        rec_drop_1=float(x[:, 1]),
        rec_act_1=int(x[:, 14]),
        dense_neuron_1=int(x[:, 7]),
        dense_neuron_2=int(x[:, 8]),
        optimizer=int(x[:, 10]),
        pool=bool(x[:, 9]),
        dense_act_1=int(x[:, 11]),
        dense_act_2=int(x[:, 12]),
        dense_drop_1=float(x[:, 2]),
        dense_drop_2=float(x[:, 3]))
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]


if __name__ == '__main__':
    opt_mnist = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)

    opt_mnist.run_optimization(max_iter=10)

    print("""
    Optimized Parameters:
    \t{0}:\t{1}
    \t{2}:\t{3}
    \t{4}:\t{5}
    \t{6}:\t{7}
    \t{8}:\t{9}
    \t{10}:\t{11}
    \t{12}:\t{13}
    """.format(bounds[0]["name"], opt_mnist.x_opt[0],
               bounds[1]["name"], opt_mnist.x_opt[1],
               bounds[2]["name"], opt_mnist.x_opt[2],
               bounds[3]["name"], opt_mnist.x_opt[3],
               bounds[4]["name"], opt_mnist.x_opt[4],
               bounds[5]["name"], opt_mnist.x_opt[5],
               bounds[6]["name"], opt_mnist.x_opt[6]))
