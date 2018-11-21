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

base_dir = "/home/jacob/Development/event_files/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "diffuse_gamma/"]
proton_dir = [base_dir + "proton/"]

# Get paths from the directories
gamma_paths = []
for directory in gamma_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            gamma_paths.append(os.path.join(root, file))

# Now do the Kfold Cross validation Part for both sets of paths
gamma_indexes = kfold.split_data(gamma_paths, kfolds=5)
gamma_train = kfold.split_data(gamma_indexes[0][0], kfolds=5)


def data(start_slice, end_slice, final_slices, rebin_size, gamma_train, gamma_test):
    shape = [start_slice, end_slice]

    gamma_configuration = {
        'rebin_size': rebin_size,
        'output_file': "../gamma.hdf5",
        'shape': shape,
        'paths': gamma_train[0][0],
        'as_channels': False
    }

    gamma_train_preprocessor = EventFilePreprocessor(config=gamma_configuration)
    print(gamma_train_preprocessor.shape)

    gamma_configuration['paths'] = gamma_train[1][0]

    gamma_validate_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    gamma_configuration['paths'] = gamma_test[1][0]

    gamma_test_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    energy_train = EventFileGenerator(paths=gamma_train[0][0], batch_size=8,
                                      preprocessor=gamma_train_preprocessor,
                                      as_channels=False,
                                      final_slices=final_slices,
                                      slices=(start_slice, end_slice),
                                      augment=True,
                                      training_type='Separation')

    energy_validate = EventFileGenerator(paths=gamma_train[1][0], batch_size=8,
                                         preprocessor=gamma_validate_preprocessor,
                                         as_channels=False,
                                         final_slices=final_slices,
                                         slices=(start_slice, end_slice),
                                         augment=False,
                                         training_type='Separation')

    energy_test = EventFileGenerator(paths=gamma_test[1][0], batch_size=8,
                                     preprocessor=gamma_test_preprocessor,
                                     as_channels=False,
                                     final_slices=final_slices,
                                     slices=(start_slice, end_slice),
                                     augment=False,
                                     training_type='Separation')

    final_shape = (final_slices, gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], 1)

    return energy_train, energy_validate, energy_test, final_shape


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


def fit_model(separation_model, train_gen, val_gen):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                               patience=5,
                                               verbose=0, mode='auto',
                                               restore_best_weights=False)
    model_checkpoint = keras.callbacks.ModelCheckpoint("models/gpyopt_thesis_disp_{val_loss:0.4}.hdf5",
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto', period=1)
    separation_model.fit_generator(
        generator=train_gen,
        epochs=200,
        verbose=2,
        steps_per_epoch=int(len(train_gen)/10),
        validation_data=val_gen,
        callbacks=[early_stop, model_checkpoint],
        use_multiprocessing=True,
        workers=10,
        max_queue_size=300,
    )
    return separation_model


def model_evaluate(separation_model, test_gen):
    evaluation = separation_model.evaluate_generator(
        generator=test_gen,
        verbose=0,
        use_multiprocessing=True,
        workers=10,
        max_queue_size=300,
    )
    return evaluation

# function to run mnist class

def run_mnist(neuron_1=64, kernel_1=3, strides_1=1, act_1=3, drop_1=0.3, rec_drop_1=0.3,
              rec_act_1=2, dense_neuron_1=64, dense_neuron_2=128, optimizer=0,
              pool=True, dense_act_1=0, dense_act_2=0, dense_drop_1=0.5, dense_drop_2=0.5, start_slice=30, end_slice=70, final_slices=5, rebin_size=5):
    train_gen, val_gen, test_gen, shape = data(start_slice=start_slice, end_slice=end_slice, final_slices=final_slices, rebin_size=rebin_size, gamma_train=gamma_train, gamma_test=gamma_indexes)
    separation_model = create_model(shape=shape, neuron_1=neuron_1, kernel_1=kernel_1, strides_1=strides_1, act_1=act_1, drop_1=drop_1, rec_drop_1=rec_drop_1,
                                    rec_act_1=rec_act_1, dense_neuron_1=dense_neuron_1, dense_neuron_2=dense_neuron_2, optimizer=optimizer,
                                    pool=pool, dense_act_1=dense_act_1, dense_act_2=dense_act_2, dense_drop_1=dense_drop_1, dense_drop_2=dense_drop_2)
    separation_model = fit_model(separation_model, train_gen, val_gen)
    evaluation = model_evaluate(separation_model, test_gen)
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

          {'name': 'start_slice', 'type': 'discrete', 'domain': (30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85)},
          {'name': 'end_slice', 'type': 'discrete', 'domain': (35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)},
          {'name': 'final_slices', 'type': 'discrete', 'domain': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)},
          {'name': 'rebin_size', 'type': 'discrete', 'domain': (3, 4, 5, 6, 7, 8, 9, 10)},

          ]

constraints = [{'name': 'constr_1', 'constraint': 'x[:,20] - x[:,21] + x[:,22]'},
               ]


def f(x):
    print(x)
    evaluation = run_mnist(
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
        dense_drop_2=float(x[:, 3]),
        start_slice=int(x[:,15]),
        end_slice=int(x[:,16]),
        final_slices=int(x[:,17]),
        rebin_size=int(x[:,18]))
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]


if __name__ == '__main__':
    opt_mnist = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, constraints=constraints)

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
    \t{14}:\t{15}
    \t{16}:\t{17}
    \t{18}:\t{19}
    \t{20}:\t{21}
    \t{22}:\t{23}
    \t{24}:\t{25}
    \t{26}:\t{27}
    \t{28}:\t{29}
    \t{30}:\t{31}
    \t{32}:\t{33}
    \t{34}:\t{35}
    \t{36}:\t{37}
    """.format(bounds[0]["name"], opt_mnist.x_opt[0],
               bounds[1]["name"], opt_mnist.x_opt[1],
               bounds[2]["name"], opt_mnist.x_opt[2],
               bounds[3]["name"], opt_mnist.x_opt[3],
               bounds[4]["name"], opt_mnist.x_opt[4],
               bounds[5]["name"], opt_mnist.x_opt[5],
               bounds[6]["name"], opt_mnist.x_opt[6],
               bounds[7]["name"], opt_mnist.x_opt[7],
               bounds[8]["name"], opt_mnist.x_opt[8],
               bounds[9]["name"], opt_mnist.x_opt[9],
               bounds[10]["name"], opt_mnist.x_opt[10],
               bounds[11]["name"], opt_mnist.x_opt[11],
               bounds[12]["name"], opt_mnist.x_opt[12],
               bounds[13]["name"], opt_mnist.x_opt[13],
               bounds[14]["name"], opt_mnist.x_opt[14],
               bounds[15]["name"], opt_mnist.x_opt[15],
               bounds[16]["name"], opt_mnist.x_opt[16],
               bounds[17]["name"], opt_mnist.x_opt[17],
               bounds[18]["name"], opt_mnist.x_opt[18],))
