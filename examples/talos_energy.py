import talos as ta
from keras.activations import relu, elu, softmax, hard_sigmoid, tanh
from keras.layers import Flatten, ConvLSTM2D, Dense, Conv2D, MaxPooling2D, Dropout
from keras.losses import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.optimizers import adam, nadam, rmsprop
from talos.model.early_stopper import early_stopper
from talos.model.layers import hidden_layers
from talos.model.normalizers import lr_normalizer
from talos.metrics.keras_metrics import root_mean_squared_error

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from factnn.utils.cross_validate import get_chunk_of_data, get_data_generators

import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
                help="Directory for Files")
ap.add_argument("-s", "--size", type=int, required=True,
                help="Chunk Size")
ap.add_argument("-g", "--grid", type=float, required=True,
                help="Downsample Amount")

args = vars(ap.parse_args())

directory = args['dir'] # "/media/jacob/WDRed8Tb1/Insync/iact_events/"
gamma_dir = [directory + "gammaFeature/no_clean/"]
proton_dir = [directory + "protonFeature/no_clean/"]

# Parameter Dictionary for talos

params = {'lr': (1, 10, 5),
          'first_neuron': [16, 64],
          'last_neuron': [8, 32],
          'hidden_layers': [2, 3, 6],
          'batch_size': [4, 16],
          'epochs': [500],
          'dropout': (0, 0.80, 3),
          'weight_regulizer': [None],
          'emb_output_dims': [None],
          'optimizer': [adam, nadam, rmsprop],
          'losses': [mean_squared_error],
          'activation': [relu, elu, hard_sigmoid],
          'last_activation': [softmax],

          'neuron_1': [8, 64],
          'kernel_1': [1, 3, 5, 7],
          'stride_1': [1],
          'layer_drop': [0.0, 0.8, 3],
          'layers': [2, 3, 4],
          'pool': [0, 1],
          'rebin': [25, 50, 75],
          'time': [1, 2, 3]

          }
'''
'rec_dropout': [0.0, 0.4, 5],
'rec_act': [hard_sigmoid, tanh],
'pool': [0, 1],
'neuron_2': [4, 8, 16, 32, 64],
'kernel_2': [1, 2, 3],
'stride_2': [1, 2],
'three': [0, 1],
'neuron_3': [4, 8, 16, 32, 64],
'kernel_3': [1, 2, 3],
'stride_3': [1, 2],
'''

def input_model(x_train, y_train, x_val, y_val, params):
    train_gen, val_gen, _, _ = get_data_generators(directory=gamma_dir, proton_directory=proton_dir,
                                                   indicies=(30, 129, params['time']), rebin=params['rebin'],
                                                   batch_size=params['batch_size'], as_channels=True,
                                                   max_elements=args['size'], model_type="Energy")
    model = Sequential()

    model.add(Conv2D(params['neuron_1'], kernel_size=params['kernel_1'], strides=params['stride_1'],
                         padding='same',
                         input_shape=(params['rebin'], params['rebin'], params['time']),
                         activation=params['activation']))
    if params['pool']:
        model.add(MaxPooling2D())
    if params['dropout'] > 0.001:
        model.add(Dropout(params['layer_drop']))
    model.add(Conv2D(params['neuron_1'], kernel_size=params['kernel_1'], strides=params['stride_1'],
                     padding='same', activation=params['activation']))
    #if params['pool']:
    #    model.add(MaxPooling2D())
    if params['dropout'] > 0.001:
        model.add(Dropout(params['layer_drop']))

    if params['layers'] >= 3:
        model.add(Conv2D(params['neuron_1'], kernel_size=params['kernel_1'], strides=params['stride_1'],
                         padding='same', activation=params['activation']))
        if params['pool']:
            model.add(MaxPooling2D())
        if params['dropout'] > 0.001:
            model.add(Dropout(params['layer_drop']))

    if params['layers'] >= 4:
        model.add(Conv2D(params['neuron_1'], kernel_size=params['kernel_1'], strides=params['stride_1'],
                         padding='same', activation=params['activation']))
     #   if params['pool']:
     #       model.add(MaxPooling2D())
        if params['dropout'] > 0.001:
            model.add(Dropout(params['layer_drop']))
    model.add(Flatten())

    hidden_layers(model, params, params['last_neuron'])

    model.add(Dense(1, activation='linear'))

    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'](lr_normalizer(params['lr'], params['optimizer'])),
                  metrics=['mae', 'mse'])

    out = model.fit_generator(
        generator=train_gen,
        epochs=params['epochs'],
        verbose=0,
        validation_data=val_gen,
        callbacks=[early_stopper(params['epochs'],
                                 mode='moderate')],
        use_multiprocessing=True,
        workers=5,
        max_queue_size=50,
    )
    return out, model

x, y = get_chunk_of_data(directory=gamma_dir, indicies=(30, 129, 3), rebin=100,
                         chunk_size=2, as_channels=True, model_type='Energy')
print("Got data")
print("X Shape", x.shape)
print("Y Shape", y.shape)
history = ta.Scan(x, y,
                  params=params,
                  dataset_name='energy_test',
                  experiment_no='5',
                  clear_tf_session=False,
                  model=input_model,
                  search_method='random',
                  grid_downsample=0.00001)
