import talos as ta
from keras.activations import relu, elu, softmax, hard_sigmoid, tanh
from keras.layers import Flatten, ConvLSTM2D, Dense, MaxPooling3D
from keras.losses import categorical_crossentropy, logcosh
from keras.models import Sequential
from keras.optimizers import adam, nadam, rmsprop
from talos.model.early_stopper import early_stopper
from talos.model.layers import hidden_layers
from talos.model.normalizers import lr_normalizer
from talos.metrics.keras_metrics import root_mean_squared_error, fmeasure_acc, matthews_correlation_acc, precision_acc, recall_acc


from factnn.utils.cross_validate import get_chunk_of_data

# Parameter Dictionary for talos

params = {'lr': (1, 10, 10),
          'first_neuron': [4, 16, 32],
          'last_neuron': [4, 16],
          'hidden_layers': [2, 4],
          'batch_size': [2, 8],
          'epochs': [500],
          'dropout': (0, 0.80, 4),
          'weight_regulizer': [None],
          'emb_output_dims': [None],
          'optimizer': [adam, nadam, rmsprop],
          'losses': [categorical_crossentropy, logcosh],
          'activation': [relu, elu],

          'neuron_1': [8, 16, 64],
          'kernel_1': [1, 3, 5],
          'stride_1': [1, 2, 3],
          'rec_dropout': [0.0, 0.8, 4],
          'rec_act': [hard_sigmoid, tanh],
          'layer_drop': [0.0, 0.8, 4],
          'layers': [2,3,4],
          'pool': [0, 1],

          }
'''

'neuron_2': [4, 8, 16, 32, 64],
'kernel_2': [1, 2, 3],
'stride_2': [1, 2],
'three': [0, 1],
'neuron_3': [4, 8, 16, 32, 64],
'kernel_3': [1, 2, 3],
'stride_3': [1, 2],
'''

def input_model(x_train, y_train, x_val, y_val, params):
    model = Sequential()

    model.add(ConvLSTM2D(params['neuron_1'], kernel_size=params['kernel_1'], strides=params['stride_1'],
                         padding='same',
                         input_shape=(10, 100, 100, 1),
                         activation=params['activation'],
                         dropout=params['layer_drop'], recurrent_dropout=params['rec_dropout'],
                         recurrent_activation=params['rec_act'],
                         return_sequences=True,
                         stateful=False))
    if params['pool']:
        model.add(MaxPooling3D())
    if params['layers'] >= 2:
        model.add(ConvLSTM2D(params['neuron_1'], kernel_size=params['kernel_1'], strides=params['stride_1'],
                             padding='same',
                             activation=params['activation'],
                             dropout=params['layer_drop'], recurrent_dropout=params['rec_dropout'],
                             recurrent_activation=params['rec_act'],
                             return_sequences=True,
                             stateful=False))
        #if params['pool']:
        #    model.add(MaxPooling3D())

    if params['layers'] >= 3:
        model.add(ConvLSTM2D(params['neuron_1'], kernel_size=params['kernel_1'], strides=params['stride_1'],
                             padding='same',
                             activation=params['activation'],
                             dropout=params['layer_drop'], recurrent_dropout=params['rec_dropout'],
                             recurrent_activation=params['rec_act'],
                             return_sequences=True,
                             stateful=False))
        if params['pool']:
            model.add(MaxPooling3D())
    if params['layers'] >= 4:
        model.add(ConvLSTM2D(params['neuron_1'], kernel_size=params['kernel_1'], strides=params['stride_1'],
                             padding='same',
                             activation=params['activation'],
                             dropout=params['layer_drop'], recurrent_dropout=params['rec_dropout'],
                             recurrent_activation=params['rec_act'],
                             return_sequences=True,
                             stateful=False))
        #if params['pool']:
        #    model.add(MaxPooling3D())

    '''
    if params['pool']:
        model.add(MaxPooling2D())
    model.add(Conv2D(params['neuron_2'], kernel_size=params['kernel_2'], strides=params['stride_2'],
                     padding='same', activation=params['activation']))
    model.add(Dropout(params['layer_drop']))

    if params['three']:
        model.add(Conv2D(params['neuron_3'], kernel_size=params['kernel_3'], strides=params['stride_3'],
                         padding='same', activation=params['activation']))
        model.add(MaxPooling2D())
        model.add(Dropout(model.add('layer_drop')))
    '''
    model.add(Flatten())

    hidden_layers(model, params, params['last_neuron'])

    model.add(Dense(2, activation='softmax'))

    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'](lr_normalizer(params['lr'], params['optimizer'])),
                  metrics=['acc',
                           fmeasure_acc,
                           matthews_correlation_acc,
                           precision_acc,
                           recall_acc])

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'], verbose=0,
                    validation_data=[x_val, y_val],
                    callbacks=[early_stopper(params['epochs'],
                                             mode='moderate')])

    return out, model


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

x, y = get_chunk_of_data(directory=gamma_dir, proton_directory=proton_dir, indicies=(30, 129, 10), rebin=100,
                         chunk_size=args['size'], as_channels=False)

print("Got data")
print("X Shape", x.shape)
print("Y Shape", y.shape)
history = ta.Scan(x, y,
                  params=params,
                  dataset_name='time_separation_test',
                  experiment_no='1',
                  model=input_model,
                  search_method='random',
                  grid_downsample=args['grid'])
