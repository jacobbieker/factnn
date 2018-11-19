# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from factnn import GammaPreprocessor, ProtonPreprocessor
from factnn.generator.keras.eventfile_generator import EventFileGenerator
from factnn.data.preprocess.eventfile_preprocessor import EventFilePreprocessor
import os.path
from factnn.utils import kfold
from keras.models import load_model

base_dir = "/home/jacob/Development/event_files/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "gamma/"]
proton_dir = [base_dir + "proton/"]

shape = [30, 70]
rebin_size = 3

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
    'as_channels': True
}

proton_configuration = {
    'rebin_size': rebin_size,
    'output_file': "../proton.hdf5",
    'shape': shape,
    'paths': proton_indexes[0][0],
    'as_channels': True
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
    'as_channels': True,
}

energy_train = EventFileGenerator(paths=gamma_indexes[0][0], batch_size=16,
                                  preprocessor=gamma_train_preprocessor,
                                  as_channels=True,
                                  final_slices=5,
                                  slices=(30, 70),
                                  augment=True,
                                  training_type='Energy')

energy_validate = EventFileGenerator(paths=gamma_indexes[1][0], batch_size=16,
                                     preprocessor=gamma_validate_preprocessor,
                                     as_channels=True,
                                     final_slices=5,
                                     slices=(30, 70),
                                     augment=True,
                                     training_type='Energy')

from keras.layers import Dense, Dropout, Flatten, ConvLSTM2D, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, PReLU, \
    BatchNormalization, ReLU
from keras.models import Sequential
import keras
import numpy as np

separation_model = Sequential()

# separation_model.add(ConvLSTM2D(32, kernel_size=3, strides=2,
#                     padding='same', input_shape=[gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
#                     activation='relu',
#                     dropout=0.3, recurrent_dropout=0.5,
#                     return_sequences=True))

# separation_model.add(BatchNormalization())
separation_model.add(Conv2D(32, input_shape=[gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], 5],
                            kernel_size=3, strides=1,
                            padding='same'))
separation_model.add(PReLU())
separation_model.add(MaxPooling2D())
# separation_model.add(BatchNormalization())
separation_model.add(Conv2D(48,
                            kernel_size=3, strides=1,
                            padding='same'))
separation_model.add(PReLU())
separation_model.add(MaxPooling2D())
separation_model.add(Conv2D(64,
                            kernel_size=3, strides=1,
                            padding='same'))
separation_model.add(PReLU())
separation_model.add(MaxPooling2D())
separation_model.add(Conv2D(128,
                            kernel_size=3, strides=1,
                            padding='same'))
separation_model.add(PReLU())
separation_model.add(MaxPooling2D())
# separation_model.add(BatchNormalization())
separation_model.add(Dropout(0.2))
separation_model.add(Flatten())
separation_model.add(Dense(32))
separation_model.add(PReLU())
separation_model.add(Dropout(0.2))
separation_model.add(Dense(64))
separation_model.add(PReLU())


# For energy

def r2(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return -1. * (1 - SS_res / (SS_tot + K.epsilon()))


separation_model.add(Dense(1, activation='linear'))
separation_model.compile(optimizer='adam', loss='mse',
                         metrics=['mae', r2])

separation_model.summary()
model_checkpoint = keras.callbacks.ModelCheckpoint("Outside_energy_prelu_large_{val_loss:.2f}.hdf5",
                                                   monitor='val_loss',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto', period=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                           patience=10,
                                           verbose=0, mode='auto')

tensorboard = keras.callbacks.TensorBoard(update_freq='epoch', log_dir='./energy_log_prelu_eventlist')

separation_model.fit_generator(
    generator=energy_train,
    epochs=500,
    verbose=1,
    validation_data=energy_validate,
    callbacks=[early_stop, model_checkpoint, tensorboard],
    use_multiprocessing=True,
    workers=12,
    max_queue_size=300,
)

# Save the base model to use for the kfold validation
"""

Now run the models with the generators!

"""
