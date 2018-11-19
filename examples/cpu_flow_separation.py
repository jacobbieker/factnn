#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from factnn import GammaPreprocessor, ProtonPreprocessor
from factnn.generator.keras.eventfile_generator import EventFileGenerator
from factnn.data.preprocess.eventfile_preprocessor import EventFilePreprocessor
import os.path
from factnn.utils import kfold
from keras.models import load_model

base_dir = "/home/jacob/Development/event_files/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "diffuse_gamma/"]
proton_dir = [base_dir + "proton/"]

shape = [30, 80]
rebin_size = 3

# Get paths from the directories
gamma_paths = []
for directory in gamma_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            gamma_paths.append(os.path.join(root, file))

# Now do the Kfold Cross validation Part for both sets of paths
gamma_indexes = kfold.split_data(gamma_paths, kfolds=5)

gamma_configuration = {
    'rebin_size': rebin_size,
    'output_file': "../gamma.hdf5",
    'shape': shape,
    'paths': gamma_indexes[0][0],
    'as_channels': True
}

gamma_train_preprocessor = EventFilePreprocessor(config=gamma_configuration)

gamma_configuration['paths'] = gamma_indexes[1][0]

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
                                  training_type='Disp')

energy_validate = EventFileGenerator(paths=gamma_indexes[1][0], batch_size=16,
                                     preprocessor=gamma_validate_preprocessor,
                                     as_channels=True,
                                     final_slices=5,
                                     slices=(30, 70),
                                     augment=False,
                                     training_type='Disp')

from keras.layers import Dense, Dropout, Flatten, ConvLSTM2D, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, PReLU, ReLU, BatchNormalization
from keras.models import Sequential
import keras
import numpy as np

def r2(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return -1.*(1 - SS_res / (SS_tot + K.epsilon()))

import keras.losses
keras.losses.r2 = r2

separation_model = Sequential()

#separation_model.add(ConvLSTM2D(32, kernel_size=3, strides=2,
#                     padding='same', input_shape=[gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
#                     activation='relu',
#                     dropout=0.3, recurrent_dropout=0.5,
#                     return_sequences=True))

#separation_model.add(BatchNormalization()
separation_model.add(Conv2D(64, input_shape=[gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], 5],
                            kernel_size=3, strides=1,
                            padding='same'))
separation_model.add(PReLU())
separation_model.add(MaxPooling2D())
#separation_model.add(BatchNormalization())
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
#separation_model.add(BatchNormalization())
separation_model.add(Dropout(0.2))
separation_model.add(Flatten())
separation_model.add(Dense(32))
separation_model.add(ReLU())
separation_model.add(Dropout(0.2))
separation_model.add(Dense(64))
separation_model.add(ReLU())
separation_model.add(Dense(1, activation='linear'))
separation_model.compile(optimizer='adam', loss='mse',
              metrics=['mae', r2])

#separation_model = load_model("Outside_cpu_source_-0.10.hdf5")
separation_model.summary()

model_checkpoint = keras.callbacks.ModelCheckpoint("Outside_cpu_source_large_{val_loss:.2f}.hdf5",
                                                   monitor='val_loss',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto', period=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                           patience=10,
                                           verbose=0, mode='auto')

tensorboard = keras.callbacks.TensorBoard(update_freq='epoch', write_images=True, log_dir='./source_log')

from examples.open_crab_sample_constants import NUM_EVENTS_GAMMA, NUM_EVENTS_PROTON


separation_model.fit_generator(
    generator=energy_train,
    epochs=500,
    steps_per_epoch=len(energy_train),
    verbose=1,
    validation_data=energy_validate,
    callbacks=[early_stop, model_checkpoint, tensorboard],
    use_multiprocessing=True,
    workers=10,
    max_queue_size=300
)

"""

from keras.layers import Dense, Dropout, Flatten, ConvLSTM2D, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, PReLU
from keras.models import Sequential
import keras
import numpy as np

separation_model = Sequential()

#separation_model.add(ConvLSTM2D(32, kernel_size=3, strides=2,
#                     padding='same', input_shape=[gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
#                     activation='relu',
#                     dropout=0.3, recurrent_dropout=0.5,
#                     return_sequences=True))

separation_model.add(Conv2D(2, input_shape=[gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], 1],
                            kernel_size=1, strides=1,
                            padding='same'))
separation_model.add(PReLU())
separation_model.add(Conv2D(2,
                            kernel_size=3, strides=1,
                            padding='same'))
separation_model.add(PReLU())
separation_model.add(MaxPooling2D())
separation_model.add(Dropout(0.4))
separation_model.add(Flatten())
separation_model.add(Dense(2))
separation_model.add(PReLU())
separation_model.add(Dropout(0.5))
separation_model.add(Dense(2))
separation_model.add(PReLU())
separation_model.add(Dense(2, activation='softmax'))
separation_model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])

separation_model.summary()
model_checkpoint = keras.callbacks.ModelCheckpoint("Outside_sep_prelu_small3560.hdf5",
                                                   monitor='val_loss',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto', period=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                           patience=10,
                                           verbose=0, mode='auto')

tensorboard = keras.callbacks.TensorBoard(update_freq='epoch')

from examples.open_crab_sample_constants import NUM_EVENTS_GAMMA, NUM_EVENTS_PROTON

event_totals = 0.8*NUM_EVENTS_PROTON
train_num = 16000#(event_totals * 0.8)
val_num = event_totals * 0.2

separation_model.fit_generator(
    generator=separation_train,
    steps_per_epoch=int(np.floor(train_num / separation_train.batch_size)),
    epochs=500,
    verbose=1,
    validation_data=separation_validate,
    callbacks=[early_stop, model_checkpoint, tensorboard],
    validation_steps=int(np.floor(val_num / separation_validate.batch_size))
)

"""