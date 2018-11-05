import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from factnn import GammaPreprocessor, ProtonPreprocessor, SeparationGenerator, SeparationModel, ObservationPreprocessor
import os.path
from factnn.utils import kfold
from keras.models import load_model

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]
proton_dir = [base_dir + "sim/proton/"]

shape = [0,100]
rebin_size = 90

# Get paths from the directories
gamma_paths = []
for directory in gamma_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                gamma_paths.append(os.path.join(root, file))


# Get paths from the directories
crab_paths = []
for directory in proton_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
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


proton_train_preprocessor = ProtonPreprocessor(config=proton_configuration)
gamma_train_preprocessor = GammaPreprocessor(config=gamma_configuration)

gamma_configuration['paths'] = gamma_indexes[1][0]
proton_configuration['paths'] = proton_indexes[1][0]

proton_validate_preprocessor = ProtonPreprocessor(config=proton_configuration)
gamma_validate_preprocessor = GammaPreprocessor(config=gamma_configuration)


separation_generator_configuration = {
    'seed': 1337,
    'batch_size': 16,
    'start_slice': 0,
    'number_slices': shape[1] - shape[0],
    'mode': 'train',
    'chunked': False,
    'augment': True,
    'from_directory': True,
    'input_shape': [-1, gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
    'as_channels': True,
}

separation_validate = SeparationGenerator(config=separation_generator_configuration)
separation_train = SeparationGenerator(config=separation_generator_configuration)

separation_validate.mode = "validate"
separation_train.mode = "train"

separation_train.proton_train_preprocessor = proton_train_preprocessor
separation_train.proton_validate_preprocessor = proton_validate_preprocessor
separation_train.train_preprocessor = gamma_train_preprocessor
separation_train.validate_preprocessor = gamma_validate_preprocessor

separation_validate.proton_train_preprocessor = proton_train_preprocessor
separation_validate.proton_validate_preprocessor = proton_validate_preprocessor
separation_validate.train_preprocessor = gamma_train_preprocessor
separation_validate.validate_preprocessor = gamma_validate_preprocessor


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

