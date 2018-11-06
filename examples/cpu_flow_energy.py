#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from factnn import GammaPreprocessor, ProtonPreprocessor, SeparationGenerator, SeparationModel, ObservationPreprocessor, EnergyGenerator
import os.path
from factnn.utils import kfold
from keras.models import load_model

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]
proton_dir = [base_dir + "sim/proton/"]

shape = [30,70]
rebin_size = 5

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

energy_gen_config = {
    'seed': 1337,
    'batch_size': 32,
    'start_slice': 0,
    'number_slices': shape[1] - shape[0],
    'mode': 'train',
    'chunked': False,
    'augment': True,
    'from_directory': True,
    'input_shape': [-1, gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
    'as_channels': True,
}

energy_train = EnergyGenerator(config=energy_gen_config)
energy_validate = EnergyGenerator(config=energy_gen_config)
energy_validate.mode = 'validate'

energy_train.train_preprocessor = gamma_train_preprocessor
energy_train.validate_preprocessor = gamma_validate_preprocessor

energy_validate.train_preprocessor = gamma_train_preprocessor
energy_validate.validate_preprocessor = gamma_validate_preprocessor

from keras.layers import Dense, Dropout, Flatten, ConvLSTM2D, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, PReLU, BatchNormalization, ReLU
from keras.models import Sequential
import keras
import numpy as np

separation_model = Sequential()

#separation_model.add(ConvLSTM2D(32, kernel_size=3, strides=2,
#                     padding='same', input_shape=[gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
#                     activation='relu',
#                     dropout=0.3, recurrent_dropout=0.5,
#                     return_sequences=True))

#separation_model.add(BatchNormalization())
separation_model.add(Conv2D(32, input_shape=[gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], 5],
                            kernel_size=3, strides=1,
                            padding='same'))
separation_model.add(ReLU())
separation_model.add(MaxPooling2D())
separation_model.add(BatchNormalization())
separation_model.add(Conv2D(32,
                            kernel_size=3, strides=1,
                            padding='same'))
separation_model.add(ReLU())
separation_model.add(MaxPooling2D())
separation_model.add(BatchNormalization())
separation_model.add(Dropout(0.4))
separation_model.add(Flatten())
separation_model.add(Dense(32))
separation_model.add(ReLU())
separation_model.add(Dropout(0.5))
separation_model.add(Dense(64))
separation_model.add(ReLU())
#separation_model.add(Dense(2, activation='softmax'))
#separation_model.compile(optimizer='adam', loss='categorical_crossentropy',
#                         metrics=['acc'])

# For energy

def r2(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return -1.*(1 - SS_res / (SS_tot + K.epsilon()))

separation_model.add(Dense(1, activation='linear'))
separation_model.compile(optimizer='adam', loss='mse',
                         metrics=['mae', r2])

separation_model.summary()
model_checkpoint = keras.callbacks.ModelCheckpoint("Outside_energy_relu_norm.hdf5",
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

event_totals = 0.8*NUM_EVENTS_GAMMA
train_num = (event_totals * 0.8)
val_num = event_totals * 0.2

separation_model.fit_generator(
    generator=energy_train,
    steps_per_epoch=int(np.floor(train_num / energy_train.batch_size)),
    epochs=500,
    verbose=1,
    validation_data=energy_validate,
    callbacks=[early_stop, model_checkpoint, tensorboard],
    validation_steps=int(np.floor(val_num / energy_validate.batch_size))
)


# Save the base model to use for the kfold validation
"""

Now run the models with the generators!

"""

import numpy as np
from examples.open_crab_sample_constants import NUM_EVENTS_PROTON
import matplotlib.pyplot as plt

gamma_configuration = {
    'rebin_size': rebin_size,
    'output_file': "../gamma.hdf5",
    'shape': shape,
    'paths': gamma_paths,
    'as_channels': True
}

proton_configuration = {
    'rebin_size': rebin_size,
    'output_file': "../proton.hdf5",
    'shape': shape,
    'paths': crab_paths,
    'as_channels': True
}


proton_test_preprocessor = ProtonPreprocessor(config=proton_configuration)
gamma_test_preprocessor = GammaPreprocessor(config=gamma_configuration)

separation_model = load_model("Outside_sep_prelu.hdf5")

separation_validate = SeparationGenerator(config=separation_generator_configuration)
separation_validate.mode = "test"
separation_validate.test_preprocessor = gamma_train_preprocessor
separation_validate.proton_test_preprocessor = proton_test_preprocessor

num_events = NUM_EVENTS_PROTON
steps = int(np.floor(num_events/16))
truth = []
predictions = []
for i in range(steps):
    print("Step: " + str(i) + "/" + str(steps))
    # Get each batch and test it
    test_images, test_labels = next(separation_validate)
    test_predictions = separation_model.predict_on_batch(test_images)
    predictions.append(test_predictions)
    truth.append(test_labels)

predictions = np.asarray(predictions).reshape(-1, )
truth = np.asarray(truth).reshape(-1, )

from factnn import plotting

plot = plotting.plot_roc(truth, predictions)
plt.show()
