import fact
import fact_plots
import h5py
from fact.io import read_h5py
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Reshape, Flatten, ConvLSTM2D, Conv2D, Conv1D, MaxPooling1D, Conv3D, MaxPooling3D, MaxPooling2D, LSTM, Reshape
import keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import keras

columns = [
    "conc_core",
    "concentration_one_pixel",
    "concentration_two_pixel",
    "leakage",
    "leakage2",
    "size",
    "width",
    "length",
    "conc_cog",
    "m3l",
    "m3t",
    "num_islands",
    "num_pixel_in_shower",
    "ph_charge_shower_max",
    "ph_charge_shower_mean",
    "ph_charge_shower_min",
    "ph_charge_shower_variance",
    "ph_charge_shower_kurtosis",
    "ph_charge_shower_skewness",
    "photoncharge_mean",
    "max_slopes_shower_kurtosis",
    "max_slopes_shower_max",
    "max_slopes_shower_mean",
    "max_slopes_shower_min",
    "max_slopes_shower_skewness",
    "max_slopes_shower_variance",
    "arr_time_shower_kurtosis",
    "arr_time_shower_max",
    "arr_time_shower_mean",
    "arr_time_shower_min",
    "arr_time_shower_skewness",
    "arr_time_shower_variance",
    "slope_long",
    "slope_spread",
    "slope_spread_weighted",
    "slope_trans",
    "timespread",
    "timespread_weighted"
]

protons = read_h5py("proton.hdf5", key="events", columns=columns, last=480000)
gammas = read_h5py("gamma.hdf5", key="events", columns=columns, last=480000)

training_gammas = gammas[0:int(0.8*len(gammas))]
validation_gammas = gammas[int(0.6*len(gammas)):int(0.8*len(gammas))]
testing_gammas = gammas[int(0.8*len(gammas)):]

training_protons = protons[0:int(0.8*len(protons))]
validation_protons = protons[int(0.6*len(protons)):int(0.8*len(protons))]
testing_protons = protons[int(0.8*len(protons)):]


labels = np.array([True] * (len(training_gammas)) + [False] * len(training_protons))
batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
batch_images = pd.concat([training_gammas, training_protons], axis=0)
batch_images = batch_images.values

batch_images, batch_image_label = shuffle(batch_images, batch_image_label)

print(batch_images.shape)
print(batch_image_label.shape)

batch_images = batch_images.reshape((-1,38,1))

model = Sequential()

model.add(
    Conv1D(64, input_shape=(38,1), kernel_size=3, strides=1,
           padding='same', activation='relu'))
model.add(Dropout(0.2))
#model.add(
#    Conv1D(64, kernel_size=3, strides=1,
#           padding='same', activation='relu'))
#model.add(Dropout(0.2))
#model.add(
#    Conv1D(128, kernel_size=3, strides=1,
#           padding='same', activation='relu'))
#model.add(Dropout(0.2))
model.add(
    Conv1D(256, kernel_size=3, strides=1,
           padding='same', activation='relu'))
model.add(Reshape((-1,)))
model.add(Dropout(0.4))
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.3))
#model.add(Dense(512, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
#model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
#model.add(BatchNormalization())
#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.3))

#Final Dense layer
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

model_checkpoint = keras.callbacks.ModelCheckpoint("ConvfeatureSepTest.h5",
                                                   monitor='val_loss',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto', period=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                           patience=10,
                                           verbose=0, mode='auto')

tb = keras.callbacks.TensorBoard(log_dir='/home/jacob/', histogram_freq=1, batch_size=32, write_graph=True,
                                 write_grads=True,
                                 write_images=False,
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None)
model.fit(x=batch_images, y=batch_image_label, batch_size=1024, epochs=500, validation_split=0.2, verbose=2,
          callbacks=[early_stop, model_checkpoint, tb])
