import os
import keras

from factnn.utils.cross_validate import cross_validate

from keras.layers import Flatten, ConvLSTM2D, MaxPooling2D, Dense, Activation, Dropout, MaxPooling3D, Conv2D, \
    AveragePooling2D, AveragePooling3D
from keras.models import Sequential
import keras
import numpy as np


directory = "/home/jacob/Documents/iact_events/"
gamma_dir = directory + "gamma/clump20/"
proton_dir = directory + "proton/clump20/"


model = Sequential()

model.add(Conv2D(64, kernel_size=3, strides=1,
                            padding='same',
                            input_shape=(50,50,3)))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.1))
#model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=3, strides=1,
                            padding='same',
                            ))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))
#model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=3, strides=1,
                            padding='same',
                            ))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.1))
#model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=3, strides=1,
                            padding='same',
                            ))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Conv2D(256, kernel_size=3, strides=1,
                 padding='same',
                 ))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(128))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(Dropout(0.1))
#model.add(Dense(256))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.1))
def r2(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return -1. * (1 - SS_res / (SS_tot + K.epsilon()))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse',
                         metrics=['mae', r2])
model.summary()

results = cross_validate(model, gamma_dir, proton_directory=proton_dir, indicies=(30, 129, 3), rebin=50,
                   as_channels=True, kfolds=5, model_type="Energy", normalize=False, batch_size=32,
                   workers=10, verbose=1, truncate=True, dynamic_resize=True, equal_slices=False, plot=False)

