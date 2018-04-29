import numpy as np
import argparse

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import to_categorical, Sequence
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
tf.logging.set_verbosity(tf.logging.ERROR)

from fact.analysis.binning import bin_runs
from fact.io import read_h5py, to_h5py
import h5py
import keras
import yaml
import pandas as pd
from FACTsourceFinding.fact_generators import SimDataGenerator, DataGenerator


architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/thesis'



early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')

kernel_size = (5, 5)
batch_size = 8
inpurt_shape = (75, 75, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=kernel_size, strides=(1, 1),
                 activation='relu', padding='same',
                 input_shape=inpurt_shape))
# model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size, strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size, strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size, strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size, strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size, strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='linear'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
