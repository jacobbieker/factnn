import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import pandas as pd
import numpy as np
import random
import pickle
import h5py
import gzip
import time
import csv
import sys
import os

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Flatten, Reshape, Conv2D, Conv3D, MaxPooling3D
from keras.layers.noise import AlphaDropout
from keras.utils import to_categorical, Sequence
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

pickled_mapping_loc = os.path.join("..", "thesisTools", "output", "hexagon_to_cube_mapping.p")
path_mc_images = '/tree/tf/MC_Cube_Images.h5'

'''
def metaYielder(path_mc_images):
    with h5py.File(path_mc_images, 'r') as f:
        keys = list(f.keys())
        events = []
        for key in keys:
            events.append(len(f[key]))

    gamma_anteil = events[0]/np.sum(events)
    hadron_anteil = events[1]/np.sum(events)

    gamma_count = int(round(num_events*gamma_anteil))
    hadron_count = int(round(num_events*hadron_anteil))

    return gamma_anteil, hadron_anteil, gamma_count, hadron_count

def batchYielder(path_mc_images):
    gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder(path_mc_images)

    gamma_batch_size = int(round(batch_size*gamma_anteil))
    hadron_batch_size = int(round(batch_size*hadron_anteil))

    for step in range(num_steps):
        gamma_offset = (step * gamma_batch_size) % (gamma_count - gamma_batch_size)
        hadron_offset = (step * hadron_batch_size) % (hadron_count - hadron_batch_size)

        with h5py.File(path_mc_images, 'r') as f:
            gamma_data = f['Gamma'][gamma_offset:(gamma_offset + gamma_batch_size), :, :, :]
            hadron_data = f['Hadron'][hadron_offset:(hadron_offset + hadron_batch_size), :, :, :]

        batch_data = np.concatenate((gamma_data, hadron_data), axis=0)
        labels = np.array([True]*gamma_batch_size+[False]*hadron_batch_size)
        batch_labels = (np.arange(2) == labels[:,None]).astype(np.float32)

        yield batch_data, batch_labels

def getValidationTesting(path_mc_images, events_in_validation_and_testing, gamma_anteil, hadron_anteil, gamma_count, hadron_count):
    with h5py.File(path_mc_images, 'r') as f:
        gamma_size = int(round(events_in_validation_and_testing*gamma_anteil))
        hadron_size = int(round(events_in_validation_and_testing*hadron_anteil))

        gamma_valid_data = f['Gamma'][gamma_count:(gamma_count+gamma_size), :, :, :]
        hadron_valid_data = f['Hadron'][hadron_count:(hadron_count+hadron_size), :, :, :]

        valid_dataset = np.concatenate((gamma_valid_data, hadron_valid_data), axis=0)
        labels = np.array([True]*gamma_size+[False]*hadron_size)
        valid_labels = (np.arange(2) == labels[:,None]).astype(np.float32)


        gamma_test_data = f['Gamma'][(gamma_count+gamma_size):(gamma_count+2*gamma_size), :, :, :]
        hadron_test_data = f['Hadron'][(hadron_count+hadron_size):(hadron_count+2*hadron_size), :, :, :]

        test_dataset = np.concatenate((gamma_test_data, hadron_test_data), axis=0)
        labels = np.array([True]*gamma_size+[False]*hadron_size)
        test_labels = (np.arange(2) == labels[:,None]).astype(np.float32)

    return valid_dataset, valid_labels, test_dataset, test_labels
'''

batch_size = 10000

params={
    'lr': 0.001,
    'conv_dr': 0.,
    'fc_dr': 0.1,
    'batch_size': 128,
    'no_epochs': 1000,
    'steps_per_epoch': 100,
    'dp_prob': 0.5,
    'batch_norm': False,
    'regularize': 0.0,
    'decay': 0.0
}

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allocator_type='BFC'))

sess = tf.Session(config=config)

K.set_session(sess)

#Define model
model = Sequential()

# Set up regulariser
regularizer = l2(0.0)

model.add(Conv3D(64, kernel_size=[3,3,3], strides=(1,1,1), padding='SAME', input_shape=[1, 40, 56, 40], activation='relu'))
#model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
#                       border_mode='valid', name='pool1'))
model.add(Conv3D(128, kernel_size=[3,3,3], strides=(1,1,1), padding='SAME', activation='relu'))
#model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
#                       border_mode='valid', name='pool1'))
model.add(Conv3D(256, kernel_size=[3,3,3], strides=(1,1,1), padding='SAME',  activation='relu'))
model.add(Conv3D(256, kernel_size=[3,3,3], strides=(1,1,1), padding='SAME', activation='relu'))
#model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
#                       border_mode='valid', name='pool1'))
model.add(Conv3D(512, kernel_size=[3,3,3], strides=(1,1,1), padding='SAME', activation='relu'))
model.add(Conv3D(512, kernel_size=[3,3,3], strides=(1,1,1), padding='SAME', activation='relu'))
#model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
#                       border_mode='valid', name='pool1'))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(AlphaDropout(0.1))
model.add(Dense(4096, activation='relu'))
model.add(AlphaDropout(0.1))
model.add(Dense(4096, activation='relu'))
model.add(AlphaDropout(0.1))
model.add(Dense(4096, activation='relu'))
model.add(AlphaDropout(0.1))
model.add(Dense(2, activation='softmax'))
print(model.summary())