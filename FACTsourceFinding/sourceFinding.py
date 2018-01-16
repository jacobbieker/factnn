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

from fact.analysis import split_on_off_source_independent, split_on_off_source_dependent
from fact.analysis.binning import bin_runs
from fact.io import read_h5py, to_h5py
import h5py
import pandas as pd

batch_size = 128
num_classes = 10
epochs = 12

# Image dimensions
img_rows, img_cols = 28, 28

def gamma_ray_loss_function():
    def weight_loss(y_true, y_pred):
        return K.mean( ((y_true - y_pred) / K.sqrt(y_true - y_pred)), axis=-1)
    return weight_loss


def basic_loss_function(y_true, y_pred):
    # Divide one by the square root of the other to get rough idea of significance before using li-ma
    return y_pred / np.sqrt(y_true)


def on_off_loss(on_events, off_events):
    # Simpler one is to take the on_events and divide bt 1/5 the number of off events, since 5 off regions to 1 on
    return on_events / np.sqrt(off_events*0.2)


crab_sample = read_h5py("../open_crab__train.hdf5", "events")
sim_sample = read_h5py("../open_sim__train.hdf5", "events")
#proton_sample = read_h5py("../proton_simulations_facttools_dl2.hdf5", "events")
#gamma_diffuse_sample = read_h5py("../gamma_simulations_diffuse_facttools_dl2.hdf5", "events")

# print(crab_sample)
#on_crab_events, off_crab_events = split_on_off_source_independent(crab_sample, theta2_cut=0.1)
on_crab_events, off_crab_events = split_on_off_source_independent(sim_sample, theta2_cut=0.1)

#binned_sim_events = bin_runs(crab_sample)
#print(binned_sim_events)
# Write separated events
#to_h5py("../on_train_crab_events_facttools_dl2_thetaOne.hdf5", df=on_crab_events, key="events")
#to_h5py("../off_train_crab_events_facttools_dl2_thetaOne.hdf5", df=off_crab_events, key="events")
#to_h5py("../on_train_sim_events_facttools_dl2_thetaOne.hdf5", df=on_sim_events, key="events")
#to_h5py("../off_train_sim_events_facttools_dl2_thetaOne.hdf5", df=off_sim_events, key="events")


print(len(on_crab_events))
print(len(off_crab_events)*0.2)

print(on_off_loss(len(on_crab_events), len(off_crab_events)*0.2))
print(on_off_loss(len(off_crab_events)*0.2, len(off_crab_events)*0.2))


# Now simple CNN to minimize the on_off loss

# Input is the flattened Eventlist format, point is to try to find the source in the flattened format
# Once it identifies the central point, that event can be matched back up with the pointing and theta to have the course?
# Use the li-ma or 1/root(n) to find the center, since it has a gradient?
# The flat eventlist gives the intensity from each event, which is what we want
# The intensity from each event can be used to get significance, and then binned,
# giving the different bins used for the significance
# This results in the sharp fall off from theta^2 for the source, hopefully
# Need to calculate loss function batchwise, since sigificance only makes sense in batches
# so have to modify the CNN to do the loss over multiple flattend events at the same time?
# Probably why my test one didn't make much sense
#

num_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(46,45,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
