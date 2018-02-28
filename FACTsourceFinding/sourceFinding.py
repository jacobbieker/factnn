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
        return K.mean(((y_true - y_pred) / K.sqrt(y_true - y_pred)), axis=-1)

    return weight_loss


def basic_loss_function(y_true, y_pred):
    # Divide one by the square root of the other to get rough idea of significance before using li-ma
    return y_pred / np.sqrt(y_true)


def on_off_loss(on_events, off_events):
    # Simpler one is to take the on_events and divide bt 1/5 the number of off events, since 5 off regions to 1 on
    return on_events / np.sqrt(off_events * 0.2)


crab_sample = read_h5py("../dl2/crab.hdf5", "events")
sim_sample = read_h5py("../dl2/gamma.hdf5", "events")
proton_sample = read_h5py("../dl2/proton.hdf5", "events")
gamma_diffuse_sample = read_h5py("../dl2/gamma_diffuse.hdf5", "events")


sim_sample['is_gamma'] = 1
proton_sample['is_gamma'] = 0
gamma_diffuse_sample['is_gamma'] = 2

print(isinstance(sim_sample, pd.DataFrame))

combined_proton_gamma = pd.concat([sim_sample, proton_sample], ignore_index=True)

print(isinstance(combined_proton_gamma, pd.DataFrame))

combined_all_sim = pd.concat([sim_sample, proton_sample, gamma_diffuse_sample], ignore_index=True)

#to_h5py(filename="../combine_proton_gamma.hdf5", df=combined_proton_gamma, key="events")
#to_h5py(filename="../combine_all_sim.hdf5", df=combined_all_sim, key="events")

print("Finished combined")

# For loop to produce the different cuts

start_point = 0.7
end_point = 1.0
number_steps = 10
step_size = 0.1
i = 3
j = 1

while start_point + (step_size * i) <= end_point:
    theta_value_one = start_point + (step_size * i)
    # Get the initial equal splits of data
    on_crab_events, off_crab_events = split_on_off_source_independent(crab_sample, theta2_cut=theta_value_one)
    on_sim_events, off_sim_events = split_on_off_source_independent(sim_sample, theta2_cut=theta_value_one)
    on_sim_all_events, off_sim_all_events = split_on_off_source_independent(combined_all_sim, theta2_cut=theta_value_one)
    on_sim_proton_events, off_sim_proton_events = split_on_off_source_independent(combined_proton_gamma, theta2_cut=theta_value_one)

    to_h5py(filename="/run/media/jacob/WDRed8Tb1/dl2_theta/on_train_crab_events_facttools_dl2_theta" + str(np.round(theta_value_one, decimals=1)) + ".hdf5", df=on_crab_events,
            key="events")
    to_h5py(filename="/run/media/jacob/WDRed8Tb1/dl2_theta/off_train_crab_events_facttools_dl2_theta" + str(np.round(theta_value_one, decimals=1)) + ".hdf5",
            df=off_crab_events, key="events")
    to_h5py(filename="/run/media/jacob/WDRed8Tb1/dl2_theta/on_train_sim_events_facttools_dl2_theta" + str(np.round(theta_value_one, decimals=1)) + ".hdf5", df=on_sim_events,
            key="events")
    to_h5py(filename="/run/media/jacob/WDRed8Tb1/dl2_theta/off_train_sim_events_facttools_dl2_theta" + str(np.round(theta_value_one, decimals=1)) + ".hdf5", df=off_sim_events,
            key="events")
    to_h5py(filename="/run/media/jacob/WDRed8Tb1/dl2_theta/on_train_sim_all_events_facttools_dl2_theta" + str(np.round(theta_value_one, decimals=1)) + ".hdf5", df=on_sim_all_events,
            key="events")
    to_h5py(filename="/run/media/jacob/WDRed8Tb1/dl2_theta/off_train_sim_all_events_facttools_dl2_theta" + str(np.round(theta_value_one, decimals=1)) + ".hdf5", df=off_sim_all_events,
            key="events")
    to_h5py(filename="/run/media/jacob/WDRed8Tb1/dl2_theta/on_train_sim_proton_events_facttools_dl2_theta" + str(np.round(theta_value_one, decimals=1)) + ".hdf5", df=on_sim_proton_events,
            key="events")
    to_h5py(filename="/run/media/jacob/WDRed8Tb1/dl2_theta/off_train_sim_proton_events_facttools_dl2_theta" + str(np.round(theta_value_one, decimals=1)) + ".hdf5", df=off_sim_proton_events,
            key="events")

    while start_point * j < end_point:
        theta_value_two = start_point * j

        # now go through, getting all the necessary permutations to use to compare to the default

        j += 1

    i += 1  # Add one to increment

# print(crab_sample)

# binned_sim_events = bin_runs(crab_sample)
# print(binned_sim_events)
# Write separated events

print(on_off_loss(len(on_crab_events), len(off_crab_events) * 0.2))
print(on_off_loss(len(off_crab_events) * 0.2, len(off_crab_events) * 0.2))

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
                 input_shape=(46, 45, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
