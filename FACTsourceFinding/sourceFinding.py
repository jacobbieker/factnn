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
import keras
import pandas as pd

batch_size = 128
num_classes = 10
epochs = 12

# Image dimensions
img_rows, img_cols = 28, 28

num_steps = 25001  # Max batches for model

min_batch_image_size = 500  # How many images will be in a combined image
max_batch_image_size = 5000

min_batch_size = 64  # Number of images in a batch
max_batch_size = 257

patch_size = [3, 5]  # 3x3 or 5x5
num_channels = 1  # Only number of photons matter, so greyscale
num_labels = 2  # Source or not source


# Make a batch generator to generate the combined image batches to feed into the Keras model

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, length_dataset, source_file, image_size=5000, seed=0, batch_size=32, dim=(46, 45), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.length_dataset = length_dataset
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.source_file = source_file
        self.seed = seed
        self.image_size = image_size
        self.batch_index = 1 # Start
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.length_dataset / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        with h5py.File(self.source_file, 'r') as hdf:
            items = list(hdf['Night'])
            self.indexes = np.arange(len(items))
            np.random.seed(self.seed)
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def batchYielder(self, list_indexes):
        # Generate the batches
        batch_images = []
        batch_triggers = []
        with h5py.File(self.source_file, 'r') as hdf:
            i = 0
            j = 0 # The index of the image size
            k = 0 # Batch size
            while k < self.batch_size:
                nights = []
                runs = []
                events = []
                images = []
                triggers = []
                for current_index in list_indexes[j:j+self.image_size]:
                    night = hdf['Night'][current_index]
                    run = hdf['Run'][current_index]
                    event = hdf['Event'][current_index]
                    image = hdf['Image'][current_index]
                    trigger = hdf['Trigger'][current_index]
                    nights.append(night)
                    runs.append(run)
                    events.append(event)
                    images.append(image)
                    triggers.append(trigger)
                    i += 1
                    print(i)
                #nights = np.asarray(nights)
                #runs = np.asarray(runs)
                #events = np.asarray(events)
                images = np.asarray(images)
                triggers = np.asarray(triggers)
                images = np.sum(images, axis=0)
                batch_images.append(images)
                trigger_class = 0
                if np.bincount(triggers).argmax() == 4:
                    trigger_class = 1

                batch_triggers.append(trigger_class)
                i = 0 # Reset to 0 after getting to image size, like 5000
                j += self.image_size # Increase so chooses next set of it
                self.batch_index += self.batch_size # Add batch size to keep track of where I am in the self.indexes
                k += 1
            batch_images = np.asarray(batch_images)
            batch_triggers = np.asarray(batch_triggers)
            yield (batch_images, batch_triggers)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i in range(0, self.batch_size):
            # Store sample
            image_gen = self.batchYielder(list_IDs_temp)
            next_image = next(image_gen)
            X[i,] = next_image[0]
            # Store class
            y[i] = next_image[1]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


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


def li_ma_significance_loss(on_events, off_events):
    # Derivative of Li and Ma with respect to N_on and negative version to go to lower
    on_events = np.asarray(on_events)
    off_events = np.asarray(off_events)
    return -K.log((2 * on_events) / (on_events + off_events)) / (K.sqrt(2) * K.sqrt(
        on_events * K.log((2 * on_events) / (on_events + off_events)) + off_events * K.log(
            (2 * off_events) / (on_events + off_events))))


def lima_sig_loss(on_events, off_events):
    return -1 * K.sqrt(2) * K.sqrt(on_events * K.log((on_events) / (on_events + off_events)) + off_events * K.log(
        off_events / (on_events + off_events)))


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

# crab_df = read_h5py("/run/media/jacob/WDRed8Tb1/00_crab1314_preprocessed_images.h5", key='events')
# gamma_df = read_h5py("/run/media/jacob/WDRed8Tb1/MC_2D_Images.h5", key="events")

# print(gamma_df.columns.values)

source_file = "/run/media/jacob/WDRed8Tb1/Crab_preprocessed_images.h5"
length_data = 0
with h5py.File(source_file, 'r') as hdf:
    length_data = len(hdf['Night'])

params = {'dim': (46,45),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True,
          'source_file': source_file,
          'length_dataset': length_data,
          }

training_generator = DataGenerator(**params)
validating_generator = DataGenerator(**params)


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
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit_generator(generator=training_generator,
                    validation_data=validating_generator,
                    use_multiprocessing=True)

'''
crab_sample = read_h5py("/run/media/jacob/SSD/Development/open_crab_sample_analysis/dl2/crab.hdf5", "events")
sim_sample = read_h5py("/run/media/jacob/SSD/Development/open_crab_sample_analysis/dl2/gamma.hdf5", "events")
proton_sample = read_h5py("/run/media/jacob/SSD/Development/open_crab_sample_analysis/dl2/proton.hdf5", "events")
gamma_diffuse_sample = read_h5py("/run/media/jacob/SSD/Development/open_crab_sample_analysis/dl2/gamma_diffuse.hdf5", "events")


sim_sample['is_gamma'] = 1
proton_sample['is_gamma'] = 0
gamma_diffuse_sample['is_gamma'] = 2

combined_proton_gamma = pd.concat([sim_sample, proton_sample], ignore_index=True)

combined_all_sim = pd.concat([sim_sample, proton_sample, gamma_diffuse_sample], ignore_index=True)

crab_proton_diffuse = pd.concat([crab_sample, proton_sample, gamma_diffuse_sample], ignore_index=True)

crab_proton = pd.concat([crab_sample, proton_sample], ignore_index=True)

crab_gamma = pd.concat([crab_sample, sim_sample], ignore_index=True)


to_h5py(filename="/run/media/jacob/SSD/Development/open_crab_sample_analysis/dl2/crab_proton_diffuse.hdf5", df=crab_proton_diffuse, key="events")
to_h5py(filename="/run/media/jacob/SSD/Development/open_crab_sample_analysis/dl2/crab_gamma.hdf5", df=crab_gamma, key="events")
to_h5py(filename="/run/media/jacob/SSD/Development/open_crab_sample_analysis/dl2/crab_proton.hdf5", df=crab_proton, key="events")

to_h5py(filename="/run/media/jacob/SSD/Development/open_crab_sample_analysis/dl2/combine_proton_gamma.hdf5", df=combined_proton_gamma, key="events")
to_h5py(filename="/run/media/jacob/SSD/Development/open_crab_sample_analysis/dl2/combine_all_sim.hdf5", df=combined_all_sim, key="events")

# For loop to produce the different cuts


start_point = 0.05
end_point = 1.0
number_steps = 10
step_size = 0.05
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

    while start_point + (step_size * j) <= end_point:
        theta_value_two = start_point +(step_size * j)

        # now go through, getting all the necessary permutations to use to compare to the default

        j += 1

    # remove DF of splits so it doesn't run out of memory
    del on_crab_events, off_crab_events
    del on_sim_events, off_sim_events
    del on_sim_all_events, off_sim_all_events
    del on_sim_proton_events, off_sim_proton_events

    i += 1  # Add one to increment
'''
# print(crab_sample)

# binned_sim_events = bin_runs(crab_sample)
# print(binned_sim_events)
# Write separated events
