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

from fact.analysis.binning import bin_runs
from fact.io import read_h5py, to_h5py
import h5py
import keras
import pandas as pd
from FACTsourceFinding.fact_generators import SimDataGenerator, DataGenerator

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

source_file = "/run/media/jacob/WDRed8Tb1/FACTSources/Crab_preprocessed_images.h5"
sim_source_file = "/run/media/jacob//WDRed8Tb1/FACTSources/MC_2D_Images.h5"
length_data = 0
#with h5py.File(source_file, 'r') as hdf:
#    length_data = len(hdf['Night'])

#with h5py.File(sim_source_file, 'r') as hdf:
#    sim_length_data = len(hdf['Hadron']) # Hadron has less event, so this ensures it won't go out of bounds on the hadron events, need better way to do this

'''
def batchYielder(path_mc_images):
    gamma_batch_size = int(round(batch_size))
    hadron_batch_size = int(round(batch_size))

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




def getValidationTesting(path_mc_images):
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

def Dategenerator(path_to_truth_images_1, path_to_truth_images_2):
    with h5py.File(path_to_truth_images_1, 'r') as f:
        with h5py.File(path_to_truth_images_2, 'r') as f_1:
            # Get some truth data for now, just use Crab images
            items = list(f.items())[0][1].shape[0]
            i = 0

            while (i+1)*batch_size < items/160:
                images = f['Image'][ i*(batch_size/2):(i+1)*(batch_size/2) ]
                images_false = f_1['Image'][ i*(batch_size/2):(i+1)*(batch_size/2) ]
                test_dataset = np.concatenate((images, images_false), axis=0)
                labels = np.array([True]*(len(images))+[False]*len(images_false))
                test_labels = (np.arange(2) == labels[:,None]).astype(np.float32)
                i += 1
            while items/160 < (i+1)*batch_size < 2*items/160:
                images = f['Image'][ i*(batch_size/2):(i+1)*(batch_size/2) ]
                images_false = f_1['Image'][ i*(batch_size/2):(i+1)*(batch_size/2) ]
                validating_dataset = np.concatenate((images, images_false), axis=0)
                labels = np.array([True]*(len(images))+[False]*len(images_false))
                validation_labels = (np.arange(2) == labels[:,None]).astype(np.float32)
                i += 1

    return validating_dataset, validation_labels, test_dataset, test_labels

params = {'dim': (64),
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
                 input_shape=(46,45,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
#model.add(Dense(1000, activation='relu', input_shape=(64)))
#model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(num_labels, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

x, x_label, y, y_label = Dategenerator("/run/media/jacob/WDRed8Tb1/FACTSources/Mrk 421_preprocessed_images.h5", "/run/media/jacob/WDRed8Tb1/FACTSources/Crab_preprocessed_images.h5")

model.fit(x=x, y=x_label, batch_size=32, epochs=300, verbose=2, validation_split=0.2, shuffle='batch')

model.save("Test_training_model.h5")

#model.fit_generator(generator=training_generator,
#                    validation_data=validating_generator,
#                    use_multiprocessing=False)

#model.save("crab_trained_model.h5")

# Now do it with the simulated data

#params = {'dim': (46,45),
#          'batch_size': 64,
#          'n_classes': 2,
#          'n_channels': 1,
#          'shuffle': True,
#          'source_file': sim_source_file,
#          'length_dataset': sim_length_data,
#          }

#training_generator = SimDataGenerator(**params)
#validating_generator = SimDataGenerator(**params)

#model.fit_generator(generator=training_generator,
#                    validation_data=validating_generator,
#                    use_multiprocessing=True)

#model.save("sim_trained_model.h5")

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
