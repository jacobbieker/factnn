import numpy as np
import argparse
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Flatten, Reshape
from keras.layers.noise import AlphaDropout
from keras.utils import to_categorical, Sequence
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import tensorflow as tf
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

from thesisTools import kerasTools

path_mc_images = "/tree/tf/00_MC_Diffuse_flat_Images.h5"
save_model_path = "/notebooks/thesis/jan/hyperModels/"

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

model.add(Dense(64, input_shape=(46,45,1), activation='selu', kernel_regularizer=regularizer))
model.add(AlphaDropout(0.1))
model.add(Dense(64, activation='selu', kernel_regularizer=regularizer))
model.add(AlphaDropout(0.1))
model.add(Dense(64, activation='selu', kernel_regularizer=regularizer))
model.add(AlphaDropout(0.1))
model.add(Dense(64, activation='selu', kernel_regularizer=regularizer))
model.add(AlphaDropout(0.1))
model.add(Dense(64, activation='selu', kernel_regularizer=regularizer))
model.add(AlphaDropout(0.1))
model.add(Dense(64, activation='selu', kernel_regularizer=regularizer))
model.add(AlphaDropout(0.1))
model.add(Dense(32, activation='selu', kernel_regularizer=regularizer))
model.add(AlphaDropout(0.1))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# Set up optimizer
optimizer = Adam(lr=0.001)

#Create Model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', kerasTools.precision, kerasTools.recall, kerasTools.f1, kerasTools.class_balance]
)

print(model.output_shape)
print(model.summary())

# Need to load data


dropout_rate_c = 0.9
dropout_rate_c_output = 0.75
dropout_rate_f = 0.5

learning_rate = 0.001

pretraining_steps = [5001, 2001, 8001, 1001, 1]

# Number of events in training-dataset
num_events = 800000

# Number of events in validation-/test-dataset
events_in_validation_and_testing = 5000

# Number of nets to compute
number_of_nets = 30

trainable = True

# Architectures to test
test_architectures = ['ccccccffff']

num_labels = 2 # gamma or proton
num_channels = 1 # it is a greyscale image

num_steps = 25001



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

csv_path = '/run/media/jbieker/WDRed8Tb2/pretraining.csv'
pickle_path = '/run/media/jbieker/WDRed8Tb2/Pickle_{}.p'
if 'pretraining.csv' not in os.listdir('/run/media/jbieker/WDRed8Tb2/'):
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Accuracy', 'Auc', 'Pretraining'])



def training(steps, best_auc):
    #TODO Train the SNN
    gen = batchYielder(path_mc_images)
    for step in range(steps):
        batch_data, batch_labels = next(gen)
        # Creating a feed_dict to train the model on in this step
        #feed_dict = {'tf_train_dataset': batch_data, 'tf_train_labels' : batch_labels}
        # Train the model for this step
        #_ = sess.run([optimizer], feed_dict=feed_dict)
        model.fit(x=batch_data, y=batch_labels, batch_size=batch_size, epochs=25000, verbose=2)

        # Updating the output to stay in touch with the training process
        # Checking for early-stopping with scikit-learn
        if (step % 500 == 0):
            #s = sess.run(summ, feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels})
            #writer.add_summary(s, step)

            # Compute the accuracy and the roc-auc-score with scikit-learn
            #pred = sess.run(valid_prediction)
            pred = np.array(list(zip(pred[:,0], pred[:,1])))
            stop_acc = accuracy_score(np.argmax(valid_labels, axis=1), np.argmax(pred, axis=1))
            stop_auc = roc_auc_score(valid_labels, pred)

            # Check if early-stopping is necessary
            auc_now = stop_auc
            if step == 0:
                stopping_auc = 0.0
                sink_count = 0
            else:
                if auc_now > stopping_auc:
                    stopping_auc = auc_now
                    sink_count = 0
                    # Check if the model is better than the existing one and has to be saved
                    if stopping_auc > best_auc:
                        best_auc = stopping_auc
                else:
                    sink_count += 1

            # Printing a current evaluation of the model
            print('St_auc: {}, sc: {},val: {}, Step: {}'.format(stopping_auc, sink_count, stop_acc*100, step))
            if sink_count == 10:
                break

    return stop_acc, stopping_auc, step, best_auc


gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder(path_mc_images)
valid_dataset, valid_labels, test_dataset, test_labels = getValidationTesting(path_mc_images, events_in_validation_and_testing, gamma_anteil, hadron_anteil, gamma_count, hadron_count)
training(num_steps, 0.0)
