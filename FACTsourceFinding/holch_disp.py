from keras import backend as K
import h5py
from fact.io import read_h5py
import yaml
import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D

architecture = 'intel'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

# Hyperparameters

batch_sizes = [4, 8, 16, 32, 64, 128, 256]
gamma_trains = [1, 2, 3, 4, 5]
patch_sizes = [(3, 3), (5, 5)]
dropout_layers = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_conv_neurons = [8, 16, 32, 64, 128]
num_dense_neuron = [64, 128, 256, 512, 1024]
num_pooling_layers = [0,1]
number_of_training = 640000
number_of_testing = 10000

