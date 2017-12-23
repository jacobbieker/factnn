import numpy as np
import argparse

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, BatchNormalization
from keras.utils import to_categorical, Sequence
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from fact.analysis import split_on_off_source_independent, split_on_off_source_dependent
from fact.io import read_h5py
import h5py
import pandas as pd

batch_size = 128
num_classes = 10
epochs = 12

# Image dimensions
img_rows, img_cols = 28, 28




model = Sequential()

def gamma_ray_loss_function():
    def weight_loss(y_true, y_pred):
        return K.mean( ((y_true - y_pred) / K.sqrt(y_true - y_pred)), axis=-1)
    return weight_loss


crab_sample = read_h5py("/home/jacob/Development/thesis/open_crab_sample_facttools_dl2.hdf5", "events")
# print(crab_sample)
on_crab_events, off_crab_events = split_on_off_source_independent(crab_sample, theta2_cut=0.8)
print(len(on_crab_events))
print(len(off_crab_events))

