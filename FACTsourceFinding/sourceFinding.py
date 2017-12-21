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

batch_size = 128
num_classes = 10
epochs = 12

# Image dimensions
img_rows, img_cols = 28, 28



model = Sequential()

def gamma_ray_loss_function():
    def weight_loss(y_true, y_pred):
        return K.mean( (y_true - y_pred) / K.sqrt(y_true - y_pred))
    return weight_loss

