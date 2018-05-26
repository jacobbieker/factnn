import os
# to force on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pickle
from keras import backend as K
import h5py
from fact.io import read_h5py, read_h5py_chunked
import yaml
import tensorflow as tf
import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, ELU, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera
import pandas as pd
from sklearn.metrics import roc_auc_score

