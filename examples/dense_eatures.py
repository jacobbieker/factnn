import os
import keras

from factnn.utils.cross_validate import cross_validate

from keras.layers import Flatten, ConvLSTM2D, MaxPooling2D, Dense, Activation, Dropout, MaxPooling3D, Conv2D, Conv3D, \
    AveragePooling2D, AveragePooling3D, PReLU, BatchNormalization, PReLU, Input, Conv1D, ELU
from keras.models import Model
from keras.models import Sequential
import keras
import keras.layers as layers
import numpy as np

from keras_applications.xception import Xception
import keras.backend as K

# from ..gradient_checkpointing.memory_saving_gradients import gradients_memory

# K.__dict__["gradients"] = gradients_memory


directory = "/home/jacob/Documents/iact_events/"
gamma_dir = [directory + "gammaFeature/clump20/"]
proton_dir = [directory + "protonFeature/clump20/"]

# Inputs

time_input = Input(shape=(10, 75, 75, 1), name='time_input')
feature_input = Input(shape=(8,), name='feature_input')

# Feature Layers
feature_layer = Dense(512, activation='relu')(feature_input)
feature_layer = Dropout(0.2)(feature_layer)
feature_layer = Dense(1024, activation='relu')(feature_layer)
feature_layer = Dropout(0.2)(feature_layer)
feature_layer = Dense(2048, activation='relu')(feature_layer)
feature_layer = Dropout(0.2)(feature_layer)
feature_layer = Dense(2048, activation='relu')(feature_layer)
feature_layer = Dropout(0.2)(feature_layer)
feature_output = Dense(2, activation='softmax', name='feature_output')(feature_layer)

# Time Layers

time_layer_one = ConvLSTM2D(32, kernel_size=3, strides=1,
                            padding='same',
                            activation='tanh', dropout=0.3,
                            recurrent_dropout=0.3, recurrent_activation='hard_sigmoid',
                            return_sequences=True)(time_input)
time_layer_one = MaxPooling3D()(time_layer_one)

time_layer_two = ConvLSTM2D(32, kernel_size=3, strides=1,
                            padding='same',
                            activation='tanh', dropout=0.3,
                            recurrent_dropout=0.3, recurrent_activation='hard_sigmoid',
                            return_sequences=True)(time_layer_one)
time_layer_two = MaxPooling3D()(time_layer_two)

time_layer_two = ConvLSTM2D(32, kernel_size=3, strides=1,
                            padding='same',
                            activation='tanh', dropout=0.3,
                            recurrent_dropout=0.3, recurrent_activation='hard_sigmoid',
                            return_sequences=False)(time_layer_two)
time_layer_two_pool = MaxPooling2D()(time_layer_two)

time_flattened = Flatten()(time_layer_two_pool)

# Merge Layers and add Dense

merged_layers = keras.layers.concatenate([time_flattened, feature_layer])

dense_layer = Dense(64)(merged_layers)
dense_layer = Activation(ELU())(dense_layer)

dense_layer = Dense(64)(dense_layer)
dense_layer = Activation(ELU())(dense_layer)

dense_layer = Dense(64)(dense_layer)
dense_layer = Activation(ELU())(dense_layer)

main_output = Dense(2, activation='softmax', name='final_output')(dense_layer)

model = Model(inputs=[time_input, feature_input],
              outputs=[main_output, feature_output])

# Compile and assign weights to different outputs

model.compile(optimizer='adam', loss='categorical_crossentropy', loss_weights=[0.5, 1.0], metrics=['acc'])

model.summary()

results = cross_validate(model, gamma_dir, proton_directory=proton_dir, indicies=(40, 130, 10), rebin=75,
                         as_channels=False, kfolds=5, model_type="Separation", normalize=False, batch_size=16,
                         workers=12, verbose=1, truncate=True, dynamic_resize=True, equal_slices=False,
                         return_features=True, return_collapsed=False, plot=False)
