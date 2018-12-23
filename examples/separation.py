import os
import keras

from factnn.utils.cross_validate import cross_validate

from keras.layers import Flatten, ConvLSTM2D, MaxPooling2D, Dense, Activation, Dropout, MaxPooling3D, Conv2D, Conv3D, \
    AveragePooling2D, AveragePooling3D, PReLU, BatchNormalization, PReLU
from keras.models import Sequential
import keras
import keras.layers as layers
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
import numpy as np

directory = "/home/jacob/Documents/iact_events/"
gamma_dir = [directory + "gamma/core20/"]
proton_dir = [directory + "proton/core20/"]

#model = ResNet50(include_top=True, weights=None, input_shape=(50,50,3), pooling=None, classes=2)

#model = DenseNet201(include_top=True, weights=None, input_shape=(50,50,3), pooling=None, classes=2)

#model = Xception(include_top=True, weights=None, input_shape=(75,75,1), pooling=None, classes=2)

model = Sequential()

model.add(Conv2D(64, kernel_size=3, strides=1,
                 padding='same',
                 input_shape=(50,50,1)))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.1))
#model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=3, strides=1,
                 padding='same',
                 ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=3, strides=1,
                 padding='same',
                 ))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.1))
#model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=3, strides=1,
                 padding='same',
                 ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size=3, strides=1,
                 padding='same',
                 ))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size=3, strides=1,
                 padding='same',
                 ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(256))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()

results = cross_validate(model, gamma_dir, proton_directory=proton_dir, indicies=(40, 130, 3), rebin=50,
                         as_channels=True, kfolds=5, model_type="Separation", normalize=False, batch_size=64,
                         workers=10, verbose=1, truncate=True, dynamic_resize=True, equal_slices=False, plot=False)

