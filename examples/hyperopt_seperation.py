import os
import keras

from factnn.utils.cross_validate import cross_validate, data

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

from keras.layers import Dense, Dropout, Flatten, ConvLSTM2D, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, PReLU, \
    BatchNormalization, ReLU
from keras.models import Sequential
import keras
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from hyperas import optim
from hyperas.distributions import choice, uniform, choice

def create_model(x_train, y_train, x_test, y_test):
    separation_model = Sequential()
    # separation_model.add(BatchNormalization())
    separation_model.add(
        Conv2D({{choice([16, 32, 64])}},
               input_shape=[75, 75, 5],
               kernel_size=3,
               strides=1,
               padding='same'))
    separation_model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    separation_model.add(MaxPooling2D())
    separation_model.add(Dropout({{uniform(0, 1)}}))
    # separation_model.add(BatchNormalization())
    separation_model.add(Conv2D({{choice([16, 32, 64])}},
                                kernel_size=3,
                                strides=1,
                                padding='same'))
    separation_model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    separation_model.add(MaxPooling2D())
    separation_model.add(Dropout({{uniform(0, 1)}}))
    separation_model.add(Conv2D({{choice([16, 32, 64])}},
                                kernel_size=3,
                                strides=1,
                                padding='same'))
    separation_model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    separation_model.add(MaxPooling2D())
    if {{choice(['three', 'four'])}} == 'four':
        separation_model.add(Conv2D({{choice([16, 32, 64, 128])}},
                                    kernel_size=3,
                                    strides=1,
                                    padding='same'))
        separation_model.add(Activation({{choice(['relu', 'sigmoid'])}}))
        separation_model.add(MaxPooling2D())
        # separation_model.add(BatchNormalization())
        separation_model.add(Dropout({{uniform(0, 1)}}))
    separation_model.add(Flatten())
    separation_model.add(Dense({{choice([16, 32, 64])}}))
    separation_model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    separation_model.add(Dropout({{uniform(0, 1)}}))
    separation_model.add(Dense({{choice([16, 32, 64, 128])}}))
    separation_model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    separation_model.add(Dropout({{uniform(0, 1)}}))
    if {{choice(['three', 'four'])}} == 'four':
        separation_model.add(Dense({{choice([16, 32, 64, 128])}}))
        separation_model.add(Activation({{choice(['relu', 'sigmoid'])}}))
        separation_model.add(Dropout({{uniform(0, 1)}}))
    # For separation
    separation_model.add(Dense(2, activation='softmax'))
    separation_model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss='categorical_crossentropy',
                             metrics=['acc'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                               patience=5,
                                               verbose=0, mode='auto')
    result = separation_model.fit(x_train, y_train,
                                  batch_size={{choice([8, 16, 32, 64])}},
                                  epochs=200,
                                  verbose=2,
                                  validation_split=0.2,
                                  callbacks=[early_stop])
    # get the highest validation accuracy of the training epochs
    validation_acc = np.amin(result.history['val_loss'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': validation_acc, 'status': STATUS_OK, 'model': separation_model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials())
    best_model.summary()
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save("hyperas_seperation_best.hdf5")

results = cross_validate(model, gamma_dir, proton_directory=proton_dir, indicies=(40, 80, 1), rebin=75,
                         as_channels=True, kfolds=5, model_type="Separation", normalize=True, batch_size=32,
                         workers=6, verbose=1, truncate=True, dynamic_resize=True, equal_slices=False, plot=False)