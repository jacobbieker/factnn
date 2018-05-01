from keras import backend as K
import h5py
from fact.io import read_h5py
import yaml
import os
import keras
import numpy as np
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from hyperopt import Trials, STATUS_OK, tpe, hp
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
from hyperas import optim
from hyperas.distributions import uniform, choice, conditional, randint


def data():
    '''
    Data providing function
    :return:
    '''
    architecture = 'manjaro'

    if architecture == 'manjaro':
        base_dir = '/run/media/jacob/WDRed8Tb1'
        thesis_base = '/run/media/jacob/SSD/Development/thesis'
    else:
        base_dir = '/projects/sventeklab/jbieker'
        thesis_base = base_dir + '/git-thesis/thesis'

    number_of_training = 100000 * (0.6)
    number_of_testing = 100000 * (0.2)
    number_validate = 100000 * (0.2)
    num_labels = 2

    # Total fraction to use per epoch of training data, need inverse
    frac_per_epoch = 5
    num_epochs = 100*frac_per_epoch

    path_mc_images = base_dir + "/FACTSources/Rebinned_5_MC_Phi_Images.h5"

    np.random.seed(0)

    def metaYielder():
        with h5py.File(path_mc_images, 'r') as f:
            gam = len(f['GammaImage'])
            had = len(f['Image'])
            sumEvt = gam + had

            gamma_anteil = gam / sumEvt
            hadron_anteil = had / sumEvt

            gamma_count = int(round(number_of_training * gamma_anteil))
            hadron_count = int(round(number_of_training * hadron_anteil))

            return gamma_anteil, hadron_anteil, gamma_count, hadron_count


    with h5py.File(path_mc_images, 'r') as f:
        gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder()
        # Get some truth data for now, just use Crab images
        images = f['GammaImage'][-int(np.floor((gamma_anteil * number_of_testing))):-1]
        images_false = f['Image'][-int(np.floor((hadron_anteil * number_of_testing))):-1]
        validating_dataset = np.concatenate([images, images_false], axis=0)
        labels = np.array([True] * (len(images)) + [False] * len(images_false))
        del images
        del images_false
        validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
        y = validating_dataset
        y_labels = validation_labels

        images = f['GammaImage'][0:int(np.floor((gamma_anteil * number_of_training)))]
        images_false = f['Image'][0:int(np.floor((hadron_anteil * number_of_testing)))]
        training_dataset = np.concatenate([images, images_false], axis=0)
        training_labels = np.array([True] * (len(images)) + [False] * len(images_false))
        x_train = training_dataset
        x_labels = training_labels

        print("Finished getting data")
        return x_train, x_labels, y, y_labels


def create_model(x_train, x_labels, y, y_labels):
    '''
    Model ceration function
    :return:
    '''
    num_labels = 2

    model = Sequential()
    # Base Conv layer
    conv_neurons = {{choice(np.arange(8,128, dtype=int))}}
    dense_neurons = {{choice(np.arange(8,256, dtype=int))}}
    patch_size = {{choice([(2,2), (3,3), (5,5)])}}
    dropout_layer = 0.0
    model.add(Conv2D(conv_neurons, kernel_size=patch_size, strides=(1, 1),
                     activation='relu', padding='same',
                     input_shape=(75, 75, 1)))
    for i in range(3):
        model.add(
            Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu',
                   padding='same'))
        if {{choice([True, False])}}:
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        if dropout_layer > 0.0:
            model.add(Dropout(dropout_layer))

    model.add(Flatten())

    # Now do the dense layers
    for i in range(0):
        model.add(Dense(dense_neurons, activation='relu'))
        if dropout_layer > 0.0:
            model.add(Dropout(dropout_layer))

    # Final Dense layer
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss='categorical_crossentropy',
                  metrics=['acc'])

    model.fit(x_train, x_labels,
              batch_size={{choice([16, 128])}},
              epochs=20,
              verbose=2,
              validation_data=(y, y_labels))
    score, acc = model.evaluate(y, y_labels, verbose=0)
    print("Test Acc: ", acc)
    return {"loss": -acc, "status": STATUS_OK, 'model': model}


# Now optimize over them

best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=10,
                                      trials=Trials())

x_train, x_labels, y, y_labels = data()
print("Evaluation of best performing model:")
print(best_model.evaluate(y, y_labels))
print("Best performing model chosen hyper-perameters:")
print(best_run)