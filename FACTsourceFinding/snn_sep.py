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
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, AlphaDropout, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera
import pandas as pd

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

# Hyperparameters

batch_sizes = [16, 64, 256]
patch_sizes = [(2, 2), (3, 3), (5, 5), (4, 4)]
dropout_layers = [0.0, 1.0]
num_conv_layers = [0, 6]
num_dense_layers = [0, 6]
num_conv_neurons = [8,128]
num_dense_neuron = [8,256]
num_pooling_layers = [0, 2]
num_runs = 500
number_of_training = 120000*(0.6)
number_of_testing = 120000*(0.2)
number_validate = 120000*(0.2)
optimizers = ['same']
epoch = 500

path_mc_images = "/run/media/jacob/SSD/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_proton_images = "/run/media/jacob/SSD/Rebinned_5_MC_Proton_JustImage_Images.h5"
np.random.seed(0)

def metaYielder():
    with h5py.File(path_mc_images, 'r') as f:
        gam = len(f['Image'])
        with h5py.File(path_proton_images, 'r') as f2:
            had = len(f2['Image'])
            sumEvt = gam + had
            print(sumEvt)

    gamma_anteil = gam / sumEvt
    hadron_anteil = had / sumEvt

    gamma_count = int(round(number_of_training * gamma_anteil))
    hadron_count = int(round(number_of_training * hadron_anteil))

    gamma_anteil = 0.5
    hadron_anteil = 0.5

    return gamma_anteil, hadron_anteil, gamma_count, hadron_count


with h5py.File(path_mc_images, 'r') as f:
    with h5py.File(path_proton_images, 'r') as f2:
        gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder()
        # Get some truth data for now, just use Crab images
        items = len(f2["Image"])
        images = f['Image'][0:40]
        images_false = f2['Image'][0:40]
        validating_dataset = np.concatenate([images, images_false], axis=0)
        #Normalize each image
        transformed_images = []
        for image_one in validating_dataset:
            #print(image_one.shape)
            image_one = image_one/np.sum(image_one)
            #print(np.sum(image_one))
            transformed_images.append(image_one)
            #print(np.max(image_one))
        validating_dataset = np.asarray(transformed_images)
        #print(validating_dataset.shape)
        labels = np.array([True] * (len(images)) + [False] * len(images_false))
        np.random.seed(0)
        rng_state = np.random.get_state()
        np.random.shuffle(validating_dataset)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        validating_dataset = validating_dataset[0:int(0.5*len(validating_dataset))]
        labels = labels[0:int(0.5*len(labels))]
        #print(ind)
        #print(counts)
        #rng_state = np.random.get_state()
        #np.random.set_state(rng_state)
        #np.random.shuffle(validating_dataset)
        #np.random.set_state(rng_state)
        #np.random.shuffle(labels)
        #del images
        #del images_false
        validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
        y = validating_dataset
        y_label = validation_labels
        print("Finished getting data")


model_base = base_dir + "/Models/RealFinalSep/"
model_name = "MC_SMM_b" + str(32) + "_p_"


csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=90, min_lr=0.001)
model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "{val_acc:.3f}_" + model_name + ".h5",
                                                   monitor='val_loss',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto', period=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                           patience=100,
                                           verbose=0, mode='auto')

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(75,75,1),kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Activation('selu'))
model.add(Conv2D(32, (3, 3),kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(AlphaDropout(0.1))

model.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Activation('selu'))
model.add(Conv2D(64, (3, 3),kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(AlphaDropout(0.40))

model.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Activation('selu'))
model.add(Conv2D(64, (3, 3),kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(AlphaDropout(0.25))

model.add(Flatten())
model.add(Dense(512,kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Activation('selu'))
model.add(AlphaDropout(0.2))
model.add(Dense(512,kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Activation('selu'))
model.add(AlphaDropout(0.2))
model.add(Dense(512,kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Activation('selu'))
model.add(AlphaDropout(0.2))
model.add(Dense(2,kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

model.fit(x=y, y=y_label, batch_size=10, epochs=epoch, shuffle=True, validation_split=0.2, callbacks=[early_stop, csv_logger, model_checkpoint])

