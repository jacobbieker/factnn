import os
# to force on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
import h5py
from fact.io import read_h5py
import yaml
import os
import keras
import numpy as np
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, LeakyReLU, Reshape, BatchNormalization, Conv2D, MaxPooling2D

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

# Hyperparameters

number_of_training = 200000 * (0.8)
number_of_testing = 200000 * (0.2)
number_validate = 200000 * (0.0)
num_labels = 2

# Total fraction to use per epoch of training data, need inverse
frac_per_epoch = 1
num_epochs = 1000*frac_per_epoch

path_mc_images = "/run/media/jacob/SSD/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_proton_images = "/run/media/jacob/SSD/Rebinned_5_MC_Proton_BothTracking_Images.h5"
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
        images = f['Image'][0:10000]
        images_false = f2['Image'][0:10000]
        validating_dataset = np.concatenate([images, images_false], axis=0)
        #print(validating_dataset.shape)
        labels = np.array([True] * (len(images)) + [False] * len(images_false))
        np.random.seed(0)
        #rng_state = np.random.get_state()
        #np.random.shuffle(validating_dataset)
        #np.random.set_state(rng_state)
        #np.random.shuffle(labels)
        validating_dataset = validating_dataset[0:int(0.8*len(validating_dataset))]
        labels = labels[0:int(0.8*len(labels))]
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


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, frac_per_epoch):
    try:
        model_base = "" #base_dir + "/Models/RealFinalSep/"
        model_name = "MC_SepNoGenNoShuffle_b" + str(batch_size) + "_p_" + str(
            patch_size) + "_drop_" + str(dropout_layer) + "_numDense_" + str(num_dense) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + \
                     "_denseN_" + str(dense_neuron) + "_convN_" + str(conv_neurons)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "SOURCE_" + model_name + ".h5",
                                                               monitor='val_loss',
                                                               verbose=0,
                                                               save_best_only=True,
                                                               save_weights_only=False,
                                                               mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                       patience=10 * frac_per_epoch,
                                                       verbose=0, mode='auto')

            def batchYielder():
                gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder()
                with h5py.File(path_mc_images, 'r') as f:
                    with h5py.File(path_proton_images, 'r') as f2:
                        items = len(f2['Image'])
                        items = items - (hadron_anteil * number_of_testing) - (hadron_anteil  * number_validate)
                        # Shuffle every time it starts from the beginning again
                        #rng_state = np.random.get_state()
                        times_train_in_items = int(np.floor(items / number_of_training))
                        if items > (hadron_anteil * number_of_training):
                            items = int(np.floor((hadron_anteil * number_of_training)))
                        section = 0
                        #section = section % times_train_in_items
                        offset = 0 * items
                        image1 = f['Image'][offset:int(offset + items)]
                        #np.random.set_state(rng_state)
                        image_false1 = f2['Image'][offset:int(offset + items)]
                        while True:
                            # Get some truth data for now, just use Crab images
                            batch_num = 0
                            np.random.shuffle(image1)
                            np.random.shuffle(image_false1)

                            # Roughly 5.6 times more simulated Gamma events than proton, so using most of them
                            while (hadron_anteil * batch_size) * (batch_num + 1) < items:
                                # Now the data is shuffled each time, hopefully improvi
                                images1 = image1[int(np.floor((batch_num) * (batch_size * gamma_anteil))):int(
                                    np.floor((batch_num + 1) * (batch_size * gamma_anteil)))]
                                images_false1 = image_false1[int(np.floor(batch_num * batch_size * hadron_anteil)):int(
                                    (batch_num + 1) * batch_size * hadron_anteil)]
                                validating_dataset1 = np.concatenate([images1, images_false1], axis=0)
                                labels1 = np.array([True] * (len(images1)) + [False] * len(images_false1))
                                #rng_state = np.random.get_state()
                                #np.random.set_state(rng_state)
                                #np.random.shuffle(validating_dataset1)
                                #np.random.set_state(rng_state)
                                #np.random.shuffle(labels1)
                                validation_labels1 = (np.arange(2) == labels1[:, None]).astype(np.float32)
                                x = validating_dataset1
                                x_label = validation_labels1
                                # print("Finished getting data")
                                batch_num += 1
                                yield (x, x_label)
                            section += 1
            # Make the model
            model = Sequential()

            # Base Conv layer
            model.add(Conv2D(64, kernel_size=(6,6), strides=(2, 2),
                            padding='same',
                             input_shape=(75, 75, 1)))
            model.add(LeakyReLU())
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            model.add(Dropout(0.25))

            model.add(
                Conv2D(128, (3,3), strides=(1, 1),
                       padding='same'))
            #model.add(BatchNormalization())
            model.add(LeakyReLU())
            #model.add(Activation('relu'))
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            model.add(Dropout(0.25))
            '''
            model.add(
                Conv2D(128, (3,3), strides=(1, 1),
                       padding='same'))
            #model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            
            #model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
            model.add(Dropout(0.25))

            model.add(
                Conv2D(64, patch_size, strides=(1, 1),
                       padding='same'))
            #model.add(BatchNormalization())
            model.add(Activation('relu'))

            model.add(
                Conv2D(32, patch_size, strides=(1, 1),
                       padding='same'))
            #model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
            model.add(Dropout(dropout_layer))
            '''
            model.add(Flatten())

            # Now do the dense layers
            #for i in range(num_dense):
            #    model.add(Dense(dense_neuron, activation='relu'))
                #model.add(BatchNormalization())
            #    if dropout_layer > 0.0:
            #        model.add(Dropout(dropout_layer))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))

        # Final Dense layer
            model.add(Dense(num_labels, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['acc'])
            model.summary()
            # Makes it only use
            #model.fit_generator(generator=batchYielder(), steps_per_epoch=int(
            #    np.floor(((number_of_training / (frac_per_epoch * batch_size)))))
            #                    , epochs=num_epochs,
            #                    verbose=2, validation_data=(y, y_label),
            #                    callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])
            model.fit(x=y, y=y_label, batch_size=batch_size, epochs=num_epochs, verbose=2, validation_split=0.2, callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])

            K.clear_session()
            tf.reset_default_graph()

    except Exception as e:
        print(e)
        K.clear_session()
        tf.reset_default_graph()
        pass


batch_sizes = [64,256]
patch_sizes = [(3, 3), (5, 5), (4, 4)]
dropout_layers = [0.0, 0.6]
num_conv_layers = [3, 4]
num_dense_layers = [3, 4]
num_conv_neurons = [27,128]
num_dense_neuron = [27,256]
num_pooling_layers = [1, 2]
num_runs = 500

for i in range(num_runs):
    dropout_layer = np.round(np.random.uniform(0.2, 1.0), 2)
    batch_size = np.random.randint(batch_sizes[0], batch_sizes[1])
    num_conv = np.random.randint(num_conv_layers[0], num_conv_layers[1])
    num_dense = np.random.randint(num_dense_layers[0], num_dense_layers[1])
    patch_size = patch_sizes[np.random.randint(0, 2)]
    num_pooling_layer = np.random.randint(num_pooling_layers[0], num_pooling_layers[1])
    dense_neuron = np.random.randint(num_dense_neuron[0], num_dense_neuron[1])
    conv_neurons = np.random.randint(num_conv_neurons[0], num_conv_neurons[1])
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, frac_per_epoch)
