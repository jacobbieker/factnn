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
num_conv_layers = [0,1,2,3,4,5]
num_dense_layers = [0,1,2,3,4,5]
num_conv_neurons = [8, 16, 32, 64, 128]
num_dense_neuron = [64, 128, 256, 512, 1024]
num_pooling_layers = [0,1]
number_of_training = 200000
number_of_testing = 10000
num_labels = 2

path_mc_images = base_dir + "/FACTSources/Rebinned_5_MC_Preprocessed_Images.h5"
for batch_size in batch_sizes:
    for patch_size in patch_sizes:
        for dropout_layer in dropout_layers:
            for num_conv in num_conv_layers:
                for num_dense in num_dense_layers:
                    for num_pooling_layer in num_pooling_layers:
                        for conv_neurons in num_conv_neurons:
                            for dense_neuron in num_dense_neuron:
                                for gamma_train in gamma_trains:
                                    model_name = "MC_b" + str(batch_size) +"_p_" + str(patch_size) + "_drop_" + str(dropout_layer) \
                                                 + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + "_gamma_" + \
                                                 str(gamma_train) + "_denseN_" + str(dense_neuron) + "_convN_" + str(conv_neurons) + ".h5"
                                    model_checkpoint = keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', verbose=0,
                                                                                       save_best_only=True, save_weights_only=False, mode='auto', period=1)
                                    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')
                                    def metaYielder():
                                        with h5py.File(path_mc_images, 'r') as f:
                                            keys = list(f.keys())
                                            events = []
                                            for key in keys:
                                                events.append(len(f[key]))

                                        gamma_anteil = events[0]/np.sum(events)
                                        hadron_anteil = events[1]/np.sum(events)

                                        gamma_count = int(round(number_of_training*gamma_anteil))
                                        hadron_count = int(round(number_of_training*hadron_anteil))

                                        return gamma_anteil, hadron_anteil, gamma_count, hadron_count


                                    with h5py.File(path_mc_images, 'r') as f:
                                        # Get some truth data for now, just use Crab images
                                        images = f['Gamma'][-number_of_testing:-1]
                                        images_false = f['Hadron'][-number_of_testing:-1]
                                        validating_dataset = np.concatenate([images, images_false], axis=0)
                                        labels = np.array([True] * (len(images)) + [False] * len(images_false))
                                        del images
                                        del images_false
                                        validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
                                        y = validating_dataset
                                        y_label = validation_labels
                                        print("Finished getting data")


                                    def batchYielder():
                                        gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder()
                                        while True:
                                            with h5py.File(path_mc_images, 'r') as f:
                                                # Get some truth data for now, just use Crab images
                                                items = list(f.items())[1][1].shape[0]
                                                items = items - number_of_testing
                                                batch_num = 0
                                                # Roughly 5.6 times more simulated Gamma events than proton, so using most of them
                                                while (hadron_count) * (batch_num + 1) < items:
                                                    images = f['Gamma'][batch_num * gamma_count:(batch_num + 1) * gamma_count]
                                                    images_false = f['Hadron'][batch_num * hadron_count:(batch_num + 1) * hadron_count]
                                                    validating_dataset = np.concatenate([images, images_false], axis=0)
                                                    labels = np.array([True] * (len(images)) + [False] * len(images_false))
                                                    del images
                                                    del images_false
                                                    validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
                                                    x = validating_dataset
                                                    x_label = validation_labels
                                                    # print("Finished getting data")
                                                    batch_num += 1
                                                    yield (x, x_label)

                                    gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder()
                                    # Make the model
                                    model = Sequential()

                                    # Base Conv layer
                                    model.add(Conv2D(conv_neurons, kernel_size=patch_size, strides=(1, 1),
                                                     activation='relu', padding='same',
                                                     input_shape=(75, 75, 1)))

                                    for i in range(num_conv):
                                        model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding='same'))
                                        if num_pooling_layer == 1:
                                            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                                        model.add(Dropout(dropout_layer))

                                    # Now do the dense layers
                                    for i in range(num_dense):
                                        model.add(Dense(dense_neuron, activation='relu'))
                                        model.add(Dropout(dropout_layer))

                                    # Final Dense layer
                                    model.add(Dense(num_labels, activation='sigmoid'))
                                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
                                    model.fit_generator(generator=batchYielder(), steps_per_epoch=np.floor(((number_of_training / batch_size))), epochs=100,
                                                        verbose=2, validation_data=(y, y_label), callbacks=[early_stop, model_checkpoint])