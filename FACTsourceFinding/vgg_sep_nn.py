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
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
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

path_mc_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Phi_Images.h5"
#path_mrk501 = "/run/media/jacob/WDRed8Tb1/dl2_theta/Mrk501_precuts.hdf5"

#mrk501 = read_h5py(path_mrk501, key="events", columns=["event_num", "night", "run_id", "source_x_prediction", "source_y_prediction"])
#mc_image = read_h5py_chunked(path_mc_images, key='events', columns=['Image', 'Event', 'Night', 'Run'])

def metaYielder():
    gamma_anteil = 0.5
    hadron_anteil = 0.5
    gamma_count = int(round(number_of_training*gamma_anteil))

    return gamma_anteil, hadron_anteil,  gamma_count


with h5py.File(path_mc_images, 'r') as f:
    gamma_anteil, hadron_anteil, gamma_count = metaYielder()
    # Get some truth data for now, just use Crab images
    images = f['GammaImage'][-int(np.floor((gamma_anteil * number_of_testing))):-1]
    images_false = f['Image'][-int(np.floor((hadron_anteil * number_of_testing))):-1]
    validating_dataset = np.concatenate([images, images_false], axis=0)
    labels = np.array([True] * (len(images)) + [False] * len(images_false))
    del images
    del images_false
    validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
    y = validating_dataset
    y_label = validation_labels
    print("Finished getting data")


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer):
    try:
        model_base = base_dir + "/Models/Sep/"
        model_name = "MC_vggSep_b" + str(batch_size) +"_p_" + str(patch_size) + "_drop_" + str(dropout_layer) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + "_denseN_" + str(dense_neuron) + "_numDense_" + str(num_dense) + "_convN_" + \
                     str(conv_neurons) + "_opt_" + str(optimizer)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.01, patience=70, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "{val_acc:.3f}_" + model_name + ".h5", monitor='val_acc', verbose=0,
                                                               save_best_only=True, save_weights_only=False, mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=80, verbose=0, mode='auto')

            def batchYielder():
                gamma_anteil, hadron_anteil, gamma_count = metaYielder()
                with h5py.File(path_mc_images, 'r') as f:
                    items = len(f['Image'])
                    items = items - (hadron_anteil * number_of_testing) - (hadron_anteil  * number_validate)
                    # Shuffle every time it starts from the beginning again
                    #rng_state = np.random.get_state()
                    times_train_in_items = int(np.floor(items / number_of_training))
                    if items > (hadron_anteil * number_of_training):
                        items = int(np.floor((hadron_anteil * number_of_training)))
                    section = 0
                    while True:
                        # Get some truth data for now, just use Crab images
                        batch_num = 0
                        section = section % times_train_in_items
                        offset = section * items
                        image = f['GammaImage'][offset:int(offset + items)]
                        #np.random.set_state(rng_state)
                        image_false = f['Image'][offset:int(offset + items)]
                        #np.random.shuffle(image)
                        #np.random.shuffle(image_false)

                        # Roughly 5.6 times more simulated Gamma events than proton, so using most of them
                        while (hadron_anteil * batch_size) * (batch_num + 1) < items:
                            # Now the data is shuffled each time, hopefully improvi
                            images = image[int(np.floor((batch_num) * (batch_size * gamma_anteil))):int(
                                np.floor((batch_num + 1) * (batch_size * gamma_anteil)))]
                            images_false = image_false[int(np.floor(batch_num * batch_size * hadron_anteil)):int(
                                (batch_num + 1) * batch_size * hadron_anteil)]
                            validating_dataset = np.concatenate([images, images_false], axis=0)
                            labels = np.array([True] * (len(images)) + [False] * len(images_false))
                            validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
                            x = validating_dataset
                            x_label = validation_labels
                            # print("Finished getting data")
                            batch_num += 1
                            yield (x, x_label)
                        section += 1

            # Make the model
            model = Sequential()

            # Base Conv layer
            model.add(Conv2D(64, kernel_size=(3, 3),
                             activation='relu', padding=optimizer,
                             input_shape=(75, 75, 1)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=optimizer))

            model.add(Conv2D(128, (3, 3), activation='relu', padding=optimizer))
            model.add(Conv2D(128, (3, 3), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=optimizer))

            model.add(Conv2D(256, (3, 3), activation='relu', padding=optimizer))
            model.add(Conv2D(256, (3, 3), activation='relu', padding=optimizer))
            model.add(Conv2D(256, (3, 3), activation='relu', padding=optimizer))
            model.add(Conv2D(256, (3, 3), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=optimizer))

            model.add(Conv2D(512, (3, 3), activation='relu', padding=optimizer))
            model.add(Conv2D(512, (3, 3), activation='relu', padding=optimizer))
            model.add(Conv2D(512, (3, 3), activation='relu', padding=optimizer))
            model.add(Conv2D(512, (3, 3), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=optimizer))

            model.add(Conv2D(512, (3, 3), activation='relu', padding=optimizer))
            model.add(Conv2D(512, (3, 3), activation='relu', padding=optimizer))
            model.add(Conv2D(512, (3, 3), activation='relu', padding=optimizer))
            model.add(Conv2D(512, (3, 3), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=optimizer))

            # Now classification part
            model.add(Flatten())
            model.add(Dense(4096, activation='relu'))
            model.add(Dense(4096, activation='relu'))



            # Final Dense layer
            # 2 so have one for x and one for y
            model.add(Dense(2, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
            model.fit_generator(generator=batchYielder(), steps_per_epoch=np.floor(((number_of_training / batch_size))), epochs=epoch,
                                verbose=2, validation_data=(y, y_label), callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])

            K.clear_session()
            tf.reset_default_graph()

    except Exception as e:
        print(e)
        K.clear_session()
        tf.reset_default_graph()
        pass


for i in range(num_runs):
    dropout_layer = np.round(np.random.uniform(0.0, 1.0), 2)
    batch_size = 100#np.random.randint(batch_sizes[0], batch_sizes[1])
    num_conv = np.random.randint(num_conv_layers[0], num_conv_layers[1])
    num_dense = np.random.randint(num_dense_layers[0], num_dense_layers[1])
    patch_size = patch_sizes[np.random.randint(0, 3)]
    num_pooling_layer = np.random.randint(num_pooling_layers[0], num_pooling_layers[1])
    dense_neuron = np.random.randint(num_dense_neuron[0], num_dense_neuron[1])
    conv_neurons = np.random.randint(num_conv_neurons[0], num_conv_neurons[1])
    optimizer = optimizers[np.random.randint(0,1)]
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer)
