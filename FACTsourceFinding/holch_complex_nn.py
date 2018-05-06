import os
# to force on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
import h5py
from fact.io import read_h5py
import yaml
import tensorflow as tf
import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera

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
number_of_training = 100000*(0.6)
number_of_testing = 100000*(0.2)
number_validate = 100000*(0.2)
optimizer = 'adam'
epoch = 100

path_mc_images = base_dir + "/FACTSources/Rebinned_5_MC_Gamma_1_Diffuse_Images.h5"

def metaYielder():
    gamma_anteil = 1
    gamma_count = int(round(number_of_training*gamma_anteil))

    return gamma_anteil, gamma_count


with h5py.File(path_mc_images, 'r') as f:
    gamma_anteil, gamma_count = metaYielder()
    # Get some truth data for now, just use Crab images
    images = f['Image'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    images_source_zd = f['Theta'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    images_source_az = f['Phi'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    images_point_az = f['Az_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    images_point_zd = f['Zd_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    images_source_az = (-1.*images_source_az + 540) % 360
    source_x, source_y = horizontal_to_camera(
        zd=images_source_zd, az=images_source_az,
        az_pointing=images_point_az, zd_pointing=images_point_zd
    )

    y = images
    y_label = np.asarray([source_x, source_y]).reshape((-1, 2))
    print(y_label.shape)
    print("Finished getting data")


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons):
    try:
        model_base = base_dir + "/Models/Disp/"
        model_name = "MC_dispPhi_b" + str(batch_size) +"_p_" + str(patch_size) + "_drop_" + str(dropout_layer) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + "_denseN_" + str(dense_neuron) + "_numDense_" + str(num_dense) + "_convN_" + \
                     str(conv_neurons) + "_opt_" + str(optimizer)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=25, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "{loss:.3f}_" + model_name + ".h5", monitor='loss', verbose=0,
                                                               save_best_only=True, save_weights_only=False, mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode='auto')

            def batchYielder():
                gamma_anteil, gamma_count = metaYielder()
                with h5py.File(path_mc_images, 'r') as f:
                    items = list(f.items())[1][1].shape[0]
                    items = items - number_of_testing - number_validate
                    times_train_in_items = int(np.floor(items / number_of_training))
                    length_dataset = len(f['Image'])
                    if items > number_of_training:
                        items = number_of_training
                    section = 0
                    while True:
                        batch_num = 0
                        section = section % times_train_in_items
                        offset = int(section * items)
                        image = f['Image'][offset:int(offset + items)]
                        source_zd = f['Theta'][offset:int(offset + items)]
                        source_az = f['Phi'][offset:int(offset + items)]
                        point_az = f['Az_deg'][offset:int(offset + items)]
                        point_zd = f['Zd_deg'][offset:int(offset + items)]
                        source_az = (-1.*source_az + 540) % 360
                        source_x, source_y = horizontal_to_camera(
                            zd=source_zd, az=source_az,
                            az_pointing=point_az, zd_pointing=point_zd
                        )

                        rng_state = np.random.get_state()
                        np.random.shuffle(image)
                        np.random.set_state(rng_state)
                        np.random.shuffle(source_x)
                        np.random.set_state(rng_state)
                        np.random.shuffle(source_y)
                        #np.random.set_state(rng_state)
                        #np.random.shuffle(point_zd)
                        #np.random.set_state(rng_state)
                        #np.random.shuffle(point_az)
                        # Roughly 5.6 times more simulated Gamma events than proton, so using most of them
                        while (batch_size) * (batch_num + 1) < items:
                            # Get some truth data for now, just use Crab images
                            images = image[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            #images_source_zd = source_zd[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            #images_source_az = source_az[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            #images_point_az = point_az[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            #images_point_zd = point_zd[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            images_source_x = source_x[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            images_source_y = source_y[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            x = images
                            x_label = np.asarray([images_source_x, images_source_y]).reshape((-1, 2))
                            #print(x_label.shape)
                            batch_num += 1
                            yield (x, x_label)
                        section += 1

            # Make the model
            model = Sequential()

            # Base Conv layer
            model.add(Conv2D(conv_neurons, kernel_size=patch_size, strides=(1, 1),
                             activation='relu', padding='same',
                             input_shape=(75, 75, 1)))
            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding='same'))
            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding='same'))
            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding='same'))
            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding='same'))
            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

            model.add(Flatten())

            model.add(Dense(dense_neuron, activation=None))
            model.add(Dropout(0.8))
            model.add(Dense(dense_neuron, activation=None))
            model.add(Dropout(0.8))
            model.add(Dense(dense_neuron, activation=None))
            model.add(Dropout(0.8))

            # Final Dense layer
            # 2 so have one for x and one for y
            model.add(Dense(2, activation=None))
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
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
    dropout_layer = 0.0 #np.round(np.random.uniform(0.0, 1.0, size=1), 2)
    batch_size = np.random.randint(batch_sizes[0], batch_sizes[1])
    num_conv = np.random.randint(num_conv_layers[0], num_conv_layers[1])
    num_dense = np.random.randint(num_dense_layers[0], num_dense_layers[1])
    patch_size = patch_sizes[np.random.randint(0, 3)]
    num_pooling_layer = np.random.randint(num_pooling_layers[0], num_pooling_layers[1])
    dense_neuron = np.random.randint(num_dense_neuron[0], num_dense_neuron[1])
    conv_neurons = np.random.randint(num_conv_neurons[0], num_conv_neurons[1])
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons)
