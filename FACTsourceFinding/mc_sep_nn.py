import os
# to force on CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


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

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

# Hyperparameters

number_of_training = 600000 * (0.6)
number_of_testing = 600000 * (0.2)
number_validate = 600000 * (0.2)
num_labels = 2

# Total fraction to use per epoch of training data, need inverse
frac_per_epoch = 1
num_epochs = 600*frac_per_epoch

path_mc_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_proton_images = base_dir + "/Rebinned_5_MC_Proton_BothTracking_Images.h5"
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

    #gamma_anteil = 0.5
    #hadron_anteil = 0.5

    return gamma_anteil, hadron_anteil, gamma_count, hadron_count


with h5py.File(path_mc_images, 'r') as f:
    with h5py.File(path_proton_images, 'r') as f2:
        gamma_anteil, hadron_anteil, gamma_count, hadron_count = metaYielder()
        # Get some truth data for now, just use Crab images
        images = f['Image'][-int(np.floor((gamma_anteil * number_of_testing))):-1]
        images_false = f2['Image'][-int(np.floor((hadron_anteil * number_of_testing))):-1]
        validating_dataset = np.concatenate([images, images_false], axis=0)
        print(validating_dataset.shape)
        labels = np.array([True] * (len(images)) + [False] * len(images_false))
        ind, counts = np.unique(labels, return_counts=True)
        print(ind)
        print(counts)
        del images
        del images_false
        validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
        y = validating_dataset
        y_label = validation_labels
        print("Finished getting data")


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, frac_per_epoch):
    try:
        model_base = base_dir + "/Models/Sep/"
        model_name = "MC_SepAll_b" + str(batch_size) + "_p_" + str(
            patch_size) + "_drop_" + str(dropout_layer) + "_numDense_" + str(num_dense) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + \
                     "_denseN_" + str(dense_neuron) + "_convN_" + str(conv_neurons)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=25, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "{val_acc:.3f}_" + model_name + ".h5",
                                                               monitor='val_acc',
                                                               verbose=0,
                                                               save_best_only=True,
                                                               save_weights_only=False,
                                                               mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0,
                                                       patience=30 * frac_per_epoch,
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
                        image = f['Image'][offset:int(offset + items)]
                        #np.random.set_state(rng_state)
                        image_false = f2['Image'][offset:int(offset + items)]
                        while True:
                            # Get some truth data for now, just use Crab images
                            batch_num = 0
                            np.random.shuffle(image)
                            np.random.shuffle(image_false)

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
            model.add(Conv2D(conv_neurons, kernel_size=patch_size, strides=(1, 1),
                             activation='relu', padding='same',
                             input_shape=(75, 75, 1)))

            for i in range(num_conv):
                model.add(
                    Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu',
                           padding='same'))
                if num_pooling_layer == 1:
                    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                if dropout_layer > 0.0:
                    model.add(Dropout(dropout_layer))

            model.add(Flatten())

            # Now do the dense layers
            for i in range(num_dense):
                model.add(Dense(dense_neuron, activation='relu'))
                if dropout_layer > 0.0:
                    model.add(Dropout(dropout_layer))

            # Final Dense layer
            model.add(Dense(num_labels, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['acc'])
            print(model.summary())
            # Makes it only use
            model.fit_generator(generator=batchYielder(), steps_per_epoch=int(
                np.floor(((number_of_training / (frac_per_epoch * batch_size)))))
                                , epochs=num_epochs,
                                verbose=2, validation_data=(y, y_label),
                                callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])
            K.clear_session()
            tf.reset_default_graph()

    except Exception as e:
        print(e)
        K.clear_session()
        tf.reset_default_graph()
        pass


batch_sizes = [16,256]
patch_sizes = [(3, 3), (5, 5), (4, 4)]
dropout_layers = [0.0, 1.0]
num_conv_layers = [3, 6]
num_dense_layers = [1, 6]
num_conv_neurons = [27,128]
num_dense_neuron = [27,256]
num_pooling_layers = [1, 2]
num_runs = 500

for i in range(num_runs):
    dropout_layer = 0.0 #np.round(np.random.uniform(0.0, 1.0, size=1), 2)
    batch_size = 20#np.random.randint(batch_sizes[0], batch_sizes[1])
    num_conv = np.random.randint(num_conv_layers[0], num_conv_layers[1])
    num_dense = np.random.randint(num_dense_layers[0], num_dense_layers[1])
    patch_size = patch_sizes[np.random.randint(0, 2)]
    num_pooling_layer = np.random.randint(num_pooling_layers[0], num_pooling_layers[1])
    dense_neuron = np.random.randint(num_dense_neuron[0], num_dense_neuron[1])
    conv_neurons = np.random.randint(num_conv_neurons[0], num_conv_neurons[1])
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, frac_per_epoch)
