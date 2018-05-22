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
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, LeakyReLU, Reshape, BatchNormalization, Conv2D, MaxPooling2D, ConvLSTM2D
import fact.plotting as factplot

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def batchYielder(path_to_training_data, type_training, percent_training, num_events_per_epoch=1000, path_to_proton_data=None, batch_size=64):
    with h5py.File(path_to_training_data, 'r') as f:
        items = list(f.items())[1][1].shape[0]
        items = int(items*percent_training)
        length_dataset = len(f['Image'])
        section = 0
        times_train_in_items = int(np.floor(items / num_events_per_epoch))
        image = f['Image']
        if type_training == 'Energy':
            energy = f['Energy']
        elif type_training == "Disp":
            source_y = f['Source_X']
            source_x = f['Source_Y']
            cog_x = f['COG_X']
            cog_y = f['COG_Y']

        elif type_training == "Sign":
            source_y = f['Source_X']
            source_x = f['Source_Y']
            cog_x = f['COG_X']
            cog_y = f['COG_Y']
            delta = f['Delta']

        elif type_training == "Separation":
            if path_to_proton_data is None:
                print("Error: No Proton File")
                exit(-1)
            else:
                with h5py.File(path_to_proton_data, 'r') as f2:
                    proton_data = f2['Image']

        while True:
            # Now create the batches from labels and other things
            batch_num = 0
            section = section % times_train_in_items
            offset = int(section * num_events_per_epoch)
            while batch_size * (batch_num + 1) < items:
                batch_images = image[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                if type_training == 'Energy':
                    batch_image_label = energy[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                elif type_training == "Disp":
                    source_x_tmp = source_x[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    source_y_tmp = source_y[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    cog_x_tmp = cog_x[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    cog_y_tmp = cog_y[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    batch_image_label = euclidean_distance(
                        source_x_tmp, source_y_tmp,
                        cog_x_tmp, cog_y_tmp
                    )
                elif type_training == "Sign":
                    source_x_tmp = source_x[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    source_y_tmp = source_y[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    cog_x_tmp = cog_x[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    cog_y_tmp = cog_y[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    delta_tmp = delta[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    true_delta = np.arctan2(
                        source_x_tmp, source_y_tmp,
                        cog_x_tmp, cog_y_tmp
                    )
                    true_sign = np.sign(np.abs(delta_tmp - true_delta) - np.pi / 2)
                    temp_sign = []
                    for sign in true_sign:
                        if sign < 0:
                            temp_sign.append([1,0])
                        else:
                            temp_sign.append([0,1])
                    batch_image_label = np.asarray(temp_sign)
                elif type_training == "Separation":
                    proton_images = proton_data[offset + int(batch_num*batch_size):offset + int((batch_num+1)*batch_size)]
                    #proton_images = np.swapaxes(proton_images, 0, 2)
                    #batch_images = np.swapaxes(batch_images, 0, 2)
                    labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                    batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                    batch_images = np.concatenate([batch_images, proton_images], axis=0)

                batch_num += 1
                yield (batch_images, batch_image_label)
            section += 1


architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

# Hyperparameters

path_mc_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Gamma_TimInfo_Images.h5"
path_proton_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Proton_Timing_Images.h5"
np.random.seed(0)
# Make the model

# Get length of model for later use
with h5py.File(path_proton_images, 'r') as f2:
    length_items = len(f2['Image'])

predicting_labels = []

# Need validation generator
def validationGenerator(validation_percentage, batch_size=64, predicting=False):
    with h5py.File(path_mc_images, 'r') as f:
        with h5py.File(path_proton_images, 'r') as f2:
            # Get some truth data for now, just use Crab images
            items = len(f2["Image"])
            validation_test = validation_percentage * items
            num_batch_in_validate = int(validation_test / batch_size)
            section = 0
            images = f['Image']
            images_false = f2['Image']
            while True:
                batch_num = 0
                section = section % num_batch_in_validate
                offset = int(section * num_batch_in_validate)
                while batch_size * (batch_num + 1) < items:
                    batch_images = images[-1*(offset + int((batch_num+1)*batch_size)):-1*(offset + int((batch_num)*batch_size))]
                    proton_images = images_false[-1*(offset + int((batch_num+1)*batch_size)):-1*(offset + int((batch_num)*batch_size))]
                    labels = np.array([True] * (len(batch_images)) + [False] * len(proton_images))
                    batch_image_label = (np.arange(2) == labels[:, None]).astype(np.float32)
                    batch_images = np.concatenate([batch_images, proton_images], axis=0)
                    batch_num += 1
                    if predicting:
                        predicting_labels = np.concatenate([predicting_labels, batch_image_label], axis=0)
                    yield (batch_images, batch_image_label)
                section += 1

from sklearn.metrics import roc_auc_score

def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, frac_per_epoch):
    #try:
        model_base = base_dir + "/Models/3DSep/"
        model_name = "MC_Seperation3D" + "_p_" + str(
            patch_size) + "_drop_" + str(dropout_layer) + "_numDense_" + str(num_dense) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + \
                     "_denseN_" + str(dense_neuron) + "_convN_" + str(conv_neurons)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "_" + model_name + ".h5",
                                                               monitor='val_loss',
                                                               verbose=0,
                                                               save_best_only=True,
                                                               save_weights_only=False,
                                                               mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                       patience=10 * frac_per_epoch,
                                                       verbose=0, mode='auto')

                # Make the model
            model = Sequential()

            # Base Conv layer
            model.add(ConvLSTM2D(32, kernel_size=3, strides=2,
                                 padding='same',
                                 input_shape=(100, 75, 75, 1), activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True))
            model.add(
                ConvLSTM2D(64, kernel_size=3, strides=2,
                           padding='same', activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True))
            model.add(
                ConvLSTM2D(128, kernel_size=3, strides=2,
                           padding='same', recurrent_activation='hard_sigmoid', activation='relu'))
            model.add(Dropout(0.5))

            model.add(Flatten())

            for i in range(1):
                model.add(Dense(256, activation='relu'))
                model.add(Dropout(1/4))
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(1/4))

            # Final Dense layer
            model.add(Dense(2, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['acc'])
            model.summary()
            # Makes it only use
            model.fit_generator(generator=batchYielder(path_to_training_data=path_mc_images, path_to_proton_data=path_proton_images, type_training="Seperation", batch_size=64, percent_training=0.6),
                                steps_per_epoch=int(np.floor(0.6*length_items/64))
                                , epochs=400,
                                verbose=2, validation_data=validationGenerator(0.2), validation_steps=int(np.floor(0.2*length_items/64)),
                                callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint],
                                use_multiprocessing=True, workers=10)
            predictions = model.predict_generator(validationGenerator(0.2, predicting=True), steps=int(np.floor(0.2*length_items/64)))
            print(roc_auc_score(predicting_labels, predictions))
            K.clear_session()
            tf.reset_default_graph()

    #except Exception as e:
    #    print(e)
    #    K.clear_session()
    #    tf.reset_default_graph()
    #    pass


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
    batch_size = 64#np.random.randint(batch_sizes[0], batch_sizes[1])
    num_conv = np.random.randint(num_conv_layers[0], num_conv_layers[1])
    num_dense = np.random.randint(num_dense_layers[0], num_dense_layers[1])
    patch_size = patch_sizes[np.random.randint(0, 2)]
    num_pooling_layer = np.random.randint(num_pooling_layers[0], num_pooling_layers[1])
    dense_neuron = np.random.randint(num_dense_neuron[0], num_dense_neuron[1])
    conv_neurons = np.random.randint(num_conv_neurons[0], num_conv_neurons[1])
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, frac_per_epoch)
