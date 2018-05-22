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
from keras.layers import Dense, Dropout, Activation, Conv1D, ELU, Flatten, ConvLSTM2D, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score


architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

def atan2(x, y, epsilon=1.0e-12):
    x = tf.where(tf.equal(x, 0.0), x+epsilon, x)
    y = tf.where(tf.equal(y, 0.0), y+epsilon, y)
    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
    return angle

# y in radians and second column is az, first is zd, adds the errors together, seems to work?
def rmse_360_2(y_true, y_pred):
    az_error = K.mean(K.abs(tf.atan2(K.sin(y_true - y_pred), K.cos(y_true - y_pred))))
    #zd_error = K.mean(K.square(y_pred[:,0] - y_true[:,0]), axis=-1)
    return az_error

def euc_dist_keras(y_true, y_pred):
    taxicab_error_wrong = K.abs(y_true[:,0] - y_pred[:,1]) + K.abs(y_true[:,1] - y_pred[:,0])
    other_error = K.sqrt(K.sum(K.square(y_true[:,0] - y_pred[:,0]) + K.square(y_true[:,1] - y_pred[:,1])))
    return other_error
# Hyperparameters

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_sourceX_Y_confusion(performace_df, label, log_xy=True, log_z=True, ax=None):

    ax = ax or plt.gca()

    #label = performace_df.label.copy()
    prediction = performace_df.copy()

    if log_xy is False:
        label = np.log10(label)
        prediction = np.log10(prediction)

    min_label = np.min(label)
    min_pred = np.min(prediction)
    max_pred = np.max(prediction)
    max_label = np.max(label)

    if min_label < min_pred:
        min_ax = min_label
    else:
        min_ax = min_pred

    if max_label > max_pred:
        max_ax = max_label
    else:
        max_ax = max_pred

    if log_z:
        min_ax = min_label
        max_ax = max_label
    limits = [
        min_ax,
        max_ax
    ]
    print(limits)
    print("Max, min Label")
    print([min_label, max_label])

    counts, x_edges, y_edges, img = ax.hist2d(
        label,
        prediction,
        bins=[100, 100],
        range=[limits, limits],
        norm=LogNorm() if log_xy is False else None
    )
    ax.set_aspect(1)
    ax.figure.colorbar(img, ax=ax)

    if log_xy is True:
        ax.set_xlabel(r'$X$')
        ax.set_ylabel(r'$Y$')
    else:
        ax.set_xlabel(r'$X_{\mathrm{MC}}$')
        ax.set_ylabel(r'$Y_{\mathrm{Est}}$')

    return ax


batch_sizes = [16, 64, 256]
patch_sizes = [(2, 2), (3, 3), (5, 5), (4, 4)]
dropout_layers = [0.1, 0.6]
num_conv_layers = [0, 6]
num_dense_layers = [0, 6]
num_conv_neurons = [8,128]
num_dense_neuron = [8,256]
num_pooling_layers = [0, 2]
num_runs = 500
number_of_training = 529000*(0.6)
number_of_testing = 529000*(0.2)
number_validate = 529000*(0.2)
optimizers = ['same']
epoch = 500

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

path_mc_images = "/run/media/jacob/SSD/Rebinned_5_MC_diffuse_BothSource_Images.h5"
path_mc_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_diffuse_DELTA5000_Images.h5"
path_mc_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_diffuse_SOURCEXYALLSTDDEV_Images.h5"
#path_mrk501 = "/run/media/jacob/WDRed8Tb1/dl2_theta/Mrk501_precuts.hdf5"

#mrk501 = read_h5py(path_mrk501, key="events", columns=["event_num", "night", "run_id", "source_x_prediction", "source_y_prediction"])
#mc_image = read_h5py_chunked(path_mc_images, key='events', columns=['Image', 'Event', 'Night', 'Run'])

def batchYielder(path_to_training_data, type_training, percent_training, num_events_per_epoch=1000, path_to_proton_data=None, batch_size=64):
    with h5py.File(path_to_training_data, 'r') as f:
        items = list(f.items())[1][1].shape[0]
        items = int(items*percent_training)
        length_dataset = len(f['Image'])
        section = 0
        offset = int(section * num_events_per_epoch)
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

with h5py.File(path_mc_images, 'r') as f:
    items = len(f["Image"])
    validation_test = 0.2 * items
    images = f['Image'][-validation_test:]
    source_y = f['Source_X'][-validation_test:]
    source_x = f['Source_Y'][-validation_test:]
    cog_x = f['COG_X'][-validation_test:]
    cog_y = f['COG_Y'][-validation_test:]
    np.random.seed(0)

    true_disp = euclidean_distance(
        source_x, source_y,
        cog_x, cog_y
    )
    images = images[0:int(0.8*len(images))]#int(0.01*len(images))]
    disp_train = true_disp[0:int(0.8*len(true_disp))]

    print(images.shape)
    # Now convert to this camera's coordinates
    y = images#[1000:-1]#np.rot90(images, axis=2)
    title = "3D Disp"
    desc = "3D Disp"
    print(images.shape)
    y_label = disp_train#[1000:-1]
    print("Finished getting data")


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer):
    #try:
    model_base = ""# base_dir +"/" # + "/Models/FinalSourceXY/test/test/"
    model_name = "MC_3DDisp" + "_drop_" + str(dropout_layer)
    if not os.path.isfile(model_base + model_name + ".csv"):
        csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
        #reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=70, min_lr=0.001)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, verbose=0, mode='auto')

        model_checkpointy = keras.callbacks.ModelCheckpoint(model_base + "Y_" + desc + model_name + ".h5", monitor='val_loss', verbose=0,
                                                            save_best_only=True, save_weights_only=False, mode='auto', period=1)
        # Make the model
        inp = keras.layers.Input((100,75,75,1))
        # Block - conv
        y = ConvLSTM2D(64, kernel_size=8, strides=4, activation='elu', recurrent_activation='hard_sigmoid', return_sequences=True)(inp)
        y = ConvLSTM2D(128, kernel_size=5, strides=2, activation='elu', recurrent_activation='hard_sigmoid', return_sequences=True)(y)
        y = Dropout(dropout_layer)(y)
        y = ConvLSTM2D(256, kernel_size=5, strides=2, activation='elu', recurrent_activation='hard_sigmoid', return_sequences=True)(y)
        y = ConvLSTM2D(512, kernel_size=5, strides=2, activation='elu', recurrent_activation='hard_sigmoid')(y)

        # Block - flatten
        y = Flatten()(y)
        y = Dropout(dropout_layer)(y)
        y = ELU()(y)

        # Block - fully connected
        y = Dense(dense_neuron, activation='elu', name='FC1')(y)
        y = Dropout(0.3)(y)
        y = ELU()(y)
        y = Dense(dense_neuron, activation='elu', name='FC2')(y)
        y = Dropout(0.3)(y)
        y = ELU()(y)

        y_out = Dense(1, name="y_out", activation='linear')(y)

        model = keras.models.Model(inp, y_out)
        # Block - output
        model.summary()
        adam = keras.optimizers.adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse', metrics=['mae'])

        model.fit_generator(generator=batchYielder(path_to_training_data=path_mc_images, path_to_proton_data=None, type_training="Disp", percent_training=0.6),
                            steps_per_epoch=np.floor(items/64)
                            , epochs=400,
                            verbose=2, validation_data=(y, y_label),
                            callbacks=[early_stop, csv_logger])
        predictions = model.predict(y, batch_size=64)
        predictions_x = predictions.reshape(-1,)

        score = r2_score(y_label, predictions_x)

        fig1 = plt.figure()
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_title(title + " R^2: " + str(score) + ' Reconstructed Train Disp vs. True Train Disp')
        plot_sourceX_Y_confusion(predictions_x, y_label, ax=ax)
        fig1.show()

        K.clear_session()
        tf.reset_default_graph()

#except Exception as e:
#    print(e)
#    K.clear_session()
#    tf.reset_default_graph()
#    pass


for i in range(num_runs):
    dropout_layer = np.round(np.random.uniform(0.0, 1.0), 2)
    batch_size = np.random.randint(batch_sizes[0], batch_sizes[1])
    num_conv = np.random.randint(num_conv_layers[0], num_conv_layers[1])
    num_dense = np.random.randint(num_dense_layers[0], num_dense_layers[1])
    patch_size = patch_sizes[np.random.randint(0, 3)]
    num_pooling_layer = np.random.randint(num_pooling_layers[0], num_pooling_layers[1])
    dense_neuron = np.random.randint(num_dense_neuron[0], num_dense_neuron[1])
    conv_neurons = np.random.randint(num_conv_neurons[0], num_conv_neurons[1])
    optimizer = optimizers[np.random.randint(0,1)]
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer)
