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
from keras.layers import Dense, Dropout, Activation, Conv1D, ELU, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera
import pandas as pd

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
    az_error = tf.reduce_mean(K.abs(tf.atan2(K.sin(y_true[:,1] - y_pred[:,1]), K.cos(y_true[:,1] - y_pred[:,1]))))
    zd_error = K.mean(K.square(y_pred[:,0] - y_true[:,0]), axis=-1)
    return az_error + zd_error

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

    min_label = np.floor(np.min(label))
    min_pred = np.floor(np.min(prediction))
    max_pred = np.ceil(np.max(prediction))
    max_label = np.ceil(np.max(label))

    if min_label < min_pred:
        min_ax = min_label
    else:
        min_ax = min_pred

    if max_label > max_pred:
        max_ax = max_label
    else:
        max_ax = max_pred

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
        norm=LogNorm() if log_z is True else None,
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
dropout_layers = [0.0, 1.0]
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

path_mc_images = "/run/media/jacob/SSD/Rebinned_5_MC_diffuse_BothSource_Images.h5"
#path_mrk501 = "/run/media/jacob/WDRed8Tb1/dl2_theta/Mrk501_precuts.hdf5"

#mrk501 = read_h5py(path_mrk501, key="events", columns=["event_num", "night", "run_id", "source_x_prediction", "source_y_prediction"])
#mc_image = read_h5py_chunked(path_mc_images, key='events', columns=['Image', 'Event', 'Night', 'Run'])

def metaYielder():
    gamma_anteil = 1
    gamma_count = int(round(number_of_training*gamma_anteil))

    return gamma_anteil, gamma_count




with h5py.File(path_mc_images, 'r') as f:
    gamma_anteil, gamma_count = metaYielder()
    images = f['Image'][0:-1]
    source_x = f['Source_X'][0:-1]
    source_y = f['Source_Y'][0:-1]
    #images_source_az = f['Az_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    #images_source_zd = f['Zd_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    #images_source_az = (-1.*images_source_az + 540) % 360
    np.random.seed(0)
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(source_x)
    np.random.set_state(rng_state)
    np.random.shuffle(source_y)
    images = images[0:250000]#int(0.01*len(images))]
    source_x = source_x[0:250000]#int(0.01*len(source_x))]
    source_y = source_y[0:250000]#int(0.01*len(source_y))]

    #Normalize each image
    transformed_images = []
    for image_one in images:
        #print(image_one.shape)
        image_one = image_one/np.sum(image_one)
        #print(np.sum(image_one))
        transformed_images.append(image_one)
        #print(np.max(image_one))
    images = np.asarray(transformed_images)
    print(images.shape)
    # Now convert to this camera's coordinates
    y = images #np.flip(images, axis=2)
    print(images.shape)
    print(source_x[0])
    #print(source_y[0])
    y_label = source_y# np.column_stack((source_x, source_y))
    x_label = source_x
    print(y_label[0])
    print(np.min(y_label))
    print(np.max(y_label))
    print(y_label.shape)
    print("Finished getting data")


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer):
    #try:
        model_base = "" #base_dir + "/Models/FinalSourceXY/test/test/"
        model_name = "MC_Split_b" + str(batch_size) +"_p_" + str(patch_size) + "_drop_" + str(dropout_layer) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + "_denseN_" + str(dense_neuron) + "_numDense_" + str(num_dense) + "_convN_" + \
                     str(conv_neurons) + "_opt_" + str(optimizer)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            #reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=70, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "{loss:.3f}_" + model_name + ".h5", monitor='loss', verbose=0,
                                                               save_best_only=True, save_weights_only=False, mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=80, verbose=0, mode='auto')

            inp = keras.models.Input((75,75,1))
            # Block - conv
            x = Conv2D(16, 8, 8, border_mode='same', subsample=[4,4], activation='elu', name='Conv1')(inp)
            # Block - conv
            x = Conv2D(32, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv2')(x)
            # Block - conv
            x = Conv2D(64, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv3')(x)
            x = MaxPooling2D(padding='same')(x)
            x = Dropout(dropout_layer)(x)
            # Block - conv
            x = Conv2D(32, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv5')(x)
            # Block - conv
            x = Conv2D(64, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv6')(x)
            # Block - flatten
            x = Flatten()(x)
            x = Dropout(dropout_layer)(x)
            x = ELU()(x)
            # Block - fully connected
            x = Dense(dense_neuron, activation='elu', name='FC1')(x)
            x = Dropout(0.5)(x)
            x = ELU()(x)

            x_out = Dense(1, name="x_out")(x)
            y_out = Dense(1, name="y_out")(x)

            model_x = keras.models.Model(inp, x_out)
            # Block - output
            model_x.summary()

            '''
            # Base Conv layer
            model.add(Conv2D(32, kernel_size=patch_size, strides=(1, 1),
                             activation='relu', padding=optimizer,
                             input_shape=(75, 75, 1)))
            model.add(Conv2D(8, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), padding=optimizer))
            for i in range(num_conv):
                model.add(Conv2D(16, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
                model.add(Conv2D(16, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
                model.add(MaxPooling2D(pool_size=(2, 2), padding=optimizer))

            model.add(Flatten())

            for i in range(num_dense):
                model.add(Dense(512, activation='linear'))
                model.add(Dropout(dropout_layer))


            # Final Dense layer
            # 2 so have one for x and one for y
            model.add(Dense(2, activation='linear'))
            '''
            adam = keras.optimizers.adam(lr=0.00001)
            model_x.compile(optimizer=adam, loss='mse', metrics=['mae'])
            #model.fit_generator(generator=batchYielder(), steps_per_epoch=np.floor(((number_of_training / batch_size))), epochs=epoch,
            #                    verbose=2, validation_data=(y, y_label), callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])
            model_x.fit(x=y, y=x_label, batch_size=batch_size, epochs=500, verbose=2, validation_split=0.2, callbacks=[early_stop, csv_logger, model_checkpoint])
            model_x.save("x_temp.h5")
            predictions_x = model_x.predict(y, batch_size=64)
            fig1 = plt.figure()
            ax = fig1.add_subplot(1, 1, 1)
            ax.set_title(' Reconstructed vs. True X')
            plot_sourceX_Y_confusion(predictions_x, x_label, ax=ax)
            fig1.show()
            K.clear_session()
            tf.reset_default_graph()

            inp = keras.models.Input((75,75,1))
            # Block - conv
            x = Conv2D(32, 8, 8, border_mode='same', subsample=[4,4], activation='elu', name='Conv1')(inp)
            # Block - conv
            x = Conv2D(64, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv2')(x)
            # Block - conv
            x = Conv2D(128, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv3')(x)
            x = MaxPooling2D(padding='same')(x)
            x = Dropout(dropout_layer)(x)
            # Block - conv
            x = Conv2D(64, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv5')(x)
            # Block - conv
            x = Conv2D(128, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv6')(x)
            # Block - flatten
            x = Flatten()(x)
            x = Dropout(dropout_layer)(x)
            x = ELU()(x)
            # Block - fully connected
            x = Dense(dense_neuron, activation='elu', name='FC1')(x)
            x = Dropout(0.5)(x)
            x = ELU()(x)

            y_out = Dense(1, name="y_out")(x)
            model_y = keras.models.Model(inp, y_out)
            adam = keras.optimizers.adam(lr=0.0001)
            model_y.compile(optimizer=adam, loss='mse', metrics=['mae'])
            model_y.fit(x=y, y=y_label, batch_size=batch_size, epochs=500, verbose=2, validation_split=0.2, callbacks=[early_stop, csv_logger, model_checkpoint])

            predictions_y = model_y.predict(y, batch_size=64)

            #predictions = model.predict(images, batch_size=64)
            print(predictions_x.shape)
            print(predictions_x)
            predictions_x = predictions_x.reshape(-1,)
            predictions_y = predictions_y.reshape(-1,)
            #predictions[:,0] += 180.975/2 # shifts everything to positive
            #predictions[:,1] += 185.25/2 # shifts everything to positive
            #predictions[:,0] = predictions[:,0] / 4.94 # Ratio between the places
            #predictions[:,1] = predictions[:,1] / 4.826 # Ratio between y in original and y here
            # Now make the confusion matrix

            #Loss Score so can tell which one it is

            fig1 = plt.figure()
            ax = fig1.add_subplot(1, 1, 1)
            ax.set_title(' Reconstructed vs. True X')
            plot_sourceX_Y_confusion(predictions_x, x_label, ax=ax)
            fig1.show()


            fig1 = plt.figure()
            ax = fig1.add_subplot(1, 1, 1)
            ax.set_title(' Reconstructed X vs. Rec Y')
            plot_sourceX_Y_confusion(predictions_x, predictions_y, ax=ax)
            fig1.show()

            fig1 = plt.figure()
            ax = fig1.add_subplot(1, 1, 1)
            ax.set_title(' True X vs. True Y')
            plot_sourceX_Y_confusion(x_label, y_label, ax=ax)
            fig1.show()

            fig1 = plt.figure()
            ax = fig1.add_subplot(1, 1, 1)
            ax.set_title(' Reconstructed vs. True Y')
            plot_sourceX_Y_confusion(predictions_y, y_label, log_xy=True, ax=ax)
            fig1.show()
            exit(1)
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
