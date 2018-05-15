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
from sklearn.metrics import roc_auc_score


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
        #norm=LogNorm()
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
path_mc_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_diffuse_DELTAALL_Images.h5"
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
    source_az = f['Source_Az'][0:-1]
    point_x = f['Az_deg'][0:-1]
    point_y = f['Zd_deg'][0:-1]
    #source_x = np.deg2rad(source_x)
    #point_x = np.deg2rad(point_x)
    source_zd = f['Source_Zd'][0:-1]
    cog_x = f['COG_X'][0:-1]
    cog_y = f['COG_Y'][0:-1]
    delta = f['Delta'][0:-1]
    #energy = f['Energy'][0:-1]
    #images_source_az = f['Az_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    #images_source_zd = f['Zd_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    #images_source_az = (-1.*images_source_az + 540) % 360
    np.random.seed(0)
    source_x, source_y = horizontal_to_camera(
        az=source_az,
        zd=source_zd,
        az_pointing=point_x,
        zd_pointing=point_y,
    )

    #true_disp = euclidean_distance(
    #    source_x, source_y,
    #    cog_x, cog_y
    #)
    true_delta = np.arctan2(
        cog_y - source_y,
        cog_x - source_x,
        )
    true_sign = np.sign(np.abs(delta - true_delta) - np.pi / 2)
    temp_sign = []
    for sign in true_sign:
        if sign < 0:
            temp_sign.append([1,0])
        else:
            temp_sign.append([0,1])
    true_sign = np.asarray(temp_sign)
    del temp_sign
    y_train_images = images #np.asarray(transformed_images)
    images_test = images[-int(0.5*len(images)):]#int(0.01*len(images))]
    disp_test = true_sign[-int(0.5*len(images)):]
    images = images[0:int(0.5*len(images))]#int(0.01*len(images))]
    disp_train = true_sign[0:int(0.5*len(true_sign))]

    images_test_y = images_test
    #transformed_images = []
    print(images.shape)
    # Now convert to this camera's coordinates
    y = images#[1000:-1]#np.rot90(images, axis=2)
    y_train = images
    title = "SeparateOutputs Sign"
    desc = "SeparateOutputs Sign"
    print(images.shape)
    #print(source_x[0])
    #print(source_y[0])
    y_label = disp_train#[1000:-1]
    x_label = disp_test#sign_train#[1000:-1]
    print(y_label[0])
    print(y_label.shape)
    y_label = y_label
    print("Finished getting data")

def create_model2(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, frac_per_epoch):
    #try:
        model_base = "" #base_dir + "/Models/RealFinalSep/"
        model_name = "MC_SignModel_b" + str(batch_size) + "_p_" + str(
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

            # Make the model
            model = Sequential()

            # Base Conv layer
            model.add(Conv2D(64, kernel_size=(6,6), strides=(2, 2),
                             padding='same',
                             input_shape=(75, 75, 1)))
            model.add(Activation('relu'))
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            model.add(Dropout(0.25))

            model.add(
                Conv2D(128, (3,3), strides=(1, 1),
                       padding='same'))
            #model.add(BatchNormalization())
            model.add(Activation('relu'))
            #model.add(Activation('relu'))
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            model.add(Dropout(0.25))
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
            model.add(Dense(2, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['acc'])
            model.summary()
            # Makes it only use
            #model.fit_generator(generator=batchYielder(), steps_per_epoch=int(
            #    np.floor(((number_of_training / (frac_per_epoch * batch_size)))))
            #                    , epochs=num_epochs,
            #                    verbose=2, validation_data=(y, y_label),
            #                    callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])
            model.fit(x=y, y=y_label, batch_size=batch_size, epochs=500, verbose=2, validation_split=0.2, callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])
            predictions = model.predict(y, batch_size=64)
            test_pred = model.predict(images_test, batch_size=64)
            print(roc_auc_score(y_label, predictions))
            print(roc_auc_score(disp_test, test_pred))
            K.clear_session()
            tf.reset_default_graph()

    #except Exception as e:
    #    print(e)
    #    K.clear_session()
    #    tf.reset_default_graph()
    #    pass

def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer):
    #try:
    model_base = ""# base_dir +"/" # + "/Models/FinalSourceXY/test/test/"
    model_name = "MC_OneOutputPoolSIGN" + "_drop_" + str(dropout_layer)
    if not os.path.isfile(model_base + model_name + ".csv"):
        csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
        #reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=70, min_lr=0.001)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, verbose=0, mode='auto')

        model_checkpointy = keras.callbacks.ModelCheckpoint(model_base + "Y_" + desc + model_name + ".h5", monitor='val_loss', verbose=0,
                                                            save_best_only=True, save_weights_only=False, mode='auto', period=1)
        # Make the model
        inp = keras.layers.Input((75,75,1))
        # Block - conv
        y = Conv2D(64, 8, 8, border_mode='same', subsample=[4,4], name='yConv1')(inp)
        #y = BatchNormalization()(y)
        y = ELU()(y)
        # Block - conv
        y = Conv2D(128, 5, 5, border_mode='same', subsample=[2,2], name='yConv2')(y)
        # Block - conv
        #y = BatchNormalization()(y)
        y = ELU()(y)
        y = MaxPooling2D(padding='same')(y)
        y = Dropout(dropout_layer)(y)

        y = Conv2D(256, 5, 5, border_mode='same', subsample=[2,2], name='yConv3')(y)
        #y = BatchNormalization()(y)
        y = ELU()(y)
        y = Conv2D(512, 5, 5, border_mode='same', subsample=[2,2], name='yConv7')(y)
        #y = BatchNormalization()(y)
        y = ELU()(y)

        #y = Dropout(dropout_layer)(y)

        #y = Conv2D(256, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='yConv10')(y)
        #y = Conv2D(512, 2, 2, border_mode='same', subsample=[2,2], activation='elu', name='yConv11')(y)

        # Block - flatten
        # Block - flatten
        y = Flatten()(y)
        y = Dropout(dropout_layer)(y)
        y = ELU()(y)

        # Block - bring in pointing zd
        # Block - fully connected
        y = Dense(dense_neuron, activation='elu', name='yFC1')(y)
        y = Dropout(0.3)(y)
        y = ELU()(y)
        y = Dense(dense_neuron, activation='elu', name='yFC2')(y)
        y = Dropout(0.3)(y)
        y = ELU()(y)

        # Block - conv
        x = Conv2D(64, 8, 8, border_mode='same', activation='relu', subsample=[4,4], name='Conv1')(inp)
        #x = BatchNormalization()(x)
        #x = ELU()(x)
        # Block - conv
        x = Conv2D(128, 5, 5, border_mode='same', activation='relu', subsample=[2,2], name='Conv2')(x)
        #x = BatchNormalization()(x)
        #x = ELU()(x)
        # Block - conv
        x = MaxPooling2D(padding='same')(x)
        x = Dropout(dropout_layer)(x)

        x = Conv2D(256, 5, 5, border_mode='same', activation='relu', subsample=[2,2], name='Conv3')(x)
        #x = BatchNormalization()(x)
        #x = ELU()(x)
        x = Conv2D(512, 5, 5, border_mode='same', activation='relu', subsample=[2,2], name='Conv7')(x)
        #x = BatchNormalization()(x)
        #x = ELU()(x)

        #x = Dropout(dropout_layer)(x)

        #x = Conv2D(128, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv10')(x)
        #x = Conv2D(512, 2, 2, border_mode='same', subsample=[2,2], activation='elu', name='Conv11')(x)

        # Block - flatten
        # Block - flatten
        x = Flatten()(x)
        x = Dropout(dropout_layer)(x)
        #x = ELU()(x)
        # Block - fully connected
        x = Dense(dense_neuron, activation='relu', name='FC1')(x)
        x = Dropout(0.3)(x)
        #x = ELU()(x)
        x = Dense(dense_neuron, activation='relu', name='FC2')(x)
        x = Dropout(0.3)(x)
        #x = ELU()(x)
        x_out = Dense(2, name="x_out", activation='softmax')(x)
        y_out = Dense(1, name="y_out", activation='linear')(y)

        #merged_out = keras.layers.merge([x, y])
        #combined_out = Dense(2, name="combined_out")(merged_out)

        model = keras.models.Model(inp, x_out)
        # Block - output
        model.summary()
        adam = keras.optimizers.adam(lr=0.001)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        #model.fit_generator(generator=batchYielder(), steps_per_epoch=np.floor(((number_of_training / batch_size))), epochs=epoch,
        #                    verbose=2, validation_data=(y, y_label), callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])
        #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))
        model.fit(x=y_train, y=y_label, batch_size=batch_size, epochs=500, verbose=2, validation_split=0.2, callbacks=[early_stop, model_checkpointy, csv_logger])

        predictions = model.predict(y_train, batch_size=64)
        test_pred = model.predict(images_test_y, batch_size=64)
        print(roc_auc_score(y_label, predictions))
        print(roc_auc_score(disp_test, test_pred))
        predictions_x = predictions.reshape(-1,)
        predictions_y = predictions.reshape(-1,)
        test_pred_y = test_pred.reshape(-1,)
        test_pred_x = test_pred.reshape(-1,)

        #Loss Score so can tell which one it is
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
    create_model2(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer)
