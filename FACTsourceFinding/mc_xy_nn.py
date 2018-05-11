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
from numpy import savetxt, loadtxt, round, zeros, sin, cos, arctan2, clip, pi, tanh, exp, arange, dot, outer, array, shape, zeros_like, reshape, mean, median, max, min
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera

# Returns encoded angle using specified method ("binned","scaled","cossin","gaussian")
def encode_angle(angle, method):
    if method == "binned": # 1-of-500 encoding
        X = zeros(500)
        X[int(round(250*(angle/pi + 1)))%500] = 1
    elif method == "gaussian": # Leaky binned encoding
        X = array([i for i in range(500)])
        idx = 250*(angle/pi + 1)
        X = exp(-pi*(X-idx)**2)
    elif method == "scaled": # Scaled to [-1,1] encoding
        X = array([angle/pi])
    elif method == "cossin": # Oxinabox's (cos,sin) encoding
        X = array([cos(angle),sin(angle)])
    else:
        pass
    return X

# Returns decoded angle using specified method
def decode_angle(X, method):
    if method == "binned" or method == "gaussian": # 1-of-500 or gaussian encoding
        M = max(X)
        for i in range(len(X)):
            if abs(X[i]-M) < 1e-5:
                angle = pi*i/250 - pi
                break
    return angle

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



architecture = 'manjar'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

# Hyperparameters

batch_sizes = [16, 64, 512]
patch_sizes = [(2, 2), (3, 3), (5, 5), (4, 4)]
dropout_layers = [0.0, 1.0]
num_conv_layers = [3, 10]
num_dense_layers = [0, 10]
num_conv_neurons = [8,256]
num_dense_neuron = [8,512]
num_pooling_layers = [0, 2]
num_runs = 500
number_of_training = 150000*(0.6)
number_of_testing = 150000*(0.2)
number_validate = 150000*(0.2)
optimizer = 'adam'
epoch = 300

path_mc_images = base_dir + "/Rebinned_5_MC_diffuse_BothSource_Images.h5"

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
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(source_x)
    np.random.set_state(rng_state)
    np.random.shuffle(source_y)
    images = images[0:int(0.8*len(images))]
    source_x = source_x[0:int(0.8*len(source_x))]
    source_y = source_y[0:int(0.8*len(source_y))]

    # Now convert to this camera's coordinates
    source_x += 180.975 # shifts everything to positive
    source_y += 185.25 # shifts everything to positive
    source_x = source_x / 4.94 # Ratio between the places
    source_y = source_y / 4.826 # Ratio between y in original and y here
    y = images #np.flip(images, axis=2)
    print(images.shape)
    print(source_x[0])
    print(source_y[0])
    y_label = np.column_stack((source_x, source_y))
    print(y_label[0])
    print(y_label.shape)
    print("Finished getting data")


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons):
    try:
        model_base = base_dir + "/Models/FinalDisp/"
        model_name = "MC_XYNoGen_b" + str(batch_size) +"_p_" + str(patch_size) + "_drop_" + str(dropout_layer) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + "_denseN_" + str(dense_neuron) + "_numDense_" + str(num_dense) + "_convN_" + \
                     str(conv_neurons) + "_opt_" + str(optimizer)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "{val_loss:.3f}_" + model_name + ".h5", monitor='val_loss', verbose=0,
                                                               save_best_only=True, save_weights_only=False, mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='auto')

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
                    offset = int(section * items)
                    image = f['Image'][offset:int(offset + items)]
                    image = image
                    source_zd = f['Source_Zd'][offset:int(offset + items)]
                    source_az = f['Source_Az'][offset:int(offset + items)]
                    #source_az = (source_az + 360) % 360
                    source_az = np.deg2rad(source_az)
                    source_zd = np.deg2rad(source_zd)
                    while True:
                        batch_num = 0
                        section = section % times_train_in_items

                        #point_az = f['Az_deg'][offset:int(offset + items)]
                        #point_zd = f['Zd_deg'][offset:int(offset + items)]

                        #source_x, source_y = horizontal_to_camera(
                        #    zd=source_zd, az=source_az,
                        #    az_pointing=point_az, zd_pointing=point_zd
                        #)

                        rng_state = np.random.get_state()
                        np.random.set_state(rng_state)
                        np.random.shuffle(image)
                        np.random.set_state(rng_state)
                        np.random.shuffle(source_az)
                        np.random.set_state(rng_state)
                        np.random.shuffle(source_zd)
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
                            images_source_x = source_zd[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            images_source_y = source_az[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            x = images
                            x_label = np.column_stack((images_source_x, images_source_y))
                            #print(x_label.shape)
                            batch_num += 1
                            yield (x, x_label)
                        section += 1

            gamma_anteil, gamma_count = metaYielder()
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

            model.add(Flatten())

            # Now do the dense layers
            for i in range(num_dense):
                model.add(Dense(dense_neuron, activation='relu'))
                model.add(Dropout(dropout_layer))

            # Final Dense layer
            # 2 so have one for x and one for y
            model.add(Dense(2, activation='linear'))
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            model.fit(x=y, y=y_label, batch_size=batch_size, epochs=epoch, validation_split=0.2, callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])


            K.clear_session()
            tf.reset_default_graph()

    except Exception as e:
        print(e)
        K.clear_session()
        tf.reset_default_graph()
        pass


for i in range(num_runs):
    dropout_layer = np.round(np.random.uniform(0.0, 1.0), 3)
    batch_size = np.random.randint(batch_sizes[0], batch_sizes[1])
    num_conv = np.random.randint(num_conv_layers[0], num_conv_layers[1])
    num_dense = np.random.randint(num_dense_layers[0], num_dense_layers[1])
    patch_size = patch_sizes[np.random.randint(0, 3)]
    num_pooling_layer = np.random.randint(num_pooling_layers[0], num_pooling_layers[1])
    dense_neuron = np.random.randint(num_dense_neuron[0], num_dense_neuron[1])
    conv_neurons = np.random.randint(num_conv_neurons[0], num_conv_neurons[1])
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons)
