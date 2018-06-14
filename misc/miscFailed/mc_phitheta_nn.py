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
from keras.layers import Dense, Dropout, Activation, Conv1D, ELU, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from fact.coordinates.utils import horizontal_to_camera

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

# y in radians and second column is az, first is zd
def rmse_360_2(y_true, y_pred):
    az_error = K.mean(K.abs(atan2(K.sin(y_true[:,1] - y_pred[:,1]), K.cos(y_true[:,1] - y_pred[:,1]))))
    zd_error = K.mean(K.square(y_pred[:,0] - y_true[:,0]), axis=-1)
    return az_error + zd_error


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
number_of_training = 10000*(0.6)
number_of_testing = 10000*(0.2)
number_validate = 10000*(0.2)
optimizer = keras.optimizers.SGD(lr=0.00001)
epoch = 300

number_bins = 500
num_classes = number_bins # Degrees in Zd or Theta by bins for classification
map_deg_to_class = np.linspace(0,90, number_bins)

path_mc_images = "/run/media/jacob/SSD/Rebinned_5_MC_diffuse_BothSource_Images.h5"
path_mc_proton = "/run/media/jacob/SSD/Rebinned_5_MC_Proton_JustImage_Images.h5"

def metaYielder():
    gamma_anteil = 1
    gamma_count = int(round(number_of_training*gamma_anteil))

    return gamma_anteil, gamma_count

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

with h5py.File(path_mc_images, 'r') as f:
    with h5py.File(path_mc_proton, 'r') as f2:
        gamma_anteil, gamma_count = metaYielder()
        # Get some truth data for now, just use Crab images
        images = f['Image'][0:-1]
        images_source_zd = f['Theta'][0:-1]
       # images_point_az = f['Az_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
       # images_point_zd = f['Zd_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
       # source_x, source_y = horizontal_to_camera(
       #     zd=images_source_zd, az=images_source_az,
       #     az_pointing=images_point_az, zd_pointing=images_point_zd
       # )
        #images_source_az = np.deg2rad(images_source_az)
        #images_source_zd = np.deg2rad(images_source_zd)
        np.random.seed(0)
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.set_state(rng_state)
        np.random.shuffle(images_source_zd)
        images = images[0:int(0.2*len(images))]
        images_source_zd = images_source_zd[0:int(0.2*len(images_source_zd))]
        # now put into classes

        transformed_images = []
        for image_one in images:
            #print(image_one.shape)
            image_one = image_one/np.sum(image_one)
            #print(np.sum(image_one))
            transformed_images.append(image_one)
            #print(np.max(image_one))
        images = np.asarray(transformed_images)

        #print(validating_dataset.shape)
        y = images
        y_label = images_source_zd
        print(images_source_zd[0])
        print(y_label[0])
        print(y_label.shape)
        print("Finished getting data")


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons):
    try:
        model_base = ""
        model_name = "MC_OnlyZdNoGenDriver_b" + str(batch_size) +"_p_" + str(patch_size) + "_drop_" + str(dropout_layer) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + "_denseN_" + str(dense_neuron) + "_numDense_" + str(num_dense) + "_convN_" + \
                     str(conv_neurons)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            #reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "{val_loss:.3f}_" + model_name + ".h5", monitor='val_loss', verbose=0,
                                                               save_best_only=False, save_weights_only=False, mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='auto')

            gamma_anteil, gamma_count = metaYielder()
            # Make the model
            model = Sequential()
            # Preprocess incoming data, centered around zero with small standard deviation
            #model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(75, 75, 1)))
            # Block - conv
            model.add(Conv2D(16, 8, 8, border_mode='same', subsample=[4,4], activation='elu', name='Conv1', input_shape=(75,75,1)))
            # Block - conv
            model.add(Conv2D(35, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv2'))
            # Block - conv
            model.add(Conv2D(64, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv3'))
            model.add(Conv2D(128, 5, 5, border_mode='same', subsample=[2,2], activation='elu', name='Conv4'))
            # Block - flatten
            model.add(Flatten())
            model.add(Dropout(dropout_layer))
            model.add(ELU())
            # Block - fully connected
            model.add(Dense(dense_neuron, activation='elu', name='FC1'))
            model.add(Dropout(0.5))
            model.add(ELU())
            model.add(Dense(dense_neuron, activation='elu', name='FC2'))
            model.add(Dropout(0.5))
            model.add(ELU())
            # Block - output
            model.add(Dense(1, name='output'))
            model.summary()
            adam = keras.optimizers.adam(lr=0.0001)
            model.compile(optimizer=adam, loss='mse', metrics=['mae'])

            model.fit(x=y, y=y_label, batch_size=batch_size, epochs=epoch, validation_split=0.2, callbacks=[early_stop, csv_logger, model_checkpoint])

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
