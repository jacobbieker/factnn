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

architecture = 'manjar'

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
    az_error = K.mean(K.abs(atan2(K.sin(y_true[:,1] - y_pred[:,1]), K.cos(y_true[:,1] - y_pred[:,1]))))
    zd_error = K.mean(K.square(y_pred[:,0] - y_true[:,0]), axis=-1)
    return az_error + zd_error

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
number_of_training = 531000*(0.6)
number_of_testing = 531000*(0.2)
number_validate = 531000*(0.2)
optimizers = ['same']
epoch = 500

path_mc_images = base_dir + "/Rebinned_5_MC_diffuse_BothSource_Images.h5"
#path_mrk501 = "/run/media/jacob/WDRed8Tb1/dl2_theta/Mrk501_precuts.hdf5"

#mrk501 = read_h5py(path_mrk501, key="events", columns=["event_num", "night", "run_id", "source_x_prediction", "source_y_prediction"])
#mc_image = read_h5py_chunked(path_mc_images, key='events', columns=['Image', 'Event', 'Night', 'Run'])

def metaYielder():
    gamma_anteil = 1
    gamma_count = int(round(number_of_training*gamma_anteil))

    return gamma_anteil, gamma_count

model = Sequential()
optimizer = 'same'
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
model.add(Dense(2, activation='linear'))
model.compile(optimizer='adam', loss=rmse_360_2, metrics=['mae', 'mse'])
from keras.utils.vis_utils import plot_model
plot_model(model, to_file="VGG_Model.png")
exit()

with h5py.File(path_mc_images, 'r') as f:
    gamma_anteil, gamma_count = metaYielder()
    # Get some truth data for now, just use Crab images
    images = f['Image'][0:-1]
    images_source_zd = f['Source_Zd'][0:-1]
    images_source_az = f['Source_Az'][0:-1]
    # now convert to radians
    images_source_az = np.deg2rad(images_source_az)
    images_source_zd = np.deg2rad(images_source_zd)
    np.random.seed(0)
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(images_source_az)
    np.random.set_state(rng_state)
    np.random.shuffle(images_source_zd)
    images = images[0:int(0.5*len(images))]
    images_source_az = images_source_az[0:int(0.5*len(images_source_az))]
    images_source_zd = images_source_zd[0:int(0.5*len(images_source_zd))]
    # Now remove the last 20% for testing later
    # images_point_az = f['Az_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    # images_point_zd = f['Zd_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
    # source_x, source_y = horizontal_to_camera(
    #     zd=images_source_zd, az=images_source_az,
    #     az_pointing=images_point_az, zd_pointing=images_point_zd
    # )

    y = images
    y_label = np.column_stack((images_source_zd, images_source_az))
    print(y_label[:,1][2])
    print(y_label[:,0][2])
    print(images_source_zd[0])
    print(images_source_az[0])
    print(y_label[0])
    print(y_label.shape)
    #exit(1)
    print("Finished getting data")


def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer):
    try:
        model_base = base_dir + "/Models/FinalDisp/"
        model_name = "MC_vggAzNoGen_b" + str(batch_size) +"_p_" + str(patch_size) + "_drop_" + str(dropout_layer) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + "_denseN_" + str(dense_neuron) + "_numDense_" + str(num_dense) + "_convN_" + \
                     str(conv_neurons) + "_opt_" + str(optimizer)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=70, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "{val_loss:.3f}_" + model_name + ".h5", monitor='val_loss', verbose=0,
                                                               save_best_only=True, save_weights_only=False, mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=80, verbose=0, mode='auto')

            def batchYielder():
                with h5py.File(path_mc_images, 'r') as f:
                    items = list(f.items())[1][1].shape[0]
                    items = items - number_of_testing - number_validate
                    times_train_in_items = int(np.floor(items / number_of_training))
                    print(times_train_in_items)
                    length_dataset = len(f['Image'])
                    if items > number_of_training:
                        items = number_of_training
                    section = 0
                    offset = int(section * items)
                    image = f['Image'][offset:int(offset + items)]
                    image = np.rot90(image, 1, axes=(1,2))
                    source_zd = f['Source_Zd'][offset:int(offset + items)]
                    source_az = f['Source_Az'][offset:int(offset + items)]
                    source_az = (source_az + 360) % 360
                    source_y = np.deg2rad(source_az)
                    source_x = np.deg2rad(source_zd)
                    while True:
                        batch_num = 0
                        section = section % times_train_in_items

                        rng_state = np.random.get_state()
                        #np.random.shuffle(image)
                        np.random.set_state(rng_state)
                        #np.random.shuffle(source_y)
                        np.random.set_state(rng_state)
                        #np.random.shuffle(source_x)
                        #np.random.set_state(rng_state)
                        #np.random.shuffle(point_zd)
                        #np.random.set_state(rng_state)
                        #np.random.shuffle(point_az)
                        # Roughly 5.6 times more simulated Gamma events than proton, so using most of them
                        while (batch_size) * (batch_num + 1) < len(image):
                            # Get some truth data for now, just use Crab images
                            images = image[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            #images_source_zd = source_zd[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            #images_source_az = source_az[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            #images_point_az = point_az[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            #images_point_zd = point_zd[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            images_source_x = source_x[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            images_source_y = source_y[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            x = np.flip(images, axis=2)
                            x_label = np.column_stack((images_source_x, images_source_y)) #np.asarray([images_source_x, images_source_y]).reshape((-1, 2))
                            #print(x_label.shape)
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
            model.add(Dense(2, activation='linear'))
            model.compile(optimizer='adam', loss=rmse_360_2, metrics=['mae', 'mse'])
            from keras.utils.vis_utils import plot_model
            plot_model(model, to_file="VGG_Model.png")
            exit()

            model.fit(x=y, y=y_label, batch_size=batch_size, epochs=epoch, validation_split=0.2, callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])


            K.clear_session()
            tf.reset_default_graph()

    except Exception as e:
        print(e)
        K.clear_session()
        tf.reset_default_graph()
        pass


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
