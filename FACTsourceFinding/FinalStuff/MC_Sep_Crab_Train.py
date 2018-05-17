import os
# to force on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
import h5py
from fact.io import read_h5py
import yaml
import os
import keras
import numpy as np
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, LeakyReLU, Reshape, BatchNormalization, Conv2D, MaxPooling2D
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
from fact.coordinates import equatorial_to_camera, camera_to_equatorial
from astropy.coordinates.angle_utilities import angular_separation

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/git-thesis/thesis'

# Total fraction to use per epoch of training data, need inverse
frac_per_epoch = 1
num_epochs = 1000*frac_per_epoch

path_mc_images = "/run/media/jacob/SSD/Rebinned_5_MC_Gamma_BothSource_Images.h5"
path_proton_images = "/run/media/jacob/SSD/Rebinned_5_MC_Proton_BothTracking_Images.h5"
np.random.seed(0)


def get_events_near_crab(path_crab_image, len_images):
    crabdf = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/Mrk501_precuts.hdf5", key="events", last=len_images, columns=['dec_prediction', 'ra_prediction'])
    # now go and get the ones with an ra or dec within 0.16 degrees of the source
    # Source pos in sky AltAz
    coord = SkyCoord.from_name("MRK501")
    center_ra = coord.ra.deg
    center_dec = coord.dec.deg

    # Now have it in ra and dec, convert ra of FACT to degree from hourangle
    #crabdf['ra_prediction']
    center_ra = crabdf['ra_prediction'].mean()
    center_dec = crabdf['dec_prediction'].mean()

    # Get the events that have the right ra and dec
    # Give to Gamma/Hadron sep, use Mrk 501 because brighter source
    df_event = crabdf.index[(crabdf['ra_prediction'] < 1.0025*center_ra) & (crabdf['ra_prediction'] > 0.995*center_ra) & (crabdf['dec_prediction'] < 1.0025*center_dec) & (crabdf['dec_prediction'] > 0.995*center_ra)]
    print(df_event)

    # Now get the ones close to the actual thing

    return 0

get_events_near_crab("/run/media/jacob/WDRed8Tb1/Rebinned_5_Mrk501_HALFMILSTDDEV_Images.h5", 5000000)
exit()
with h5py.File(path_mc_images, 'r') as f:
    with h5py.File(path_proton_images, 'r') as f2:
        # Get some truth data for now, just use Crab images
        items = len(f2["Image"])
        images = f['Image'][0:10000]
        images_false = f2['Image'][0:10000]
        temp_train = []
        temp_test = []
        tmp_test_label = []
        tmp_train_label = []
        for batcher in range(10000):
            # Mix the datasets
            if batcher < 0.8*len(images):
                temp_train.append(images[batcher])
                temp_train.append(images_false[batcher])
                tmp_train_label.append([0,1])
                tmp_train_label.append([1,0])
            else:
                temp_test.append(images[batcher])
                temp_test.append(images_false[batcher])
                tmp_test_label.append([0,1])
                tmp_test_label.append([1,0])

        test_labels = np.asarray(tmp_test_label)
        train_labels = np.asarray(tmp_train_label)
        test_dataset = np.asarray(temp_test)
        validation_dataset = np.asarray(temp_train)
        #validating_dataset = np.concatenate([images, images_false], axis=0)
        #print(validating_dataset.shape)
        #labels = np.array([True] * (len(images)) + [False] * len(images_false))
        np.random.seed(0)
        #rng_state = np.random.get_state()
        #np.random.shuffle(validating_dataset)
        #np.random.set_state(rng_state)
        #np.random.shuffle(labels)
        #test_dataset = validating_dataset[-int(0.8*len(validating_dataset)):]
        #test_labels = labels[-int(0.8*len(labels)):]
        #test_labels = (np.arange(2) == test_labels[:, None]).astype(np.float32)
        #validating_dataset = validating_dataset[0:int(0.8*len(validating_dataset))]
        #labels = labels[0:int(0.8*len(labels))]
        #validation_labels = (np.arange(2) == labels[:, None]).astype(np.float32)
        y = validation_dataset
        y_label = train_labels

        print("Finished getting data")

from sklearn.metrics import roc_auc_score

def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, frac_per_epoch):
    try:
        model_base = "" #base_dir + "/Models/RealFinalSep/"
        model_name = "MC_SepNoGenNoShuffle_b" + str(batch_size) + "_p_" + str(
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
            model.add(LeakyReLU())
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            model.add(Dropout(0.25))

            model.add(
                Conv2D(128, (3,3), strides=(1, 1),
                       padding='same'))
            #model.add(BatchNormalization())
            model.add(LeakyReLU())
            #model.add(Activation('relu'))
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            model.add(Dropout(0.25))
            '''
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
            '''
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
            model.fit(x=y, y=y_label, batch_size=batch_size, epochs=num_epochs, verbose=2, validation_split=0.2, callbacks=[early_stop, csv_logger, reduceLR, model_checkpoint])
            predictions = model.predict(y, batch_size=64)
            test_pred = model.predict(test_dataset, batch_size=64)
            print(roc_auc_score(y_label, predictions))
            print(roc_auc_score(test_labels, test_pred))
            K.clear_session()
            tf.reset_default_graph()

    except Exception as e:
        print(e)
        K.clear_session()
        tf.reset_default_graph()
        pass


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
    batch_size = np.random.randint(batch_sizes[0], batch_sizes[1])
    num_conv = np.random.randint(num_conv_layers[0], num_conv_layers[1])
    num_dense = np.random.randint(num_dense_layers[0], num_dense_layers[1])
    patch_size = patch_sizes[np.random.randint(0, 2)]
    num_pooling_layer = np.random.randint(num_pooling_layers[0], num_pooling_layers[1])
    dense_neuron = np.random.randint(num_dense_neuron[0], num_dense_neuron[1])
    conv_neurons = np.random.randint(num_conv_neurons[0], num_conv_neurons[1])
    create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, frac_per_epoch)
