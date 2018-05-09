import os
# to force on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
number_of_training = 300000*(0.6)
number_of_testing = 300000*(0.2)
number_validate = 300000*(0.2)
optimizers = ['same']
epoch = 300
frac_test = 1
frac_train = 1

path_mc_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_crab_preprocessed_images.h5"
path_crab = "/run/media/jacob/SSD/open_crab_sample_analysis/build/crab_precuts.hdf5"

crab = read_h5py(path_crab, key="events", columns=["event_num", "night", "run_id", "source_x_prediction", "source_y_prediction"])
#mc_image = read_h5py_chunked(path_mc_images, key='events', columns=['Image', 'Event', 'Night', 'Run'])

def metaYielder():
    gamma_anteil = 1
    gamma_count = int(round(number_of_training*gamma_anteil))

    return gamma_anteil, gamma_count


with h5py.File(path_mc_images, 'r') as f:
    gamma_anteil, gamma_count = metaYielder()
    if os.path.isfile(base_dir + "/crab_testing_images.npy"):
        # Just load from that
        image = np.load(base_dir + "/crab_testing_images.npy")
        source_label_x = np.load(base_dir + "/crab_testing_x.npy")
        source_label_y = np.load(base_dir + "/crab_testing_y.npy")
    else:
        if not os.path.isfile("crab_precut_testing.p"):
            raw_event_nums = np.asarray(f['Event'][0:2*int(np.floor((gamma_anteil*number_of_testing)))])
            raw_nights = np.asarray(f['Night'][0:2*int(np.floor((gamma_anteil*number_of_testing)))])
            raw_run_ids = np.asarray(f['Run'][0:2*int(np.floor((gamma_anteil*number_of_testing)))])

            #raw_df = pd.DataFrame(data=[raw_images, raw_event_nums, raw_nights, raw_run_ids])
            #raw_df.columns = ["Image", "Event", "Night", "Run"]
            #print(raw_df)

            # Get some truth data for now, just use Mrk501 images
            #images = crab['Image'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
            #images_source_zd = crab['source_x_prediction'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
            #images_source_az = crab['source_y_prediction'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
            # now get the photon stream data
            #event_nums = crab['event_num'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
            #nights = crab['night'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
            #run_ids = crab['run_id'][-int(np.floor((gamma_anteil*number_of_testing))):-1]

            night_images = []
            source_label_x = []
            source_label_y = []
            indicies_that_work = []

            for index, night in enumerate(raw_nights):
                night_index = index
                raw_run_id = raw_run_ids[night_index]
                raw_event_num = raw_event_nums[night_index]

                #print(night)
                #print(raw_run_id)
                #print(raw_event_num)

                #print("Now Both")
                #print("Night: ")
                #print(night)
                testing = crab.loc[(crab['event_num'] == raw_event_num) & (crab['run_id'] == raw_run_id) & (crab['night'] == night)]
                if not testing.empty:
                    indicies_that_work.append(index)
                    #print(testing)
                exact_position = crab.loc[(crab['night'] == night) & (crab['run_id'] == raw_run_id) & (crab['event_num'] == raw_event_num)]
                #print(exact_position)
                # now get range of nights from run_id and event_num
                if not exact_position.empty:
                    # got the exact event now, need the image
                    night_images.append(f['Image'][index])
                    source_label_x.append(exact_position['source_x_prediction'].values)
                    source_label_y.append(exact_position['source_y_prediction'].values)

            # After done with that convert to Numpy
            night_images = np.asarray(night_images)
            source_label_x = np.asarray(source_label_x)
            source_label_y = np.asarray(source_label_y)
            with open("crab_precut_testing.p", "wb") as path_store:
                pickle.dump(indicies_that_work, path_store)
                #images_point_az = f['Az_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
                #images_point_zd = f['Zd_deg'][-int(np.floor((gamma_anteil*number_of_testing))):-1]
                #images_source_az = (-1.*images_source_az + 540) % 360
                #source_x, source_y = horizontal_to_camera(
                #    zd=images_source_zd, az=images_source_az,
                #    az_pointing=images_point_az, zd_pointing=images_point_zd
                #)
        else:
            # It does exits
            with open("crab_precut_testing.p", "rb") as path_store:
                indicies_that_work = pickle.load(path_store)
            total_testing = len(indicies_that_work)
            np.random.shuffle(indicies_that_work)
            indicies_that_work = indicies_that_work[0:int(total_testing/frac_test)]
            max_index = np.max(indicies_that_work)
            raw_event_nums = np.asarray(f['Event'][0:max_index+1])
            raw_nights = np.asarray(f['Night'][0:max_index+1])
            raw_run_ids = np.asarray(f['Run'][0:max_index+1])
            night_images = []
            source_label_x = []
            source_label_y = []
            for index in indicies_that_work:
                night_index = index
                night = raw_nights[index]
                raw_run_id = raw_run_ids[night_index]
                raw_event_num = raw_event_nums[night_index]
                #print(testing)
                exact_position = crab.loc[(crab['night'] == night) & (crab['run_id'] == raw_run_id) & (crab['event_num'] == raw_event_num)]
                #print(exact_position)
                # now get range of nights from run_id and event_num
                if not exact_position.empty:
                    # got the exact event now, need the image
                    night_images.append(f['Image'][index])
                    source_label_x.append(exact_position['source_x_prediction'].values)
                    source_label_y.append(exact_position['source_y_prediction'].values)

            # After done with that convert to Numpy
            night_images = np.asarray(night_images)
            source_label_x = np.asarray(source_label_x)
            source_label_y = np.asarray(source_label_y)


        y = night_images
        np.save(base_dir + "/crab_testing_images.npy", night_images)
        np.save(base_dir + "/crab_testing_x.npy", source_label_x)
        np.save(base_dir + "/crab_testing_y.npy", source_label_y)
    # Now convert to this camera's coordinates
    source_label_x += 180.975 # shifts everything to positive
    source_label_y += 185.25 # shifts everything to positive
    source_label_x = source_label_x / 4.94 # Ratio between the places
    source_label_y = source_label_y / 4.826 # Ratio between y in original and y here
    y_label = np.asarray([source_label_x, source_label_y]).reshape((-1, 2))
    print(y_label.shape)
    print("Finished getting data")

exit(1)

def create_model(batch_size, patch_size, dropout_layer, num_dense, num_conv, num_pooling_layer, dense_neuron, conv_neurons, optimizer):
    try:
        model_base = base_dir + "/Models/Disp2/"
        model_name = "MC_holchTrueSource_b" + str(batch_size) +"_p_" + str(patch_size) + "_drop_" + str(dropout_layer) \
                     + "_conv_" + str(num_conv) + "_pool_" + str(num_pooling_layer) + "_denseN_" + str(dense_neuron) + "_numDense_" + str(num_dense) + "_convN_" + \
                     str(conv_neurons) + "_opt_" + str(optimizer)
        if not os.path.isfile(model_base + model_name + ".csv"):
            csv_logger = keras.callbacks.CSVLogger(model_base + model_name + ".csv")
            reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=20, min_lr=0.001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(model_base + "{val_loss:.3f}_" + model_name + ".h5", monitor='val_loss', verbose=0,
                                                               save_best_only=True, save_weights_only=False, mode='auto', period=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='auto')

            def batchYielder():
                with h5py.File(path_mc_images, 'r') as f:
                    items = list(f.items())[1][1].shape[0]
                    items = items - number_of_testing - number_validate
                    times_train_in_items = int(np.floor(items / number_of_training))
                    length_dataset = len(f['Image'])
                    if items > number_of_training:
                        items = number_of_training
                    section = 0
                    section = section % times_train_in_items
                    offset = int(section * items)
                    if os.path.isfile(base_dir + "/crab_training_images.npy"):
                        # Just load from that
                        image = np.load(base_dir + "/crab_training_images.npy")
                        source_label_x = np.load(base_dir + "/crab_training_x.npy")
                        source_label_y = np.load(base_dir + "/crab_training_y.npy")
                    else:
                        if not os.path.isfile("crab_precut_training.p"):
                            raw_event_nums = np.asarray(f['Event'][2*int(np.floor((gamma_anteil*number_of_testing))):2*int(np.floor((gamma_anteil*number_of_testing)))+2*int(np.floor((gamma_anteil*number_of_training)))])
                            raw_nights = np.asarray(f['Night'][2*int(np.floor((gamma_anteil*number_of_testing))):2*int(np.floor((gamma_anteil*number_of_testing)))+2*int(np.floor((gamma_anteil*number_of_training)))])
                            raw_run_ids = np.asarray(f['Run'][2*int(np.floor((gamma_anteil*number_of_testing))):2*int(np.floor((gamma_anteil*number_of_testing)))+2*int(np.floor((gamma_anteil*number_of_training)))])
                            night_images = []
                            source_label_x = []
                            source_label_y = []
                            indicies_that_work = []

                            for index, night in enumerate(raw_nights):
                                night_index = index
                                raw_run_id = raw_run_ids[night_index]
                                raw_event_num = raw_event_nums[night_index]

                                #print(night)
                                #print(raw_run_id)
                                #print(raw_event_num)

                                #print("Now Both")
                                #print("Night: ")
                                #print(night)
                                #testing = crab.loc[(crab['event_num'] == raw_event_num) & (crab['run_id'] == raw_run_id) & (crab['night'] == night)]
                                #if not testing.empty:
                                #    print(testing)
                                exact_position = crab.loc[(crab['night'] == night) & (crab['run_id'] == raw_run_id) & (crab['event_num'] == raw_event_num)]
                                #print(exact_position)
                                # now get range of nights from run_id and event_num
                                if not exact_position.empty:
                                    # Need this extra part so that the index in the actual Image thing is correct
                                    indicies_that_work.append(int(2*int(np.floor((gamma_anteil*number_of_testing)))+index))
                                    # got the exact event now, need the image
                                    night_images.append(f['Image'][int(2*int(np.floor((gamma_anteil*number_of_testing)))+index)])
                                    source_label_x.append(exact_position['source_x_prediction'].values)
                                    source_label_y.append(exact_position['source_y_prediction'].values)
                            with open("crab_precut_training.p", "wb") as path_store:
                                pickle.dump(indicies_that_work, path_store)
                        else:
                            # It does exits
                            with open("crab_precut_training.p", "rb") as path_store:
                                indicies_that_work = pickle.load(path_store)
                            total_testing = len(indicies_that_work)
                            np.random.shuffle(indicies_that_work)
                            indicies_that_work = indicies_that_work[0:int(total_testing/frac_test)]
                            max_index = np.max(indicies_that_work)
                            raw_event_nums = np.asarray(f['Event'][0:max_index+1])
                            raw_nights = np.asarray(f['Night'][0:max_index+1])
                            raw_run_ids = np.asarray(f['Run'][0:max_index+1])
                            night_images = []
                            source_label_x = []
                            source_label_y = []
                            for index in indicies_that_work:
                                night_index = index
                                night = raw_nights[index]
                                raw_run_id = raw_run_ids[night_index]
                                raw_event_num = raw_event_nums[night_index]
                                #print(testing)
                                exact_position = crab.loc[(crab['night'] == night) & (crab['run_id'] == raw_run_id) & (crab['event_num'] == raw_event_num)]
                                #print(exact_position)
                                # now get range of nights from run_id and event_num
                                if not exact_position.empty:
                                    # got the exact event now, need the image
                                    night_images.append(f['Image'][index])
                                    source_label_x.append(exact_position['source_x_prediction'].values)
                                    source_label_y.append(exact_position['source_y_prediction'].values)


                            # After done with that convert to Numpy
                        image = np.asarray(night_images)
                        source_label_x = np.asarray(source_label_x)
                        source_label_y = np.asarray(source_label_y)
                        np.save(base_dir + "/crab_training_images.npy", night_images)
                        np.save(base_dir + "/crab_training_x.npy", source_label_x)
                        np.save(base_dir + "/crab_training_y.npy", source_label_y)
                    # Now convert to this camera's coordinates
                    source_label_x += 180.975 # shifts everything to positive
                    source_label_y += 185.25 # shifts everything to positive
                    source_label_x = source_label_x / 4.94 # Ratio between the places
                    source_label_y = source_label_y / 4.826 # Ratio between y in original and y here
                    #image = f['Image'][offset:int(offset + items)]
                    #source_zd = f['Energy'][offset:int(offset + items)]
                    #source_az = f['Phi'][offset:int(offset + items)]
                    #point_az = f['Az_deg'][offset:int(offset + items)]
                    #point_zd = f['Zd_deg'][offset:int(offset + items)]
                    #source_az = (-1.*source_az + 540) % 360
                    #source_x, source_y = horizontal_to_camera(
                    #    zd=source_zd, az=source_az,
                    #    az_pointing=point_az, zd_pointing=point_zd
                    #)
                    while True:
                        batch_num = 0

                        rng_state = np.random.get_state()
                        np.random.shuffle(image)
                        np.random.set_state(rng_state)
                        np.random.shuffle(source_label_y)
                        np.random.set_state(rng_state)
                        np.random.shuffle(source_label_x)
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
                            images_source_x = source_label_x[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            images_source_y = source_label_y[int(batch_num*batch_size):int((batch_num+1)*batch_size)]
                            x = images
                            x_label = np.asarray([images_source_x, images_source_y]).reshape((-1,2)) #np.asarray([images_source_x, images_source_y]).reshape((-1, 2))
                            #print(x_label.shape)
                            batch_num += 1
                            yield (x, x_label)
                        section += 1

            # Make the model
            model = Sequential()

            # Base Conv layer
            model.add(Conv2D(conv_neurons, kernel_size=patch_size, strides=(1, 1),
                             activation='relu', padding=optimizer,
                             input_shape=(75, 75, 1)))
            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), padding=optimizer))

            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), padding=optimizer))

            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), padding=optimizer))

            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), padding=optimizer))

            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
            model.add(Conv2D(conv_neurons, patch_size, strides=(1, 1), activation='relu', padding=optimizer))
            model.add(MaxPooling2D(pool_size=(2, 2), padding=optimizer))

            model.add(Flatten())

            model.add(Dense(dense_neuron, activation='linear'))
            model.add(Dropout(dropout_layer))
            model.add(Dense(dense_neuron, activation='linear'))
            model.add(Dropout(dropout_layer))
            model.add(Dense(dense_neuron, activation='linear'))
            model.add(Dropout(dropout_layer))

            # Final Dense layer
            # 2 so have one for x and one for y
            model.add(Dense(2, activation=None))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
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
