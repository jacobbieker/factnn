import keras
import numpy as np
import h5py
import fact
from fact.io import read_h5py

'''
Set of generators for the different neural networks

Goal is to have: One that gives the individual images and source
 One that gives it in ra and dec in the skymap
 One that gives it with aux information, like length, cog, etc
 One that gives it bundled, like 5000 images at a time or so
 
'''


def RaDecGenerator(source_file, num_images_per_output, batch_size, dim, n_channels, truths, shuffle, seed=0,
                   source=None, trigger=4):
    '''
    Here, need to get the ra and dec from the files, and the timing information, probably from the night and run_id
    since it is not in the HDF5 files anymore

    Take the ra and dec of the true source from astropy most likely, see if it can train on that

    See if can get the angular size of the source as well in the skymap -> use for UNET finding such sources?


    :param source_file:
    :param num_images_per_output:
    :param batch_size:
    :param dim:
    :param n_channels:
    :param truths:
    :param shuffle:
    :param seed:
    :return:
    '''

    with h5py.File(source_file) as f:
        items = list(f.items())[0][1].shape[0]
        source_one_images = []
        source_pos_one = []
        tmp_arr = np.zeros((46, 45, 1))
        k = 1
        batch_index = 0
        source_truth = f['Source_Position'][0]
        for i in range(0, items):
            if not np.array_equal(f['Source_Position'][i], source_truth) and f['Trigger'][i] == trigger:
                source_truth_two = f['Source_Position'][i]
        for i in range(0, items):
            if np.array_equal(f['Source_Position'][i], source_truth) and f['Trigger'][i] == trigger:
                # arrays are the same, add to source images and ones
                # Randomly flip the array twice to augment training

                if (k % num_images_per_output) == 0:
                    # Add to temp image
                    tmp_arr += f['Image'][i]
                    k += 1
                else:
                    # Hit the 5000 cap I need
                    # print("5000 Hit")
                    tmp_arr = f['Image'][i]
                    # REsize correctly
                    if tmp_arr.shape != dim:
                        # Because it won't ever be smaller, only have to chech which dims are smaller than required
                        if tmp_arr.shape[0] < dim[0]:
                            tmp_arr = np.c_[tmp_arr.reshape((46, 45)), np.zeros((46, (dim[0] - tmp_arr.shape[0])))]
                        if tmp_arr.shape[1] < dim[1]:
                            tmp_arr = np.r_[tmp_arr, np.zeros(((dim[1] - tmp_arr.shape[1]), dim[0]))]
                    tmp_arr = tmp_arr.reshape((dim[1], dim[0], 1))
                    # tmp_arr.resize((48,48,1))
                    source_one_images.append(tmp_arr)
                    source_arr = f['Source_Position'][i]
                    if source_arr.shape[0] < dim[0]:
                        source_arr = np.c_[source_arr.reshape((46, 45)), np.zeros((46, (dim[0] - source_arr.shape[0])))]
                    if source_arr.shape[1] < dim[1]:
                        source_arr = np.r_[source_arr, np.zeros(((dim[1] - source_arr.shape[1]), dim[0]))]
                    source_arr = source_arr.reshape((dim[1], dim[0], 1))
                    # source_arr.resize((48,48,1), refcheck=False)
                    source_pos_one.append(source_arr)
                    tmp_arr = np.zeros((46, 45, 1))
                    k += 1
                    batch_index += 1

            if (batch_index % batch_size ) == 0:
                x_train = np.array(source_one_images)
                print(x_train.shape)
                # x_test = source_two_images
                x_train_source = np.array(source_pos_one)
                print(x_train_source.shape)
                # x_test_source = source_pos_two
                yield x_train, x_train_source
    return NotImplementedError


def CameraGenerator(source_file, num_images_per_output, batch_size, dim, n_channels, truths, shuffle, seed=0, trigger=4):
    '''
    This is for creating the classic 46x45 sized images of the detector and putting the source in the camera frame of reference

    :param source_file:
    :param num_images_per_output:
    :param batch_size:
    :param dim: tuple, dimension shape for the images
    :param n_channels:
    :param truths: truth sources, if not contained in 'Source_Position'
    :param shuffle:
    :param seed:
    :param trigger: Either 4 or 1024, for background or events, FACT trigger type
    :return:
    '''
    with h5py.File(source_file) as f:
        items = list(f.items())[0][1].shape[0]
        source_one_images = []
        source_pos_one = []
        tmp_arr = np.zeros((46, 45, 1))
        k = 1
        batch_index = 0
        source_truth = f['Source_Position'][0]
        for i in range(0, items):
            if not np.array_equal(f['Source_Position'][i], source_truth) and f['Trigger'][i] == trigger:
                source_truth_two = f['Source_Position'][i]
        for i in range(0, items):
            if np.array_equal(f['Source_Position'][i], source_truth) and f['Trigger'][i] == trigger:
                # arrays are the same, add to source images and ones
                # Randomly flip the array twice to augment training

                if (k % num_images_per_output) == 0:
                    # Add to temp image
                    tmp_arr += f['Image'][i]
                    k += 1
                else:
                    # Hit the 5000 cap I need
                    # print("5000 Hit")
                    tmp_arr = f['Image'][i]
                    # REsize correctly
                    if tmp_arr.shape != dim:
                        # Because it won't ever be smaller, only have to chech which dims are smaller than required
                        if tmp_arr.shape[0] < dim[0]:
                            tmp_arr = np.c_[tmp_arr.reshape((46, 45)), np.zeros((46, (dim[0] - tmp_arr.shape[0])))]
                        if tmp_arr.shape[1] < dim[1]:
                            tmp_arr = np.r_[tmp_arr, np.zeros(((dim[1] - tmp_arr.shape[1]), dim[0]))]
                    tmp_arr = tmp_arr.reshape((dim[1], dim[0], 1))
                    # tmp_arr.resize((48,48,1))
                    source_one_images.append(tmp_arr)
                    source_arr = f['Source_Position'][i]
                    if source_arr.shape[0] < dim[0]:
                        source_arr = np.c_[source_arr.reshape((46, 45)), np.zeros((46, (dim[0] - source_arr.shape[0])))]
                    if source_arr.shape[1] < dim[1]:
                        source_arr = np.r_[source_arr, np.zeros(((dim[1] - source_arr.shape[1]), dim[0]))]
                    source_arr = source_arr.reshape((dim[1], dim[0], 1))
                    # source_arr.resize((48,48,1), refcheck=False)
                    source_pos_one.append(source_arr)
                    tmp_arr = np.zeros((46, 45, 1))
                    k += 1
                    batch_index += 1

            if (batch_index % batch_size ) == 0:
                x_train = np.array(source_one_images)
                print(x_train.shape)
                # x_test = source_two_images
                x_train_source = np.array(source_pos_one)
                print(x_train_source.shape)
                # x_test_source = source_pos_two
                yield x_train, x_train_source


def AuxGenerator(source_file, aux_file, num_images_per_output, batch_size, dim, n_channels, truths, shuffle, seed=0):
    '''
    This is for taking the std_analysis files and using their data to augment that of the normal images

    :param source_file:
    :param aux_file:
    :param num_images_per_output:
    :param batch_size:
    :param dim:
    :param n_channels:
    :param truths:
    :param shuffle:
    :param seed:
    :return:
    '''
    return NotImplementedError


# Make a batch generator to generate the combined image batches to feed into the Keras model

class SimDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, length_dataset, source_file, image_size=5000, seed=0, batch_size=32, dim=(46, 45), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.length_dataset = length_dataset
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.source_file = source_file
        self.seed = seed
        self.image_size = image_size
        self.batch_index = 1  # Start
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.length_dataset / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        with h5py.File(self.source_file, 'r') as hdf:
            items = list(hdf['Night'])
            self.indexes = np.arange(len(items))
            np.random.seed(self.seed)
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def batchYielder(self, list_indexes):
        # Generate the batches
        gamma_batch_images = []
        gamma_batch_triggers = []
        hadron_batch_images = []
        hadron_batch_triggers = []
        with h5py.File(self.source_file, 'r') as hdf:
            i = 0
            j = 0  # The index of the image size
            k = 0  # Batch size
            while k < self.batch_size:
                gamma_images = []
                hadron_images = []
                for current_index in list_indexes[j:j + self.image_size]:
                    gamma_image = hdf['Gamma'][current_index]
                    hadron_image = hdf['Hadron'][current_index]
                    gamma_images.append(gamma_image)
                    hadron_images.append(hadron_image)
                    i += 1
                    print(i)
                # nights = np.asarray(nights)
                # runs = np.asarray(runs)
                # events = np.asarray(events)
                images = np.asarray(gamma_images)
                images = np.sum(images, axis=0)
                hadron_images = np.asarray(hadron_images)
                hadron_images = np.sum(hadron_images, axis=0)
                gamma_batch_images.append(images)
                gamma_batch_triggers.append(
                    1)  # 1 For Gamma, since it is a source, 0 for Hadron, since it is just background
                hadron_batch_images.append(hadron_images)
                hadron_batch_triggers.append(0)
                i = 0  # Reset to 0 after getting to image size, like 5000
                j += self.image_size  # Increase so chooses next set of it
                self.batch_index += self.batch_size  # Add batch size to keep track of where I am in the self.indexes
                k += 1
            batch_images = np.asarray(gamma_batch_images + hadron_batch_images)
            batch_triggers = np.asarray(gamma_batch_triggers + hadron_batch_triggers)
            yield (batch_images, batch_triggers)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i in range(0, self.batch_size):
            # Store sample
            image_gen = self.batchYielder(list_IDs_temp)
            next_image = next(image_gen)
            X[i,] = next_image[0]
            # Store class
            y[i] = next_image[1]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# Make a batch generator to generate the combined image batches to feed into the Keras model

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, length_dataset, source_file, image_size=5000, seed=0, batch_size=32, dim=(46, 45), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.length_dataset = length_dataset
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.source_file = source_file
        self.seed = seed
        self.image_size = image_size
        self.batch_index = 1  # Start
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.length_dataset / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        with h5py.File(self.source_file, 'r') as hdf:
            items = list(hdf['Night'])
            self.indexes = np.arange(len(items))
            np.random.seed(self.seed)
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def batchYielder(self, list_indexes):
        # Generate the batches
        batch_images = []
        batch_triggers = []
        with h5py.File(self.source_file, 'r') as hdf:
            i = 0
            j = 0  # The index of the image size
            k = 0  # Batch size
            while k < self.batch_size:
                nights = []
                runs = []
                events = []
                images = []
                triggers = []
                for current_index in list_indexes[j:j + self.image_size]:
                    night = hdf['Night'][current_index]
                    run = hdf['Run'][current_index]
                    event = hdf['Event'][current_index]
                    image = hdf['Image'][current_index]
                    trigger = hdf['Trigger'][current_index]
                    nights.append(night)
                    runs.append(run)
                    events.append(event)
                    images.append(image)
                    triggers.append(trigger)
                    i += 1
                    print(i)
                # nights = np.asarray(nights)
                # runs = np.asarray(runs)
                # events = np.asarray(events)
                images = np.asarray(images)
                triggers = np.asarray(triggers)
                images = np.sum(images, axis=0)
                batch_images.append(images)
                trigger_class = 0
                trigger_class = 1  # Assume all sources on actual data for now

                batch_triggers.append(trigger_class)
                i = 0  # Reset to 0 after getting to image size, like 5000
                j += self.image_size  # Increase so chooses next set of it
                self.batch_index += self.batch_size  # Add batch size to keep track of where I am in the self.indexes
                k += 1
            batch_images = np.asarray(batch_images)
            batch_triggers = np.asarray(batch_triggers)
            yield (batch_images, batch_triggers)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i in range(0, self.batch_size):
            # Store sample
            image_gen = self.batchYielder(list_IDs_temp)
            next_image = next(image_gen)
            X[i,] = next_image[0]
            # Store class
            y[i] = next_image[1]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class PrebatchDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, length_dataset, source_file, image_size=5000, seed=0, batch_size=32, dim=(46, 45), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.length_dataset = length_dataset
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.source_file = source_file
        self.seed = seed
        self.image_size = image_size
        self.batch_index = 1  # Start
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.length_dataset / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        with h5py.File(self.source_file, 'r') as hdf:
            items = list(hdf['Night'])
            self.indexes = np.arange(len(items))
            np.random.seed(self.seed)
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def batchYielder(self, list_indexes):
        # Generate the batches
        batch_images = []
        batch_triggers = []
        with h5py.File(self.source_file, 'r') as hdf:
            i = 0
            items = list(hdf.items())[0][1].shape[0]
            while (i + 1) * self.batch_size < items / 1:  # 160 factor to not process everything
                night = np.array(hdf['Night'][i * self.batch_size:(i + 1) * self.batch_size])
                run = np.array(hdf['Run'][i * self.batch_size:(i + 1) * self.batch_size])
                event = np.array(hdf['Event'][i * self.batch_size:(i + 1) * self.batch_size])
                images = np.array(hdf['Image'][i * self.batch_size:(i + 1) * self.batch_size])
                i += 1

                yield (night, run, event, images)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i in range(0, self.batch_size):
            # Store sample
            image_gen = self.batchYielder(list_IDs_temp)
            next_image = next(image_gen)
            X[i,] = next_image[0]
            # Store class
            y[i] = next_image[1]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


if __name__ == '__main__':
    print("Imported")
