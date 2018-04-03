import keras
import numpy as np
import h5py


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
        self.batch_index = 1 # Start
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
            j = 0 # The index of the image size
            k = 0 # Batch size
            while k < self.batch_size:
                gamma_images = []
                hadron_images = []
                for current_index in list_indexes[j:j+self.image_size]:
                    gamma_image = hdf['Gamma'][current_index]
                    hadron_image = hdf['Hadron'][current_index]
                    gamma_images.append(gamma_image)
                    hadron_images.append(hadron_image)
                    i += 1
                    print(i)
                #nights = np.asarray(nights)
                #runs = np.asarray(runs)
                #events = np.asarray(events)
                images = np.asarray(gamma_images)
                images = np.sum(images, axis=0)
                hadron_images = np.asarray(hadron_images)
                hadron_images = np.sum(hadron_images, axis=0)
                gamma_batch_images.append(images)
                gamma_batch_triggers.append(1) # 1 For Gamma, since it is a source, 0 for Hadron, since it is just background
                hadron_batch_images.append(hadron_images)
                hadron_batch_triggers.append(0)
                i = 0 # Reset to 0 after getting to image size, like 5000
                j += self.image_size # Increase so chooses next set of it
                self.batch_index += self.batch_size # Add batch size to keep track of where I am in the self.indexes
                k += 1
            batch_images = np.asarray(gamma_batch_images+hadron_batch_images)
            batch_triggers = np.asarray(gamma_batch_triggers+hadron_batch_triggers)
            yield (batch_images, batch_triggers)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

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
        self.batch_index = 1 # Start
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
            j = 0 # The index of the image size
            k = 0 # Batch size
            while k < self.batch_size:
                nights = []
                runs = []
                events = []
                images = []
                triggers = []
                for current_index in list_indexes[j:j+self.image_size]:
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
                #nights = np.asarray(nights)
                #runs = np.asarray(runs)
                #events = np.asarray(events)
                images = np.asarray(images)
                triggers = np.asarray(triggers)
                images = np.sum(images, axis=0)
                batch_images.append(images)
                trigger_class = 0
                if np.bincount(triggers).argmax() == 4:
                    trigger_class = 1

                batch_triggers.append(trigger_class)
                i = 0 # Reset to 0 after getting to image size, like 5000
                j += self.image_size # Increase so chooses next set of it
                self.batch_index += self.batch_size # Add batch size to keep track of where I am in the self.indexes
                k += 1
            batch_images = np.asarray(batch_images)
            batch_triggers = np.asarray(batch_triggers)
            yield (batch_images, batch_triggers)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i in range(0, self.batch_size):
            # Store sample
            image_gen = self.batchYielder(list_IDs_temp)
            next_image = next(image_gen)
            X[i,] = next_image[0]
            # Store class
            y[i] = next_image[1]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
