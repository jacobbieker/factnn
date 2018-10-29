from factnn.data.augment import get_completely_random_hdf5, get_random_hdf5_chunk, get_random_from_list
from sklearn.model_selection import train_test_split
import numpy as np

# TODO Add k-fold cross-validation generation

class BaseGenerator(object):
    def __init__(self, config):
        '''
        Base model for other models to use
        :param config: Dictionary of values for the given model
        '''

        if 'seed' in config:
            self.seed = config['seed']

        self.batch_size = config['batch_size']
        self.input = config['input']
        if 'second_input' in config:
            self.second_input = config['second_input']
        else:
            self.second_input = None
        self.start_slice = config['start_slice']
        self.number_slices = config['number_slices']
        self.input_data = None
        self.second_input_data = None
        self.labels = None
        self.type_gen = None
        self.input_shape = None
        # Items is either an int, the number of samples to use, or an array of indicies for the generator
        # If items is an array, then chunked must be False, and cannot be from_directory
        self.mode = config['mode']
        self.train_data = None
        self.validate_data = None
        self.test_data = None

        if 'train_data' in config:
            self.train_data = config['train_data']

        if 'validate_data' in config:
            self.validate_data = config['validate_data']

        if 'test_data' in config:
            self.test_data = config['test_data']

        # Converts into train, test, and validate datasets

        if 'chunked' in config:
            self.chunked = config['chunked']
        else:
            self.chunked = False

        if 'verbose' in config:
            self.verbose = config['verbose']
        else:
            self.verbose = False

        if 'augment' in config:
            self.augment = config['augment']

        if 'from_directory' in config:
            self.from_directory = config['from_directory']
        else:
            self.from_directory = False

        self.init()

        # Now self.input_shape will be defined, so set to the correct value of 1 at the end

    def init(self):
        '''
        Model specific inits are here, such as calculating Disp labels
        :return:
        '''
        return NotImplemented

    def __iter__(self):
        return self

    def __next__(self):
        '''
        Get the next batch of values here, should loop forever
        :return:
        '''
        if not self.from_directory:
            if self.mode == "train":
                while True:
                    batch_images, batch_image_label = get_random_from_list(self.train_data, size=self.batch_size,
                                                                           time_slice=self.start_slice,
                                                                           total_slices=self.number_slices,
                                                                           labels=self.labels,
                                                                           augment=self.augment,
                                                                           gamma=self.input,
                                                                           proton_input=self.second_input,
                                                                           shape=self.input_shape)
                    return batch_images, batch_image_label
            elif self.mode == "validate":
                while True:
                    batch_images, batch_image_label = get_random_from_list(self.validate_data, size=self.batch_size,
                                                                           time_slice=self.start_slice,
                                                                           total_slices=self.number_slices,
                                                                           labels=self.labels,
                                                                           augment=self.augment,
                                                                           gamma=self.input,
                                                                           proton_input=self.second_input,
                                                                           shape=self.input_shape)
                    return batch_images, batch_image_label

            elif self.mode == "test":
                while True:
                    batch_images, batch_image_label = get_random_from_list(self.test_data, size=self.batch_size,
                                                                           time_slice=self.start_slice,
                                                                           total_slices=self.number_slices,
                                                                           labels=self.labels,
                                                                           augment=self.augment,
                                                                           gamma=self.input,
                                                                           proton_input=self.second_input,
                                                                           shape=self.input_shape)
                    return batch_images, batch_image_label

    def __str__(self):
        return NotImplemented

    def __repr__(self):
        return NotImplemented
