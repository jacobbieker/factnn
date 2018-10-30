from factnn.data.augment import get_random_from_list, \
    get_chunk_from_list, get_random_from_paths
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
        if 'input' in config:
            self.input = config['input']
        else:
            self.input = None
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
        if 'input_shape' in config:
            self.input_shape = config['input_shape']
        else:
            self.input_shape = None
        self.mode = config['mode']
        self.train_data = None
        self.validate_data = None
        self.test_data = None

        self.validate_steps = None
        self.validate_current_step = 0

        self.test_steps = None
        self.test_current_step = 0

        # Now the preprocessor stuff, only used for streaming from files
        self.train_preprocessor = None
        self.validate_preprocessor = None
        self.test_preprocessor = None

        # Also need proton counterparts, has to be an easier way....
        self.proton_train_preprocessor = None
        self.proton_validate_preprocessor = None
        self.proton_test_preprocessor = None

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
            while True:
                if self.mode == "train":
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
                    self.validate_steps = int(np.floor(len(self.validate_data) / self.batch_size))
                    # Shouldn't be doing random for this one, should be same everytime
                    batch_images, batch_image_label = get_chunk_from_list(self.validate_data, size=self.batch_size,
                                                                          time_slice=self.start_slice,
                                                                          total_slices=self.number_slices,
                                                                          labels=self.labels,
                                                                          augment=False,
                                                                          gamma=self.input,
                                                                          proton_input=self.second_input,
                                                                          shape=self.input_shape,
                                                                          swap=False,
                                                                          current_step=self.validate_current_step)
                    self.validate_current_step += 1
                    self.validate_current_step %= self.validate_steps
                    return batch_images, batch_image_label

                elif self.mode == "test":
                    self.test_steps = int(np.floor(len(self.test_data) / self.batch_size))
                    # Shouldn't be random or augmenting this one, should be same everytime
                    batch_images, batch_image_label = get_chunk_from_list(self.test_data, size=self.batch_size,
                                                                          time_slice=self.start_slice,
                                                                          total_slices=self.number_slices,
                                                                          labels=self.labels,
                                                                          augment=False,
                                                                          gamma=self.input,
                                                                          proton_input=self.second_input,
                                                                          shape=self.input_shape,
                                                                          swap=False,
                                                                          current_step=self.test_current_step)
                    self.test_current_step += 1
                    self.test_current_step %= self.test_steps
                    return batch_images, batch_image_label
        else:
            # Now streaming from files, training, test, and validation need to be preprocessors set up for it.
            if self.mode == "train":
                batch_images, batch_image_label = get_random_from_paths(preprocessor=self.train_preprocessor,
                                                                        size=self.batch_size,
                                                                        time_slice=self.start_slice,
                                                                        total_slices=self.number_slices,
                                                                        augment=self.augment,
                                                                        shape=self.input_shape,
                                                                        type_training=self.type_gen,
                                                                        proton_preprocessor=self.proton_train_preprocessor)
                return batch_images, batch_image_label

            elif self.mode == "validate":
                batch_images, batch_image_label = get_random_from_paths(preprocessor=self.validate_preprocessor,
                                                                        size=self.batch_size,
                                                                        time_slice=self.start_slice,
                                                                        total_slices=self.number_slices,
                                                                        augment=False,
                                                                        shape=self.input_shape,
                                                                        swap=False,
                                                                        type_training=self.type_gen,
                                                                        proton_preprocessor=self.proton_validate_preprocessor)
                return batch_images, batch_image_label
            elif self.mode == "test":
                batch_images, batch_image_label = get_random_from_paths(preprocessor=self.test_preprocessor,
                                                                        size=self.batch_size,
                                                                        time_slice=self.start_slice,
                                                                        total_slices=self.number_slices,
                                                                        augment=False,
                                                                        shape=self.input_shape,
                                                                        swap=False,
                                                                        type_training=self.type_gen,
                                                                        proton_preprocessor=self.proton_test_preprocessor)
                return batch_images, batch_image_label

    def __str__(self):
        return NotImplemented

    def __repr__(self):
        return NotImplemented
