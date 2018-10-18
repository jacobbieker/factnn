from factnn.data.augment import image_augmenter, get_completely_random_hdf5, get_random_hdf5_chunk, get_random_from_list


class BaseGenerator(object):
    def __init__(self, config):
        '''
        Base model for other models to use
        :param config: Dictionary of values for the given model
        '''

        if config['seed']:
            self.seed = config['seed']

        self.batch_size = config['batch_size']
        self.input = config['input']
        self.second_input = config['second_input']
        self.start_slice = config['start_slice']
        self.number_slices = config['number_slices']
        self.train_fraction = config['train_fraction']
        self.validate_fraction = config['validate_fraction']
        self.input_data = None
        self.second_input_data = None
        self.labels = None
        self.type_gen = None
        # Items is either an int, the number of samples to use, or an array of indicies for the generator
        # If items is an array, then chunked must be False, and cannot be from_directory
        self.items = config['samples']
        self.mode = config['mode']

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
            if self.chunked:
                if self.mode == "train":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.start_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=self.labels,
                                                                                proton_data=self.second_input_data,
                                                                                type_training=self.type_gen,
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)
                elif self.mode == "validate":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.start_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=self.labels,
                                                                                proton_data=self.second_input_data,
                                                                                type_training=self.type_gen,
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)

                elif self.mode == "test":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.start_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=self.labels,
                                                                                proton_data=self.second_input_data,
                                                                                type_training=self.type_gen,
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)
            else:
                # not chunked
                if self.mode == "train":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.start_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=self.labels,
                                                                               proton_data=self.second_input_data,
                                                                               type_training=self.type_gen,
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)
                elif self.mode == "validate":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.start_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=self.labels,
                                                                               proton_data=self.second_input_data,
                                                                               type_training=self.type_gen,
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)

                elif self.mode == "test":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.start_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=self.labels,
                                                                               proton_data=self.second_input_data,
                                                                               type_training=self.type_gen,
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)

    def __str__(self):
        return NotImplemented

    def __repr__(self):
        return NotImplemented
