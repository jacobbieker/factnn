import numpy as np
import h5py
import keras
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from factnn.plotting.plotting import plot_roc, plot_disp_confusion, plot_energy_confusion
import matplotlib.pyplot as plt

from factnn.data.augment import image_augmenter, get_completely_random_hdf5, get_random_hdf5_chunk


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
        self.end_slice = config['end_slice']
        self.number_slices = config['number_slices']
        self.train_fraction = config['train_fraction']
        self.validate_fraction = config['validate_fraction']
        self.test_fraction = config['test_fraction']
        self.input_data = None
        self.second_input_data = None
        self.labels = None
        self.items = config['number_samples']
        self.mode = config['mode']

        if config['verbose']:
            self.verbose = config['verbose']
        else:
            self.verbose = False

        if config['augment']:
            self.augment = config['augment']

        if config['from_directory']:
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
        return NotImplemented

    def __str__(self):
        return NotImplemented

    def __repr__(self):
        return NotImplemented