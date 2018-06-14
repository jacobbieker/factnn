from factnn.data.base_generator import BaseGenerator
import numpy as np
import h5py

from factnn.data.augment import get_random_hdf5_chunk, get_random_from_list


class SeparationGenerator(BaseGenerator):

    def init(self):
        self.type_gen = "Separation"
        if not self.from_directory:
            # Not flowing from photonstream files
            with h5py.File(self.input, 'r') as input_one:
                with h5py.File(self.second_input, 'r') as input_two:
                    self.input_data = input_one['Image']
                    self.second_input_data = input_two['Image']
