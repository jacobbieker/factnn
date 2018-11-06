from factnn.generator.generator.base_generator import BaseGenerator
import h5py

class EnergyGenerator(BaseGenerator):

    def init(self):
        self.type_gen = "Energy"
        if not self.from_directory:
            # Not flowing from photonstream files
            with h5py.File(self.input, 'r') as input_one:
                self.input_shape = input_one['Image'].shape
                self.labels = input_one['Energy'][:]
            self.input_shape = (-1, self.input_shape[1], self.input_shape[2], self.input_shape[3], 1)

