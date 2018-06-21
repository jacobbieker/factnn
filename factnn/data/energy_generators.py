from factnn.data.base_generator import BaseGenerator
import h5py

class EnergyGenerator(BaseGenerator):

    def init(self):
        self.type_gen = "Energy"
        if not self.from_directory:
            # Not flowing from photonstream files
            with h5py.File(self.input, 'r') as input_one:
                self.input_data = input_one['Image']
                self.labels = input_one['Energy']