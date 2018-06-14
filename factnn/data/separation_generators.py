from factnn.data.base_generator import BaseGenerator
import h5py

class SeparationGenerator(BaseGenerator):

    def init(self):
        self.type_gen = "Separation"
        if not self.from_directory:
            # Not flowing from photonstream files
            with h5py.File(self.input, 'r') as input_one:
                with h5py.File(self.second_input, 'r') as input_two:
                    self.input_data = input_one['Image']
                    self.second_input_data = input_two['Image']
