from factnn.generator.generator.base_generator import BaseGenerator
import h5py


class SeparationGenerator(BaseGenerator):
    def init(self):
        self.type_gen = "Separation"
        if not self.from_directory:
            # Not flowing from photonstream files
            with h5py.File(self.input, "r") as input_one:
                with h5py.File(self.second_input, "r") as input_two:
                    self.input_shape = input_one["Image"].shape
            self.input_shape = (
                -1,
                self.input_shape[1],
                self.input_shape[2],
                self.input_shape[3],
                1,
            )
