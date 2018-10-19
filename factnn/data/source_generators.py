from factnn.data.base_generator import BaseGenerator
import h5py
import numpy as np


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class DispGenerator(BaseGenerator):

    def init(self):
        self.type_gen = "Disp"
        if not self.from_directory:
            # Not flowing from photonstream files
            with h5py.File(self.input, 'r') as input_one:
                self.input_data = input_one['Image']
                source_y = input_one['Source_X'][:]
                source_x = input_one['Source_Y'][:]
                cog_x = input_one['COG_X'][:]
                cog_y = input_one['COG_Y'][:]
                self.labels = euclidean_distance(
                    source_x, source_y,
                    cog_x, cog_y
                )


class SignGenerator(BaseGenerator):

    def init(self):
        self.type_gen = "Sign"
        if not self.from_directory:
            # Not flowing from photonstream files
            with h5py.File(self.input, 'r') as input_one:
                self.input_data = input_one['Image']
                source_y = input_one['Source_X'][:]
                source_x = input_one['Source_Y'][:]
                cog_x = input_one['COG_X'][:]
                cog_y = input_one['COG_Y'][:]
                delta = input_one['Delta'][:]
                true_delta = np.arctan2(
                    cog_y - source_y,
                    cog_x - source_x
                )
                self.labels = np.sign(np.abs(delta - true_delta) - np.pi / 2)
