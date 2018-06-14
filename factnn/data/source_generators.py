from factnn.data.base_generator import BaseGenerator
import h5py
import numpy as np
from factnn.data.augment import get_random_hdf5_chunk, get_random_from_list


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class DispGenerator(BaseGenerator):

    def init(self):
        if not self.from_directory:
            # Not flowing from photonstream files
            with h5py.File(self.input, 'r') as input_one:
                self.input_data = input_one['Image']
                source_y = input_one['Source_X'].values
                source_x = input_one['Source_Y'].values
                cog_x = input_one['COG_X'].values
                cog_y = input_one['COG_Y'].values
                self.labels = euclidean_distance(
                    source_x, source_y,
                    cog_x, cog_y
                )

    def __next__(self):
        if not self.from_directory:
            if self.chunked:
                if self.mode == "train":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.end_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=self.labels,
                                                                                proton_data=None,
                                                                                type_training='Disp',
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)
                elif self.mode == "validate":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.end_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=self.labels,
                                                                                proton_data=None,
                                                                                type_training='Disp',
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)

                elif self.mode == "test":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.end_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=self.labels,
                                                                                proton_data=None,
                                                                                type_training='Disp',
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)
            else:
                # not chunked
                if self.mode == "train":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.end_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=self.labels,
                                                                               proton_data=None,
                                                                               type_training='Disp',
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)
                elif self.mode == "validate":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.end_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=self.labels,
                                                                               proton_data=None,
                                                                               type_training='Disp',
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)

                elif self.mode == "test":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.end_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=self.labels,
                                                                               proton_data=None,
                                                                               type_training='Disp',
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)


class SignGenerator(BaseGenerator):

    def init(self):
        if not self.from_directory:
            # Not flowing from photonstream files
            with h5py.File(self.input, 'r') as input_one:
                self.input_data = input_one['Image']
                source_y = input_one['Source_X'].values
                source_x = input_one['Source_Y'].values
                cog_x = input_one['COG_X'].values
                cog_y = input_one['COG_Y'].values
                delta = input_one['Delta'].values

                true_delta = np.arctan2(
                    cog_y - source_y,
                    cog_x - source_x
                )
                self.labels = np.sign(np.abs(delta - true_delta) - np.pi / 2)

    def __next__(self):
        if not self.from_directory:
            if self.chunked:
                if self.mode == "train":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.end_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=self.labels,
                                                                                proton_data=None,
                                                                                type_training='Disp',
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)
                elif self.mode == "validate":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.end_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=self.labels,
                                                                                proton_data=None,
                                                                                type_training='Disp',
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)

                elif self.mode == "test":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.end_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=self.labels,
                                                                                proton_data=None,
                                                                                type_training='Disp',
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)
            else:
                # not chunked
                if self.mode == "train":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.end_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=self.labels,
                                                                               proton_data=None,
                                                                               type_training='Disp',
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)
                elif self.mode == "validate":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.end_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=self.labels,
                                                                               proton_data=None,
                                                                               type_training='Disp',
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)

                elif self.mode == "test":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.end_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=self.labels,
                                                                               proton_data=None,
                                                                               type_training='Disp',
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)
