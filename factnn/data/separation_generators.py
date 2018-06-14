from factnn.data.base_generator import BaseGenerator
import numpy as np
import h5py

from factnn.data.augment import get_random_hdf5_chunk, get_random_from_list


class SeparationGenerator(BaseGenerator):

    def init(self):
        if not self.from_directory:
            # Not flowing from photonstream files
            with h5py.File(self.input, 'r') as input_one:
                with h5py.File(self.second_input, 'r') as input_two:
                    self.input_data = input_one['Image']
                    self.second_input_data = input_two['Image']

    def __next__(self):
        if not self.from_directory:
            if self.chunked:
                if self.mode == "train":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.end_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=None,
                                                                                proton_data=self.second_input_data,
                                                                                type_training='Separation',
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)
                elif self.mode == "validate":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.end_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=None,
                                                                                proton_data=self.second_input_data,
                                                                                type_training='Separation',
                                                                                augment=self.augment)
                        yield (batch_images, batch_image_label)

                elif self.mode == "test":
                    while True:
                        batch_images, batch_image_label = get_random_hdf5_chunk(0, self.items, size=self.batch_size,
                                                                                time_slice=self.end_slice,
                                                                                total_slices=self.number_slices,
                                                                                training_data=self.input_data,
                                                                                labels=None,
                                                                                proton_data=self.second_input_data,
                                                                                type_training='Separation',
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
                                                                               labels=None,
                                                                               proton_data=self.second_input_data,
                                                                               type_training='Separation',
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)
                elif self.mode == "validate":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.end_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=None,
                                                                               proton_data=self.second_input_data,
                                                                               type_training='Separation',
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)

                elif self.mode == "test":
                    while True:
                        batch_images, batch_image_label = get_random_from_list(self.items, size=self.batch_size,
                                                                               time_slice=self.end_slice,
                                                                               total_slices=self.number_slices,
                                                                               training_data=self.input_data,
                                                                               labels=None,
                                                                               proton_data=self.second_input_data,
                                                                               type_training='Separation',
                                                                               augment=self.augment)
                        yield (batch_images, batch_image_label)
