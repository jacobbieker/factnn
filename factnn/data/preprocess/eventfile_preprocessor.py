import numpy as np
from factnn.data.preprocess.base_preprocessor import BasePreprocessor
import pickle


class EventFilePreprocessor(BasePreprocessor):
    def init(self):
        pass

    def on_files_processor(self, paths, collapse_time=True, final_slices=5, normalize=False):
        all_data = []
        for index, file in enumerate(paths):
            # load the pickled file from the disk
            with open(file, "rb") as pickled_event:
                data, data_format = pickle.load(pickled_event)
                input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
                chid_to_pixel = self.rebinning[0]
                pixel_index_to_grid = self.rebinning[1]
                for index in range(1440):
                    for element in chid_to_pixel[index]:
                        coords = pixel_index_to_grid[element[0]]
                        for value in data[data_format['Image']][index]:
                            if self.end > value > self.start:
                                input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1

                # Now have image in resized format, all other data is set
                data[data_format["Image"]] = np.fliplr(np.rot90(input_matrix, 3))
                # need to do the format thing here, and add auxiliary structure
                data = self.format([data, data_format])
                if normalize:
                    data = list(data)
                    data[0] = self.normalize_image(data[0], per_slice=False)
                    data = tuple(data)
                if collapse_time:
                    data = list(data)
                    data[0] = self.collapse_image_time(data[0], final_slices, self.as_channels)
                    data = tuple(data)
            all_data.append([data, data_format])
        # Now have all the data transformed as necessary, return as list of list of images, data_formats
        return all_data


    def single_processor(self, normalize=False, collapse_time=False, final_slices=5, clean_images=False):
        pass

    def event_file_processor(self, filepath, normalize=False, collapse_time=False, final_slices=5):
        """
        Obtain one event, preprocess the image and send it on its way. The pickled images need to be with a dictionary

        :param filepath:
        :param normalize:
        :param collapse_time:
        :param final_slices:
        :return:
        """
        with open(filepath, "rb") as data_file:
            data, data_format = pickle.load(data_file)
            input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
            chid_to_pixel = self.rebinning[0]
            pixel_index_to_grid = self.rebinning[1]
            for index in range(1440):
                for element in chid_to_pixel[index]:
                    coords = pixel_index_to_grid[element[0]]
                    for value in data[data_format['Image']][index]:
                        if self.end > value > self.start:
                            input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1

            # Now have image in resized format, all other data is set
            data[data_format["Image"]] = np.fliplr(np.rot90(input_matrix, 3))
            # need to do the format thing here, and add auxiliary structure
            data = self.format([data, data_format])
            if normalize:
                data = list(data)
                data[0] = self.normalize_image(data[0])
                data = tuple(data)
            if collapse_time:
                data = list(data)
                data[0] = self.collapse_image_time(data[0], final_slices, self.as_channels)
                data = tuple(data)
            yield data, data_format

    def format(self, batch):
        data = batch[0]
        data_format = batch[1]
        data[0] = np.array(data[data_format["Image"]])
        for index in range(data_format["Image"], len(data)):
            data[index] = np.array(data[index])
        return data

    def collapse_image_time(self, image, final_slices, as_channels=False):
        """
        Partially flattens an image cube to a smaller set, e.g. (1,40,75,75) with final_slices=3 becomes
        (1,3,75,75) with each new slice being a sum of the fraction of slices of the whole

        If as_channels is True, then the time_slices are moved to the channels, so the previous example
        would end up with the final shape (1,75,75,3)

        :param image: The image in (1, time_slices, width, height, channel) order
        :param final_slices: Number of slices to use
        :param as_channels: Boolean, if the time dimension should be moved to the channels
        :return: Converted image cube with the proper dimensions
        """
        temp_matrix = []
        num_slices_per_final_slice = int(np.floor(image.shape[2] / final_slices))
        # Need to now sum up along each smaller section
        for time_slice in range(final_slices):
            if time_slice < (final_slices - 1):
                image_slice = image[::, ::, time_slice*num_slices_per_final_slice:((time_slice+1)*num_slices_per_final_slice)]
            else:
                # To use all the available slices
                image_slice = image[::, ::, time_slice*num_slices_per_final_slice:]
            image_slice = np.sum(image_slice, axis=2)
            temp_matrix.append(image_slice)
        # Should be normalized now
        temp_matrix = np.array(temp_matrix)
        # Now to convert to chennel format if needed
        if as_channels:
            temp_matrix = np.swapaxes(temp_matrix, 0, 2)
            # Second one is to keep the order of the width/height
            temp_matrix = np.swapaxes(temp_matrix, 0, 1)
            temp_matrix = temp_matrix.reshape(1, temp_matrix.shape[0], temp_matrix.shape[1], temp_matrix.shape[2])
        else:
            # Else keep same format as before
            temp_matrix = temp_matrix.reshape(1, temp_matrix.shape[0], temp_matrix.shape[1], temp_matrix.shape[2])
        return temp_matrix