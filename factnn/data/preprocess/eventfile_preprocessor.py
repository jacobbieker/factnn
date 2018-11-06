import numpy as np
from factnn.data.preprocess.base_preprocessor import BasePreprocessor
import pickle


class EventFilePreprocessor(BasePreprocessor):
    def init(self):
        pass

    def single_processor(self, normalize=False, collapse_time=False, final_slices=5):
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
        data[0] = self.reformat(np.array(data[data_format["Image"]]))
        for index in range(data_format["Image"], len(data)):
            data[index] = np.array(data[index])
        return data
