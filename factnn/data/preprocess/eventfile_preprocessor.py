import numpy as np
from factnn.data.preprocess.base_preprocessor import BasePreprocessor
import pickle
import os


class EventFilePreprocessor(BasePreprocessor):
    def init(self):
        pass

    def check_files(self, paths, title):
        """
        Check various things on the given paths, like non-null photons, and overall distribution of starting and end points
        :param paths:
        :return:
        """
        starts = []
        ends = []
        means = []
        stds = []
        failed_paths = []
        import matplotlib.pyplot as plt
        for index, file in enumerate(paths):
            try:
                with open(file, "rb") as pickled_event:
                    data, data_format = pickle.load(pickled_event)
                    self.start, self.end, mean, std = self.dynamic_size(data[data_format['Image']])
                    if self.start < 0:
                        failed_paths.append(file)
                    else:
                        # Real paths
                        starts.append(self.start)
                        ends.append(self.end)
                        means.append(mean)
                        stds.append(std)
            except Exception as e:
                print(e)
                print(file)
                print(paths)
                failed_paths.append(file)
        for path in failed_paths:
            os.remove(path)
        #    print("Removed: ", path)
        #print("Number of failed paths: ", len(failed_paths))
        #plt.hist(starts)
        #plt.title("Start Spots: " + str(title) + " Mean/STD: " + str(np.round(np.mean(starts), 2)) + "+-" + str(np.round(np.std(starts), 2)))
        #plt.show()
        #plt.hist(ends)
        #plt.title("End Spots: " + str(title) + " Mean/STD: " + str(np.round(np.mean(ends), 2)) + "+-" + str(np.round(np.std(ends), 2)))
        #plt.show()
        #plt.hist(means)
        #plt.title("Mean: " + str(title) + " Mean/STD: " + str(np.round(np.mean(means), 2)) + "+-" + str(np.round(np.std(means), 2)))
        #plt.show()
        #plt.hist(std)
        #plt.title("Std: " + str(title) + " Mean/STD: " + str(np.round(np.mean(stds), 2)) + "+-" + str(np.round(np.std(stds), 2)))
        #plt.show()

        return failed_paths


    def on_files_processor(self, paths, collapse_time=True, final_slices=5, normalize=False, dynamic_resize=False,
                           truncate=False, equal_slices=False, return_features=False, return_collapsed=False):
        all_data = []
        for index, file in enumerate(paths):
            # load the pickled file from the disk
            with open(file, "rb") as pickled_event:
                data, data_format, features, feature_cluster = pickle.load(pickled_event)
                if return_features:
                    if features['extraction'] == 1:
                        # Failed feature extraction, so ignore event
                        continue
                    else:
                        # Based off a subset the Open Crab Sample Analysis
                        feature_list = []
                        feature_list.append(features['head_tail_ratio'])
                        feature_list.append(features['length'])
                        feature_list.append(features['width'])
                        feature_list.append(features['time_gradient'])
                        feature_list.append(features['number_photons'])
                        feature_list.append(features['length']*features['width']*np.pi)
                        feature_list.append(((features['length']*features['width']*np.pi)/np.log(features['number_photons'])**2))
                        feature_list.append((features['number_photons'] / (features['length']*features['width']*np.pi)))
                input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
                chid_to_pixel = self.rebinning[0]
                pixel_index_to_grid = self.rebinning[1]
                # Do dynamic resizing if wanted, so start and end are only within the bounds, potentially saving memory
                if dynamic_resize:
                    self.start, self.end, _, _ = self.dynamic_size(data[data_format['Image']])

                if truncate:
                    # Truncates the images at x timesteps in, each slice is one temporal slice
                    self.end = self.start + self.shape[3]

                if equal_slices:
                    # Creates a slice size that then assignes all values in those set of slices to a single slice to fit
                    # within the orignal constraints, each slice is an equal number of timeslices summed up
                    slice_size = int(np.ceil((self.end - self.start) / self.shape[3]))
                    slice_sizes = []
                    for i in range(self.shape[3]):
                        # Organized smallest to largest
                        slice_sizes.append(((i)*slice_size)+self.start)

                for index in range(1440):
                    for element in chid_to_pixel[index]:
                        coords = pixel_index_to_grid[element[0]]
                        for value in data[data_format['Image']][index]:
                            if self.end > value >= self.start:
                                # Now add more logic for the other cases
                                # Nothing for truncate, end is already specified
                                # Equal slices is only real one
                                if equal_slices:
                                    for idx, number in enumerate(slice_sizes):
                                        if (idx*slice_size) < value <= number:
                                            # In the range of the slice, add to it
                                            input_matrix[coords[0]][coords[1]][idx] += element[1] * 1
                                            break
                                elif not truncate:
                                    # Now sum up last one, as equal already used, so last frame has all the rest of the frames
                                    if value - self.start >= self.shape[3]:
                                        input_matrix[coords[0]][coords[1]][self.shape[3]-1] += element[1] * 1
                                    else:
                                        input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1
                                else:
                                    input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1

                # Now have image in resized format, all other data is set
                data[data_format["Image"]] = np.fliplr(np.rot90(input_matrix, 3))
                # need to do the format thing here, and add auxiliary structure
                data = self.format([data, data_format])
                if return_collapsed:
                    collapsed_data = self.collapse_image_time(data[0], 1, self.as_channels)
                if normalize:
                    data = list(data)
                    data[0] = self.normalize_image(data[0], per_slice=False)
                    data = tuple(data)
                    if return_collapsed:
                        collapsed_data = self.normalize_image(collapsed_data, per_slice=False)
                if collapse_time:
                    data = list(data)
                    data[0] = self.collapse_image_time(data[0], final_slices, self.as_channels)
                    data = tuple(data)
            temp_data = [data, data_format]
            if return_features:
                temp_data.append(feature_list)
            if return_collapsed:
                temp_data.append(collapsed_data)
            all_data.append(temp_data)
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
                        if self.end > value >= self.start:
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

        # TODO: Look into more even distribution of information, like each slce having multiple timesteps vs the last one
        # having them all
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