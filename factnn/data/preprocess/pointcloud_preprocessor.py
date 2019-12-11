from factnn.data.preprocess.eventfile_preprocessor import EventFilePreprocessor
import pickle
import os
import numpy as np
from photon_stream.representations import list_of_lists_to_raw_phs, raw_phs_to_point_cloud
from photon_stream.geometry import GEOMETRY
from photon_stream.io.magic_constants import TIME_SLICE_DURATION_S


class PointCloudPreprocessor(EventFilePreprocessor):
    """
    This class is designed for preprocessing the eventfile format into point clouds, optionally with cleaning
    """

    def on_files_processor(self, paths, final_points=1024, replacement=False, truncate=False,
                           return_features=False, **kwargs):
        """
        Takes Eventfile format file paths, and returns point clouds with optional preprocessing


        :param paths: Paths to the Eventfiles
        :param final_points: Number of final points to return, if this is larger than the total number of photons,
        replacement will be used
        :param replacement: Whether to randomly choose points with replacement (True), or not, if there are more photons than points
        :param normalize:
        :param truncate: Whether to truncate the photons before converting to a point cloud or not
        :param return_features: Whether to return the Hillas features extracted beforehand
        :param kwargs:
        :return:
        """
        all_data = []
        for index, file in enumerate(paths):
            # load the pickled file from the disk
            if os.path.getsize(file) > 0:
                # Checks that file is not 0
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
                            feature_list.append(features['length'] * features['width'] * np.pi)
                            feature_list.append(((features['length'] * features['width'] * np.pi) / np.log(
                                features['number_photons']) ** 2))
                            feature_list.append(
                                (features['number_photons'] / (features['length'] * features['width'] * np.pi)))

                    # Convert from timeslice to time
                    self.end *= TIME_SLICE_DURATION_S
                    self.start *= TIME_SLICE_DURATION_S

                    if truncate:
                        self.end = self.start + (self.shape[3]*TIME_SLICE_DURATION_S)

                    # Convert List of List to Point Cloud, then truncation is simply cutting in the z direction
                    event_photons = data[data_format["Image"]]
                    event_photons = list_of_lists_to_raw_phs(event_photons)
                    point_cloud = np.asarray(raw_phs_to_point_cloud(event_photons,
                                                                    cx=GEOMETRY.x_angle,
                                                                    cy=GEOMETRY.y_angle))
                    start_one = min(point_cloud[:,2])
                    if start_one > self.start:
                        diff = start_one - self.start
                        self.start += diff
                        self.end += diff
                    #print("Min: {} Max: {}".format(min(point_cloud[:,2]), max(point_cloud[:,2])))
                    #print("Min: {} Max: {}".format(self.start, self.end))
                    # Now in point cloud format, truncation is just cutting off in z now
                    #print("Num Points Before Mask: {}".format(len(point_cloud)))
                    #mask = (point_cloud[:, 2] <= self.end) & (point_cloud[:, 2] >= self.start)
                    # TODO Move around if no pixels contained
                    # TODO mask seems to be off slightly
                    #point_cloud = point_cloud[mask]
                    #print("Num Points: {}".format(len(point_cloud)))

                    # Now have to subsample (or resample) points
                    # Replacement has to be used if there are less points than final_points photons available
                    if replacement or point_cloud.shape[0] < final_points:
                        point_indicies = np.random.choice(point_cloud.shape[0], final_points, replace=True)
                    else:
                        point_indicies = np.random.choice(point_cloud.shape[0], final_points, replace=False)

                    point_cloud = point_cloud[point_indicies]

                    data[data_format["Image"]] = point_cloud
                    data = self.format([data, data_format])

                temp_data = [data, data_format]
                if return_features:
                    temp_data.append(feature_list)
                all_data.append(temp_data)
        return all_data

    def event_file_processor(self, filepath, final_points=1024, replacement=False, truncate=False,
                             return_features=False, ):

        with open(filepath, "rb") as pickled_event:
            data, data_format, features, feature_cluster = pickle.load(pickled_event)
            if return_features:
                if features['extraction'] == 1:
                    # Failed feature extraction, so ignore event
                    pass
                else:
                    # Based off a subset the Open Crab Sample Analysis
                    feature_list = []
                    feature_list.append(features['head_tail_ratio'])
                    feature_list.append(features['length'])
                    feature_list.append(features['width'])
                    feature_list.append(features['time_gradient'])
                    feature_list.append(features['number_photons'])
                    feature_list.append(features['length'] * features['width'] * np.pi)
                    feature_list.append(
                        ((features['length'] * features['width'] * np.pi) / np.log(features['number_photons']) ** 2))
                    feature_list.append((features['number_photons'] / (features['length'] * features['width'] * np.pi)))

            # Convert from timeslice to time
            self.end *= TIME_SLICE_DURATION_S
            self.start *= TIME_SLICE_DURATION_S
            if truncate:
                self.end = self.start + (self.shape[3]*TIME_SLICE_DURATION_S)

            # Convert List of List to Point Cloud, then truncation is simply cutting in the z direction
            event_photons = data[data_format["Image"]]
            event_photons = list_of_lists_to_raw_phs(event_photons)
            point_cloud = np.asarray(raw_phs_to_point_cloud(event_photons,
                                                            cx=GEOMETRY.x_angle,
                                                            cy=GEOMETRY.y_angle))

            # Now in point cloud format, truncation is just cutting off in z now
            mask = (point_cloud[:, 2] <= self.end) & (point_cloud[:, 2] >= self.start)
            point_cloud = point_cloud[mask]

            # Now have to subsample (or resample) points
            # Replacement has to be used if there are less points than final_points
            if replacement or point_cloud.shape[0] < final_points:
                point_indicies = np.random.choice(point_cloud.shape[0], final_points, replace=True)
            else:
                point_indicies = np.random.choice(point_cloud.shape[0], final_points, replace=False)

            point_cloud = point_cloud[point_indicies]

            data[data_format["Image"]] = point_cloud
            data = self.format([data, data_format])
        yield data, data_format
