from factnn.data.preprocess.eventfile_preprocessor import EventFilePreprocessor
import pickle
import os
import numpy as np
from photon_stream.representations import list_of_lists_to_raw_phs, raw_phs_to_point_cloud
from photon_stream.geometry import GEOMETRY

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
        replacement must be True, or an exception will be thrown
        :param replacement: Whether to randomly choose points with replacement (True), or not
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
                            feature_list.append(features['length']*features['width']*np.pi)
                            feature_list.append(((features['length']*features['width']*np.pi)/np.log(features['number_photons'])**2))
                            feature_list.append((features['number_photons'] / (features['length']*features['width']*np.pi)))

                    if truncate:
                        self.end = self.start + self.shape[3]

                    # Convert List of List to Point Cloud, then truncation is simply cutting in the z direction
                    event_photons = data[data_format["Image"]]
                    event_photons = list_of_lists_to_raw_phs(event_photons)
                    point_cloud = np.asarray(raw_phs_to_point_cloud(event_photons,
                                                                    cx=GEOMETRY.x_angle,
                                                                    cy=GEOMETRY.y_angle))

                    # Now in point cloud format, truncation is just cutting off in z now
                    mask = (point_cloud[:,2] <= self.end) & (point_cloud[:,2] >= self.start)
                    point_cloud = point_cloud[mask]

                    data[data_format["Image"]] = point_cloud
                    data = self.format([data, data_format])

                temp_data = [data, data_format]
                if return_features:
                    temp_data.append(feature_list)
                all_data.append(temp_data)
        return all_data

    def event_file_processor(self, filepath, final_points=1024, replacement=False, truncate=False,
                             return_features=False,):

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
                    feature_list.append(features['length']*features['width']*np.pi)
                    feature_list.append(((features['length']*features['width']*np.pi)/np.log(features['number_photons'])**2))
                    feature_list.append((features['number_photons'] / (features['length']*features['width']*np.pi)))

            if truncate:
                self.end = self.start + self.shape[3]

            # Convert List of List to Point Cloud, then truncation is simply cutting in the z direction
            event_photons = data[data_format["Image"]]
            event_photons = list_of_lists_to_raw_phs(event_photons)
            point_cloud = np.asarray(raw_phs_to_point_cloud(event_photons,
                                                            cx=GEOMETRY.x_angle,
                                                            cy=GEOMETRY.y_angle))

            # Now in point cloud format, truncation is just cutting off in z now
            mask = (point_cloud[:,2] <= self.end) & (point_cloud[:,2] >= self.start)
            point_cloud = point_cloud[mask]

            data[data_format["Image"]] = point_cloud
            data = self.format([data, data_format])
        yield data, data_format
