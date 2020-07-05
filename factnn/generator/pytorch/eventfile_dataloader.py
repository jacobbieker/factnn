import os.path as osp
import pickle
import numpy as np
from glob import glob

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data, DataLoader

from photon_stream.representations import list_of_lists_to_raw_phs, raw_phs_to_point_cloud
from photon_stream.geometry import GEOMETRY

from factnn.utils.augment import euclidean_distance, true_sign


class EventFileDataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None, task="Separation", num_points=0):
        """
        :param task: Either 'Separation', 'Energy', or 'Disp'
        :param num_points: The number of points to have, either using points multiple times, or subselecting from the total points
        """
        super(EventFileDataset, self).__init__(root, transform, pre_transform)
        self.processed_filenames = []
        self.task = task
        self.num_points = num_points

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.processed_filenames

    def download(self):
        pass

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # load the pickled file from the disk
            if osp.getsize(raw_path) > 0:
                # Checks that file is not 0
                with open(raw_path, "rb") as pickled_event:
                    event_data, data_format, features, feature_cluster = pickle.load(pickled_event)
                    # Convert List of List to Point Cloud, then truncation is simply cutting in the z direction
                    event_photons = event_data[data_format["Image"]]
                    event_photons = list_of_lists_to_raw_phs(event_photons)
                    point_cloud = np.asarray(raw_phs_to_point_cloud(event_photons,
                                                                    cx=GEOMETRY.x_angle,
                                                                    cy=GEOMETRY.y_angle))
                    # Read data from `raw_path`.
                    if self.num_points > 0:
                        if point_cloud.shape[0] < self.num_points:
                            point_indicies = np.random.choice(point_cloud.shape[0], self.num_points, replace=True)
                        else:
                            point_indicies = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
                        point_cloud = point_cloud[point_indicies]
                    data = Data(pos=point_cloud)  # Just need x,y,z ignore derived features, padding would in dataloader
                    if "gamma" in raw_path:
                        data.event_type = torch.tensor(0, dtype=torch.int8)
                    elif "proton" in raw_path:
                        data.event_type = torch.tensor(1, dtype=torch.int8)
                    else:
                        print("No Event Type")
                        continue
                    data.energy = torch.tensor(event_data[data_format["Energy"]], dtype=torch.float)
                    data.disp = torch.tensor(true_sign(event_data[data_format['Source_X']],
                                                       event_data[data_format['Source_Y']],
                                                       event_data[data_format['COG_X']],
                                                       event_data[data_format['COG_Y']],
                                                       event_data[data_format['Delta']])* euclidean_distance(event_data[data_format['Source_X']],
                                                                event_data[data_format['Source_Y']],
                                                                event_data[data_format['COG_X']],
                                                                event_data[data_format['COG_Y']]),
                                             dtype=torch.float16)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                    self.processed_filenames.append('data_{}.pt'.format(i))
                    i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        if self.task == "Energy":
            data.y = data.energy
        elif self.task == "Disp":
            data.y = data.disp
        else:
            data.y = data.event_type
        return data
