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

    def __init__(self, root, transform=None, pre_transform=None):
        super(EventFileDataset, self).__init__(root, transform, pre_transform)
        self.processed_filenames = []

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
                    data, data_format, features, feature_cluster = pickle.load(pickled_event)
                    # Convert List of List to Point Cloud, then truncation is simply cutting in the z direction
                    event_photons = data[data_format["Image"]]
                    event_photons = list_of_lists_to_raw_phs(event_photons)
                    point_cloud = np.asarray(raw_phs_to_point_cloud(event_photons,
                                                                    cx=GEOMETRY.x_angle,
                                                                    cy=GEOMETRY.y_angle))
                    # Read data from `raw_path`.
                    data = Data(pos=point_cloud)  # Just need x,y,z ignore derived features, padding would in dataloader
                    if "gamma" in raw_path:
                        data.event_type = torch.tensor(0, dtype=torch.int8)
                    elif "proton" in raw_path:
                        data.event_type = torch.tensor(1, dtype=torch.int8)
                    data.energy = torch.tensor(data[data_format["Energy"]], dtype=torch.float)
                    data.disp = torch.tensor(euclidean_distance(data[data_format['Source_X']], data[data_format['Source_Y']],
                                                                data[data_format['COG_X']], data[data_format['COG_Y']]),
                                             dtype=torch.float16)
                    data.sign = torch.tensor(true_sign(data[data_format['Source_X']], data[data_format['Source_Y']],
                                                       data[data_format['COG_X']], data[data_format['COG_Y']],
                                                       data[data_format['Delta']]), dtype=torch.uint8)

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
        return data
