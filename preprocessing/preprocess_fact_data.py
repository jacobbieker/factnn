import os
import gzip
import json
import pickle
import numpy as np
from multiprocessing import Pool
import h5py
import time

import photon_stream as ps
import matplotlib.pyplot as plt
import yaml


# Important variables
mc_data_path = '/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/obs/'
id_position_path = '/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/hexagonal_to_quadratic_mapping_dict.p'
temporary_path = '/run/media/jacob/Seagate/D_test'
processed_data_path = '/run/media/jacob/Seagate/C_test'

is_sim = False

def getMetadata():
    '''
    Gathers the file paths of the training data
    '''
    # Iterate over every file in the subdirs and check if it has the right file extension
    file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(mc_data_path)) for file in fileName if '.json' in file]
    return file_paths


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 46, 45, 1)).astype(np.float32)
    labels = (np.arange(2) == labels[:,None]).astype(np.float32)
    return dataset, labels

file_paths = getMetadata()
id_position = pickle.load(open(id_position_path, "rb"))

data = []
num = 0
if is_sim == False:
    for path in file_paths:
        try:
            reader = ps.EventListReader(path)
            event_list = []
            # Maybe add next thing
            event = next(reader)

            # Now list is in 3D point cloud, can sum up over arrival time if necessary
            event_list = event.photon_stream.point_cloud
            # Events are in [x-dir, y-dir, arrival time]
            # TODO Need to sum up over the time for total per link

            input_matrix = np.zeros([46,45,100])
            summed_values = []
            num_photons_per_angle = 0
            x_dir = 0.0
            y_dir = 0.0
            for i in range(event_list):
                # sum up over the x, y, dir
                if x_dir == event_list[i][0] and y_dir == event_list[i][0]:
                    # Same angle, so add photon
                    num_photons_per_angle += 1
                else:
                    # not the same, so reset
                    summed_values.append(num_photons_per_angle)
                    num_photons_per_angle = 0
                    x_dir = event_list[i][0]
                    y_dir = event_list[i][1]

            print(event_list)
            print(len(event_list))



        except:
            pass



sim_reader = ps.SimulationReader(
    photon_stream_path="/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/sim/gamma/gustav/010910.phs.jsonl.gz",
    mmcs_corsika_path="/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/sim/gamma/gustav/010910.ch.gz"
)

for event in sim_reader:
    print(event)
    pass
#plt.show()
