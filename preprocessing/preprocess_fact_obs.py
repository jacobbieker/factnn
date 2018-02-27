from multiprocessing import Pool
import photon_stream as ps
import numpy as np
import operator
import random
import pickle
import gzip
import json
import os
import h5py
import matplotlib.pyplot as plt

# Important variables
mc_data_path = '/run/media/jbieker/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/sim/'
id_position_path = '/run/media/jbieker/SSD/Development/thesis/thesisTools/output/hexagon_to_cube_mapping.p'
path_mc_images = '/run/media/jbieker/Seagate/MC_Cube_Images.h5'

def getMetadata():
    '''
    Gathers the file paths of the training data
    '''
    # Iterate over every file in the subdirs and check if it has the right file extension
    file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(mc_data_path)) for file in fileName if 'phs.json' in file]
    return file_paths


def reformat(dataset):
    #Reformat to fit into tensorflow
    dataset = np.array(dataset).reshape((-1, 40, 56, 40)).astype(np.float32)
    return dataset

file_paths = getMetadata()
id_position = pickle.load(open(id_position_path, "rb"))

path_mc_gammas = [path for path in file_paths if 'gamma' in path]
path_mc_hadrons = [path for path in file_paths if 'gamma' not in path]

'''
for path in file_paths:
        # Gamma=True, Proton=False
        label = True if 'gamma' in path else False
        try:
            reader = ps.EventListReader(path)
            # Now iterate through the Event List
            event = next(reader)
            while event:
                print(path)
                print(event.photon_stream.point_cloud)
                print(ps.PhotonStream.list_of_lists)
                ps.plot.event(event)
                plt.show()
                event = next(reader)
                # In the loop, map photons to the position in the detector space
        except:
            print(path)
            pass
'''

def batchYielder(file_paths):
    for path in file_paths:
        label = True if 'gamma' in path else False
        with gzip.open(path) as file:
            event = []
            print(path)

            for line in file:
                event_photons = json.loads(line.decode('utf-8'))['PhotonArrivals_500ps']

                input_matrix = np.zeros([40,56,40]) #x, y, z +2ish
                for i in range(1440):
                    x, y, z = id_position[i][0]
                    # Adds length of the event photons for a line, so the number of photons per pixel for one event
                    # Each line is an event
                    input_matrix[int(x)][int(y)][int(z)] = len(event_photons[i])

                event.append(input_matrix)

            event = reformat(event)

            yield event


# Use the batchYielder to concatenate every batch and store it into one h5 file
gamma_gen = batchYielder(path_mc_gammas)
gamma = next(gamma_gen)
gamma_row_count = gamma.shape[0]

with h5py.File(path_mc_images, 'w') as hdf:
    maxshape_gamma = (None,) + gamma.shape[1:]
    dset_gamma = hdf.create_dataset('Gamma', shape=gamma.shape, maxshape=maxshape_gamma, chunks=gamma.shape, dtype=gamma.dtype)

    dset_gamma[:] = gamma

    for gamma in gamma_gen:
        dset_gamma.resize(gamma_row_count + gamma.shape[0], axis=0)
        dset_gamma[gamma_row_count:] = gamma

        gamma_row_count += gamma.shape[0]



    # Use the batchYielder to concatenate every batch and store it into one h5 file
hadron_gen = batchYielder(path_mc_hadrons)
hadron = next(hadron_gen)
hadron_row_count = hadron.shape[0]

with h5py.File(path_mc_images, 'a') as hdf:
    maxshape_hadron = (None,) + hadron.shape[1:]
    dset_hadron = hdf.create_dataset('Hadron', shape=hadron.shape, maxshape=maxshape_hadron, chunks=hadron.shape, dtype=hadron.dtype)

    dset_hadron[:] = hadron

    for hadron in hadron_gen:
        dset_hadron.resize(hadron_row_count + hadron.shape[0], axis=0)
        dset_hadron[hadron_row_count:] = hadron

        hadron_row_count += hadron.shape[0]
