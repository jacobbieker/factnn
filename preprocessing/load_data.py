import numpy as np
import pickle
import h5py
import gzip
import json
import sys
import os

path_raw_mc_proton_folder = "/run/media/jbieker/WDRed8Tb1/sim/"
path_raw_mc_gamma_folder = "/run/media/jbieker/WDRed8Tb1/sim/"
path_store_mapping_dict = "/run/media/jbieker/SSD/Development/thesis/jan/07_make_FACT/hexagonal_to_quadratic_mapping_dict.p"
path_mc_diffuse_images = "/run/media/jbieker/WDRed8Tb1/00_MC_Diffuse_flat_Images.h5"

def getMetadata(path_folder):
    '''
    Gathers the file paths of the training data
    '''
    # Iterate over every file in the subdirs and check if it has the right file extension
    file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(path_folder)) for file in fileName if 'phs.json' in file]
    return file_paths


def reformat(dataset):
    #Reformat to fit into tensorflow
    dataset = np.array(dataset).reshape((-1, 46, 45, 1)).astype(np.float32)
    return dataset


proton_file_paths = getMetadata(path_raw_mc_proton_folder)
gamma_file_paths = getMetadata(path_raw_mc_gamma_folder)
id_position = pickle.load(open(path_store_mapping_dict, "rb"))


path_mc_hadrons = [path for path in proton_file_paths if 'proton' in path]
path_mc_gammas = gamma_file_paths


def batchYielder(file_paths):
    for path in file_paths:
        with gzip.open(path) as file:
            event = []

            for line in file:
                event_photons = json.loads(line.decode('utf-8'))['PhotonArrivals_500ps']

                input_matrix = np.zeros([46,45,100])
                for i in range(1440):
                    x, y = id_position[i]
                    for value in event_photons[i]:
                        input_matrix[int(x)][int(y)][value-30] += 1
                input_matrix = np.sum(input_matrix[:,:,5:30], axis=2)

                event.append(input_matrix)

            event = reformat(event)

            yield event