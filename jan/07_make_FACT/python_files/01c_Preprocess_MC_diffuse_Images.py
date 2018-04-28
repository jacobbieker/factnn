import numpy as np
import pickle
import h5py
import gzip
import json
import sys
import os

from fact.io import read_h5py


#First input: Path to the raw mc_folder
#Second input: Path to the raw_mc_diffuse_folder
#Third input: Path to the 'hexagonal_to_quadratic_mapping_dict.p'
#Fourth input: Path to the 'mc_preprocessed_images.h5'
#path_raw_mc_proton_folder = sys.argv[1]
#path_raw_mc_gamma_folder = sys.argv[2]
#path_store_mapping_dict = sys.argv[3]
#path_mc_diffuse_images = sys.argv[4]

path_raw_mc_proton_folder = "/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/proton/"
path_raw_mc_gamma_folder = "/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/gamma/"
#path_store_mapping_dict = sys.argv[2]
path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/rebinned_mapping_dict_4_flipped.p"
#path_mc_images = sys.argv[3]
path_mc_diffuse_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Diffuse_Images.h5"
path_diffuse = "/run/media/jacob/SSD/Development/open_crab_sample_analysis/dl2/gamma_diffuse.hdf5"


def getMetadata(path_folder):
    '''
    Gathers the file paths of the training data
    '''
    if "gamma" in path_folder:
        diffuse_df = read_h5py(path_diffuse, key="events", columns=["@source"])

        list_paths = diffuse_df["@source"].values

        gustav_paths = []
        werner_paths = []
        for path in list_paths:
            new_path = path.split("diffuse/gamma_")[1]
            sub_one = new_path.split("/")[0]
            sub_two = new_path.split("/")[1]
            if "werner" in sub_one:
                werner_paths.append(sub_two)
            elif "gustav" in sub_one:
                gustav_paths.append(sub_two)
            else:
                print("Problem, not in either")
        # Iterate over every file in the subdirs and check if it has the right file extension
        file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(path_folder))
                      for file in fileName if '.json' in file]
        file_paths = file_paths
        latest_paths = []
        for path in file_paths:
            if "gustav" in path:
                if any(number in path for number in gustav_paths):
                    latest_paths.append(path)
            elif "werner" in path:
                if any(number in path for number in werner_paths):
                    latest_paths.append(path)
        print(latest_paths)
    else:
        file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(path_folder))
                      for file in fileName if '.json' in file]
        latest_paths = file_paths
    return latest_paths


def reformat(dataset):
    #Reformat to fit into tensorflow
    dataset = np.array(dataset).reshape((-1, 75, 75, 1)).astype(np.float32)
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

                input_matrix = np.zeros([75,75])
                chid_to_pixel = id_position[0]
                pixel_index_to_grid = id_position[1]
                for index in range(1440):
                    for element in chid_to_pixel[index]:
                        coords = pixel_index_to_grid[element[0]]
                        input_matrix[coords[0]][coords[1]] += element[1]*len(event_photons[index])

                event.append(np.fliplr(np.rot90(input_matrix, 3)))
            
            event = reformat(event)
            
            yield event
            
            
# Use the batchYielder to concatenate every batch and store it into one h5 file
gamma_gen = batchYielder(path_mc_gammas)
gamma = next(gamma_gen)
gamma_row_count = gamma.shape[0]

with h5py.File(path_mc_diffuse_images, 'w') as hdf:
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

with h5py.File(path_mc_diffuse_images, 'a') as hdf:
    maxshape_hadron = (None,) + hadron.shape[1:]
    dset_hadron = hdf.create_dataset('Hadron', shape=hadron.shape, maxshape=maxshape_hadron, chunks=hadron.shape, dtype=hadron.dtype)
    
    dset_hadron[:] = hadron
    
    for hadron in hadron_gen:
        dset_hadron.resize(hadron_row_count + hadron.shape[0], axis=0)
        dset_hadron[hadron_row_count:] = hadron
        
        hadron_row_count += hadron.shape[0]