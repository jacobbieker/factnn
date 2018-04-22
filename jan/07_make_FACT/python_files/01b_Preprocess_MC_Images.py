import numpy as np
import pickle
import h5py
import gzip
import json
import sys
import os


#First input: Path to the raw mc_folder
#Second input: Path to the 'hexagonal_to_quadratic_mapping_dict.p'
#Third input: Path to the 'mc_preprocessed_images.h5'
#path_raw_mc_folder = sys.argv[1]
path_raw_mc_folder = "/run/media/jacob/WDRed8Tb1/sim/"
#path_store_mapping_dict = sys.argv[2]
path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/holy_squished_mapping_dict.p"
#path_mc_images = sys.argv[3]
path_mc_images = "/run/media/jacob/WDRed8Tb1/MC_Holy_Squished_prebatched_Images.h5"


def getMetadata():
    '''
    Gathers the file paths of the training data
    '''
    # Iterate over every file in the subdirs and check if it has the right file extension
    file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(path_raw_mc_folder)) for file in fileName if '.json' in file]
    return file_paths


def reformat(dataset):
    #Reformat to fit into tensorflow
    dataset = np.array(dataset).reshape((-1, 45, 40, 1)).astype(np.float32)
    return dataset


file_paths = getMetadata()
id_position = pickle.load(open(path_store_mapping_dict, "rb"))


path_mc_gammas = [path for path in file_paths if 'gamma' in path]
path_mc_hadrons = [path for path in file_paths if 'gamma' not in path]


def batchYielder(file_paths):
    batch_size_index = 0
    event = []
    input_matrix = np.zeros([45,40])
    file_index = 0
    while batch_size_index < 1:
        try:
            for index, path in enumerate(file_paths):
                with gzip.open(file_paths[file_index]) as file:
                    file_index += 1
                    print(file_paths[file_index])

                    for line in file:
                        event_photons = json.loads(line.decode('utf-8'))['PhotonArrivals_500ps']

                        for i in range(1440):
                            x, y = id_position[i]
                            input_matrix[int(x)][int(y)] += len(event_photons[i])
                        batch_size_index += 1
                        #print(batch_size_index)
                        if batch_size_index >= 1:
                            # Add to data
                            event.append([np.flip(input_matrix, 0)])
                            input_matrix = np.zeros([45,40])
                            batch_size_index = 0
        except:
            if file_index >= len(file_paths):
                print("Index longer than path")
                break
            pass
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