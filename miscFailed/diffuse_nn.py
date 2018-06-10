import photon_stream as ps
from photon_stream import SimulationReader
import os
import numpy as np
import pickle


path_raw_mc_proton_folder = "/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/proton/"
path_raw_mc_gamma_folder = "/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/gamma/"
#path_store_mapping_dict = sys.argv[2]
path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/hexagonal_to_quadratic_mapping_dict.p"
#path_mc_images = sys.argv[3]
path_mc_diffuse_images = "/run/media/jacob/WDRed8Tb1/00_MC_Diffuse_Images.h5"



def getMetadata(path_folder):
    '''
    Gathers the file paths of the training data
    '''
    # Iterate over every file in the subdirs and check if it has the right file extension
    file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(path_folder)) for file in fileName if '.json' in file]
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


for event in path_mc_gammas:
    current_event_path = event.split(".phs")[0]
    sim_reader = SimulationReader(
        photon_stream_path=event,
        mmcs_corsika_path=current_event_path+".ch.gz"
    )

    print(sim_reader.thrown_events()[24])
    for sub_event in sim_reader:
        # Theta is Zenith, Phi is Azimuth
        print(sub_event)
        sub_event
        exit(1)