import os
import numpy as np

'''
The goal of this is to build a list of all the runs used in the std analysis ones
'''

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
    std_base = '/dl2_theta/orecuts/std_analysis/'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/thesis'
    std_base = base_dir + "/FACTSources/std_analysis/"

np.random.seed(0)
#path_store_mapping_dict = sys.argv[2]
path_store_mapping_dict = thesis_base + "/jan/07_make_FACT/hexagonal_to_quadratic_mapping_dict.p"
#path_mc_images = "D:\Development\thesis\jan\07_make_FACT\hexagonal_to_quadratic_mapping_dict.p"

source_file_paths = []
output_paths = []


for subdir, dirs, files in os.walk(std_base):
    for file in files:
        path = os.path.join(subdir, file)
        source_file_paths.append(path)
        output_filename = file.split(".hdf5")[0]
        output_paths.append(base_dir + "/FACTSources/" + output_filename + "_preprocessed_source.hdf5")

