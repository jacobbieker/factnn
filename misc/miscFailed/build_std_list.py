import os
import numpy as np
from fact.io import read_h5py

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

# Can use the zd and az from the photon stream as the point direction for the telescope, so only need the source zd and az for the training



for subdir, dirs, files in os.walk(std_base):
    for file in files:
        path = os.path.join(subdir, file)
        source_file_paths.append(path)
        output_filename = file.split(".hdf5")[0]
        output_paths.append(base_dir + "/FACTSources/" + output_filename + "_essentials.hdf5")


for index, source_file in enumerate(source_file_paths):
    print(output_paths[index])
    if os.path.isfile(output_paths[index]):
        os.remove(output_paths[index])
    list_of_used_sources = []
    all_data_from_source = []
    print(source_file)
    columns = ['cong_x', 'cog_y', 'length', 'width', 'size', 'unix_time_utc', 'timestamp', 'source_position_zd',
               'source_position_az', 'theta_deg',]
    run_df = read_h5py(file_path=source_file, key='runs', columns=['night', 'run_id', 'source'])
    if "0.17.2" in source_file:
        info_df = read_h5py(file_path=source_file, key='events', columns=(columns + ['source_position', 'unix_time_utc', 'zd_pointing', 'az_pointing']))
    else:
        info_df = read_h5py(file_path=source_file, key='events', columns=(columns + ['source_position_x', 'source_position_y', 'pointing_position_zd', 'pointing_position_az', ]))