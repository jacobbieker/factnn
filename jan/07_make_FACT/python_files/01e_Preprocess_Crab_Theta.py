import pandas as pd
import numpy as np
import h5py
import sys


#First input: Path to the 'Crab1314_darknight_std_analysis_0.17.2.hdf5'
#Second input: Path to the 'Theta_preprocessed.h5'
path_theta = sys.argv[1]
path_theta_preprocesses = sys.argv[2]


keys = ['event_num', 'night', 'run_id', 'theta', 'theta_deg', 'theta_deg_off_1', 'theta_deg_off_2', 'theta_deg_off_3', 'theta_deg_off_4', 'theta_deg_off_5', 'theta_off_1', 'theta_off_1_rec_pos', 'theta_off_2', 'theta_off_2_rec_pos', 'theta_off_3', 'theta_off_3_rec_pos', 'theta_off_4', 'theta_off_4_rec_pos', 'theta_off_5', 'theta_off_5_rec_pos', 'theta_rec_pos']

with h5py.File(path_theta, 'r') as f:
    data  = []
    for key in keys:
        data.append(np.array(f['events'][key]))
        
        
data = list(map(list, zip(*data)))
df = pd.DataFrame(data)
df.columns = keys

df.to_csv(path_theta_preprocesses, chunksize=1000, index=False)