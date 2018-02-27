import pandas as pd
import numpy as np
import h5py
import os

prediction_path = '/fhgfs/users/jbehnken/make_Data/crab1314_prediction.csv'
plotting_path = '/fhgfs/users/jbehnken/make_Data/Theta_Plotting.h5'

keys_1 = ['event_num', 'night', 'run_id', 'theta', 'theta_deg', 'theta_deg_off_1', 'theta_deg_off_2', 'theta_deg_off_3', 'theta_deg_off_4', 'theta_deg_off_5', 'theta_off_1', 'theta_off_1_rec_pos', 'theta_off_2', 'theta_off_2_rec_pos', 'theta_off_3', 'theta_off_3_rec_pos', 'theta_off_4', 'theta_off_4_rec_pos', 'theta_off_5', 'theta_off_5_rec_pos', 'theta_rec_pos', 'unix_time_utc']
keys_2 = ['run_start', 'run_stop', 'source']

with h5py.File('/net/big-tank/POOL/projects/fact/datasets/Crab1314_darknight_std_analysis_0.17.2.hdf5', 'r') as f:
    data_1  = []
    data_2  = []
    for key in keys_1:
        data_1.append(np.array(f['events'][key]))
    for key in keys_2:
        data_2.append(np.array(f['runs'][key]))
        
data_1 = list(map(list, zip(*data_1)))
df_crab = pd.DataFrame(data_1)
df_crab.columns = keys_1
del data_1

data_2 = list(map(list, zip(*data_2)))
df_crab_runs = pd.DataFrame(data_2)
df_crab_runs.columns = keys_2
del data_2

df_crab.shape


df_pred = pd.read_csv(prediction_path)
df_pred.columns = ['night', 'run_id', 'event_num', 'proton_prediction', 'gamma_prediction']
df_pred.head(2)



df_merged = pd.merge(df_pred, df_crab, how='inner', on=['event_num', 'night', 'run_id'])
df_merged.shape



with h5py.File(plotting_path, 'w') as hdf:
    events = hdf.create_group('events')
    dset_gamma_prediction = events.create_dataset('gamma_prediction', data=df_merged['gamma_prediction'].values)
    dset_theta = events.create_dataset('theta', data=df_merged['theta'].values)
    dset_theta_deg_off_1 = events.create_dataset('theta_deg_off_1', data=df_merged['theta_deg_off_1'].values)
    dset_theta_deg_off_2 = events.create_dataset('theta_deg_off_2', data=df_merged['theta_deg_off_2'].values)
    dset_theta_deg_off_3 = events.create_dataset('theta_deg_off_3', data=df_merged['theta_deg_off_3'].values)
    dset_theta_deg_off_4 = events.create_dataset('theta_deg_off_4', data=df_merged['theta_deg_off_4'].values)
    dset_theta_deg_off_5 = events.create_dataset('theta_deg_off_5', data=df_merged['theta_deg_off_5'].values)
    dset_theta_deg = events.create_dataset('theta_deg', data=df_merged['theta_deg'].values)
    dset_unix_time_utc = events.create_dataset('unix_time_utc', data=df_merged['unix_time_utc'].values.tolist())
    
    runs = hdf.create_group('runs')
    dset_run_start = runs.create_dataset('run_start', data=df_crab_runs['run_start'].values.tolist())
    dset_run_stop = runs.create_dataset('run_stop', data=df_crab_runs['run_stop'].values.tolist())
    
    
    
    