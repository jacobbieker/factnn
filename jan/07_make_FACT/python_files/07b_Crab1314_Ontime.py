import h5py
import numpy as np
import pandas as pd

path = '/fhgfs/users/jbehnken/make_Data/Crab1314_darknight_std_analysis_0.17.2.hdf5'
prediction_path = '/fhgfs/users/jbehnken/make_Data/crab1314_prediction.csv'

with h5py.File(path, 'r') as f:
    night = np.array(f['runs']['night'])
    run_id = np.array(f['runs']['run_id'])
    ontime = np.array(f['runs']['ontime'])
    
df = pd.read_csv(prediction_path)
df = df[['Night', 'Run']]
df = df.drop_duplicates()

theta_df = pd.DataFrame(list(zip(night, run_id, ontime)), columns=['Night', 'Run', 'Ontime'])

merge_df = df.merge(theta_df)
print(merge_df['Ontime'].sum()/3600)