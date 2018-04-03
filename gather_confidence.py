from fact.analysis import split_on_off_source_independent, split_on_off_source_dependent
from fact.analysis.binning import bin_runs
from fact.io import read_h5py, to_h5py
import sys
import numpy as np
from math import ceil
from os import walk
import os
import pandas as pd
import pickle
import matplotlib as plt

file_directory = "/run/media/jacob/WDRed8Tb1/dl2_theta/test_files"
# First arg needs to be the path to the file
#input_df = read_h5py(sys.argv[1], "events")

#output_name = sys.argv[2]


# SO now use the indicies as event_numbers for both gamma. proton, and crab precuts data
# For each file, extract the index and gamma_pred value, if it is the gamma one, also set it that it is equal to 1, proton one = 0

#gammaness = input_df['gamma_prediction']

crab_predictions = {}
gamma_predictions = {}
crab_test_predictions = {}
gamma_test_predictions = {}
proton_predictions = {}
proton_test_predictions = {}

files_names = []
for (dirpath, dirnames, filenames) in walk(file_directory):
    files_names.extend(filenames)
    break

print(files_names)
# Now go through all the files, pulling out the prediction values
for name in files_names:
    try:
        if "crab_precuts" in name:
            print(name)
            crab_df = read_h5py(os.path.join(file_directory, name), key='events')
            gammaness = list(crab_df['gamma_prediction'].values)
            crab_predictions[name.split('.hdf5')[0]] = gammaness
            del crab_df
    except:
        print("\n\n Failed at: " + name + "\n\n")

    '''
    elif "gamma_precuts" in name:
        print(name)
        gamma_df = read_h5py(os.path.join(file_directory, name), key='events')
        gammaness = list(gamma_df['gamma_prediction'].values)
        gamma_predictions[name.split('_gamma_precuts.hdf5')[0]] = gammaness
        del gamma_df
    elif "proton_precuts" in name:
        print(name)
        gamma_df = read_h5py(os.path.join(file_directory, name), key='events')
        gammaness = list(gamma_df['gamma_prediction'].values)
        proton_predictions[name.split('_proton_precuts.hdf5')[0]] = gammaness
        del gamma_df
    '''

# Now have them all in six dictionaries, can now

predicitons_df = pd.DataFrame(crab_predictions)
to_h5py(predicitons_df, "/run/media/jacob/WDRed8Tb1/gamma_predictions.hdf5", key="data")
'''
with open("dl2/gamma_predictions.pkl", "wb") as pickler:
    pickle.dump(gamma_predictions, pickler, protocol=pickle.HIGHEST_PROTOCOL)
with open("dl2/proton_predictions.pkl", "wb") as pickler:
    pickle.dump(proton_predictions, pickler, protocol=pickle.HIGHEST_PROTOCOL)

for name in files_names:
    if "crab_test" in name:
        print(name)
        crab_df = read_h5py(os.path.join(file_directory, name), key='events')
        gammaness = list(crab_df['gamma_prediction'].values)
        crab_test_predictions[name.split('_crab_test.hdf5')[0]] = gammaness
        del crab_df
    elif "proton_test" in name:
        print(name)
        gamma_df = read_h5py(os.path.join(file_directory, name), key='events')
        gammaness = list(gamma_df['gamma_prediction'].values)
        proton_test_predictions[name.split('_proton_test.hdf5')[0]] = gammaness
        del gamma_df
    elif "gamma_test" in name:
        print(name)
        gamma_df = read_h5py(os.path.join(file_directory, name), key='events')
        gammaness = list(gamma_df['gamma_prediction'].values)
        gamma_test_predictions[name.split('_gamma_test.hdf5')[0]] = gammaness
        del gamma_df

with open("dl2/crab_test_predictions.pkl", "wb") as pickler:
    pickle.dump(crab_test_predictions, pickler, protocol=pickle.HIGHEST_PROTOCOL)
with open("dl2/gamma_test_predictions.pkl", "wb") as pickler:
    pickle.dump(gamma_test_predictions, pickler, protocol=pickle.HIGHEST_PROTOCOL)
with open("dl2/proton_test_predictions.pkl", "wb") as pickler:
    pickle.dump(proton_test_predictions, pickler, protocol=pickle.HIGHEST_PROTOCOL)

# First go through and compare gamma predictions to actual gamma events, graph number of those over 0.7 and 0.85

true_gamma_df = read_h5py(os.path.join("/run/media/jacob/WDRed8Tb1/dl2_theta", "gamma_precuts.hdf5"), key="events")
true_gamma_num = len(true_gamma_df)
del true_gamma_df
true_crab_df = read_h5py(os.path.join("/run/media/jacob/WDRed8Tb1/dl2_theta", "crab_precuts.hdf5"), key="events")
true_crab_num = len(true_crab_df)
del true_gamma_df
true_proton_df = read_h5py(os.path.join("/run/media/jacob/WDRed8Tb1/dl2_theta", "proton_precuts.hdf5"), key="events")
true_proton_num = len(true_proton_df)
del true_proton_df

'''