import h5py
import pandas as pd
import pickle
from fact.io import read_h5py, to_h5py

pd.set_option('display.max_columns', None)

def print_attrs(name, obj):
    print(name)
    #for key, val in obj.attrs.iteritems():
    #    print("    %s: %s" % (key, val))

def get_gamma_predictions(hdf5Group, suffix):
    predictions = hdf5Group.get('events/gamma_prediction')


#f = h5py.File('../../open_crab_sample_facttools_dl2.hdf5','r')
#f = h5py.File('../../gamma_simulations_facttools_dl2.hdf5','r')

#f = h5py.File('../../train_sim_events_predictions.hdf5', "r")
d = read_h5py('../../open_sim__test.hdf5', "events")
f = read_h5py('../../open_sim__test_fromCrab.hdf5', "events")

g = read_h5py('../../open_crab__test.hdf5', "events")
h = read_h5py('../../open_crab__test_fromSim.hdf5', "events")
#terminf = h5py.File('predictions.hdf','r')

#f.visititems(print_attrs)
#print("----------------------\n")
#d.visititems(print_attrs)
print(False & False)

low_set = d['gamma_prediction'] > 0.9
high_set = f['gamma_prediction'] > 0.9

both_set = (low_set & high_set)
one_but_not_other = (low_set != high_set)

print("\nSimulation Trained")
print(d[high_set].describe())
sim_sim_df = d[high_set]
print("\nCrab Nebula Trained")
print(f[low_set].describe())
sim_crab_df = f[low_set]
print("\nBoth Have:")
print(f[both_set].describe())
sim_trained_both_df = f[both_set]
print("\nIn One but not Other:")
print(f[one_but_not_other].describe())
sim_trained_not_both_df = f[one_but_not_other]

print("Below is Crab Nebula Data Results")

high_set = g['gamma_prediction'] > 0.9
low_set = h['gamma_prediction'] > 0.9

both_set = (low_set & high_set)
one_but_not_other = (low_set != high_set)

print("\nSimulation Trained")
print(g[high_set].describe())
crab_sim_df = g[high_set]
print("\nCrab Nebula Trained")
print(h[low_set].describe())
crab_crab_df = h[low_set]
print("\nBoth Have:")
print(h[both_set].describe())
crab_trained_both_df = h[both_set]
print("\nIn One but not Other:")
print(h[one_but_not_other].describe())
crab_trained_not_both_df = h[one_but_not_other]

with open("sim_sim_df.pkl", "wb") as sim_sim_dfpkl:
    pickle.dump(sim_sim_df, sim_sim_dfpkl)
with open("sim_crab_df.pkl", "wb") as sim_crab_dfpkl:
    pickle.dump(sim_crab_df, sim_crab_dfpkl)
with open("crab_sim_df.pkl", "wb") as crab_sim_dfpkl:
    pickle.dump(crab_sim_df, crab_sim_dfpkl)
with open("crab_crab_df.pkl", "wb") as crab_crab_dfpkl:
    pickle.dump(crab_crab_df, crab_crab_dfpkl)
with open("crab_trained_both_df.pkl", "wb") as crab_trained_both_dfpkl:
    pickle.dump(crab_trained_both_df, crab_trained_both_dfpkl)
with open("crab_trained_not_both_df.pkl", "wb") as crab_trained_not_both_dfpkl:
    pickle.dump(crab_trained_not_both_df, crab_trained_not_both_dfpkl)
with open("sim_trained_both_df.pkl", "wb") as fileopen:
    pickle.dump(sim_trained_both_df, fileopen)
with open("sim_trained_not_both_df.pkl", "wb") as sim_trained_not_both_dfpkl:
    pickle.dump(sim_trained_not_both_df, sim_trained_not_both_dfpkl)
