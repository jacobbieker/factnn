from fact.analysis import split_on_off_source_independent, split_on_off_source_dependent
from fact.analysis.binning import bin_runs
from fact.io import read_h5py, to_h5py
import sys
import numpy as np
from math import ceil


# Args should be: base_input_df, outputdir, start, end, step_size, output_file_name_base

def split_indices(idx, n_total, fractions):
    '''
    splits idx containing n_total distinct events into fractions given in fractions list.
    returns the number of events in each split
    '''
    print("Split Indicies")
    num_ids = [ceil(n_total * f) for f in fractions]
    print("Start Num Ids:" + str(num_ids))
    if sum(num_ids) > n_total:
        num_ids[-1] -= sum(num_ids) - n_total
    print("Final num_ids" + str(num_ids))
    return num_ids

# For loop to produce the different cuts
# First arg needs to be the path to the file
input_df = read_h5py(sys.argv[1], "events")

print("\n\nLength of all events: " + str(len(input_df)))
copied_events_on = np.unique(input_df.index.values, return_index=True)
print("Difference between before and after: " + str(len(input_df) - len(copied_events_on[0])) + "\n\n")

start_point = float(sys.argv[3])
end_point = float(sys.argv[4])
step_size = float(sys.argv[5])
output_name = sys.argv[6]

# Random seed set
np.random.seed(0)

i = 0
j = 0

while start_point + (step_size * i) <= end_point:
    theta_value_one = start_point + (step_size * i)
    # Get the initial equal splits of data
    on_crab_events, off_crab_events = split_on_off_source_independent(input_df, theta2_cut=theta_value_one)

    print("On Events Before: " + str(len(on_crab_events)))
    copied_events_on = np.unique(on_crab_events.index.values, return_index=True)
    print("Difference between before and after: " + str(len(on_crab_events) - len(copied_events_on[0])))
    print("On Event Copied Max: " + str(np.max(copied_events_on[1])))
    print("Copied Events " + str(copied_events_on))
    #on_crab_events = np.delete(on_crab_events, copied_events_on)
    print("On Events After: " + str(len(copied_events_on[0])))

    print("Off Events Before: " + str(len(off_crab_events)))
    copied_events_off = np.unique(off_crab_events.index.values, return_index=True)
    print("Difference between before and after: " + str(len(off_crab_events) - len(copied_events_off[0])))
    print("Off Event Copied Max: " + str(np.max(copied_events_off[1])))
    print("Copied Events " + str(copied_events_off))
    #off_crab_events = np.delete(off_crab_events, copied_events_off)
    print("Off Events After: " + str(len(copied_events_off[0])))

    # Cut to only unique elements
    print("Number of Columns: " + str(len(on_crab_events)))
    on_crab_events = on_crab_events.iloc[copied_events_on[1]]
    print("Number of Columns: " + str(len(on_crab_events)))
    off_crab_events = off_crab_events.iloc[copied_events_off[1]]

    # Split into different fractions
    id_on = on_crab_events.index.values
    id_off = off_crab_events.index.values
    n_on_total = len(on_crab_events)
    n_off_total = len(off_crab_events)

    num_id_on = split_indices(id_on, n_on_total, fractions=[0.7,0.3])

    print("\n\n Theta: " + str(theta_value_one) + "\n\n")

    for n, part_name in zip(num_id_on, ["_train", "_test"]):
        selected_ids = np.random.choice(id_on, size=n, replace=False)
        selected_data = on_crab_events.loc[selected_ids]
        to_h5py(filename=str(sys.argv[2]) + str(output_name) + "_on_theta" + str(np.round(theta_value_one, decimals=2)) + part_name + ".hdf5", df=selected_data,
                key="events")
        on_crab_events = on_crab_events.loc[list(set(on_crab_events.index.values) - set(selected_data.index.values))]
        id_on = on_crab_events.index.values

    # Split into different fractions

    num_id_off = split_indices(id_off, n_off_total, fractions=[0.7,0.3])

    for n, part_name in zip(num_id_off, ["_train", "_test"]):
        selected_ids = np.random.choice(id_off, size=n, replace=False)
        selected_data = off_crab_events.loc[selected_ids]
        to_h5py(filename=str(sys.argv[2]) + str(output_name) + "_off_theta" + str(np.round(theta_value_one, decimals=2)) + part_name + ".hdf5", df=selected_data,
                key="events")
        off_crab_events = off_crab_events.loc[list(set(off_crab_events.index.values) - set(selected_data.index.values))]
        id_off = off_crab_events.index.values

    #to_h5py(filename=str(sys.argv[2]) + str(output_name) + "_on_theta" + str(np.round(theta_value_one, decimals=2)) + ".hdf5", df=on_crab_events,
    #        key="events")
    #to_h5py(filename=str(sys.argv[2]) + str(output_name) + "_off_theta" + str(np.round(theta_value_one, decimals=2)) + ".hdf5",
    #        df=off_crab_events, key="events")


    while start_point + (step_size * j) <= end_point:
        theta_value_two = start_point +(step_size * j)

        # now go through, getting all the necessary permutations to use to compare to the default

        j += 1

    # remove DF of splits so it doesn't run out of memory
    del on_crab_events, off_crab_events
    print("End of Theta")
    i += 1  # Add one to increment
