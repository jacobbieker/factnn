from fact.analysis import split_on_off_source_independent, split_on_off_source_dependent
from fact.analysis.binning import bin_runs
from fact.io import read_h5py, to_h5py
import sys
import numpy as np


# Args should be: base_input_df, outputdir, start, end, step_size, output_file_name_base

# For loop to produce the different cuts
# First arg needs to be the path to the file
print("\n\n\n\n" + sys.argv[1] + "\n\n\n\n")
input_df = read_h5py(sys.argv[1], "events")
start_point = float(sys.argv[3])
end_point = float(sys.argv[4])
step_size = float(sys.argv[5])
output_name = sys.argv[6]

i = 0
j = 0

while start_point + (step_size * i) <= end_point:
    theta_value_one = start_point + (step_size * i)
    # Get the initial equal splits of data
    on_crab_events, off_crab_events = split_on_off_source_independent(input_df, theta2_cut=theta_value_one)
    ids_on = on_crab_events.index.values
    ids_off = off_crab_events.index.values


    to_h5py(filename=str(sys.argv[2]) + str(output_name) + "_on_theta" + str(np.round(theta_value_one, decimals=1)) + ".hdf5", df=on_crab_events,
            key="events")
    to_h5py(filename=str(sys.argv[2]) + str(output_name) + "_off_theta" + str(np.round(theta_value_one, decimals=1)) + ".hdf5",
            df=off_crab_events, key="events")

    while start_point + (step_size * j) <= end_point:
        theta_value_two = start_point +(step_size * j)

        # now go through, getting all the necessary permutations to use to compare to the default

        j += 1

    # remove DF of splits so it doesn't run out of memory
    del on_crab_events, off_crab_events

    i += 1  # Add one to increment
