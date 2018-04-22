from fact.io import read_h5py, to_h5py
import sys
import numpy as np
import gzip
import json
import h5py
import pickle

from fact.factdb import connect_database, RunInfo, get_ontime_by_source_and_runtype, get_ontime_by_source, Source, RunType, AnalysisResultsRunISDC, AnalysisResultsRunLP
from fact.credentials import get_credentials
import os
import datetime
import matplotlib.pyplot as plt


np.random.seed(0)
#path_store_mapping_dict = sys.argv[2]
path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/rebinned_mapping_dict.p"
#path_mc_images = "D:\Development\thesis\jan\07_make_FACT\hexagonal_to_quadratic_mapping_dict.p"

source_file_paths = []
output_paths = []

#source_file_paths.append("runs/Crab.csv")
#output_paths.append("/run/media/jacob/WDRed8Tb1/FACTSources/Crab_background_preprocessed_images.h5")
#source_file_paths.append("runs/Mrk 421.csv")
#source_file_paths.append("runs/Mrk 501.csv")

not_enough = 0

# Build list of source csv files to go through
for subdir, dirs, files in os.walk("runs/"):
    for file in files:
        if ".csv" in file:
            path = os.path.join(subdir, file)
            source_file_paths.append(path)
            output_filename = file.split(".csv")[0]
            output_paths.append("/run/media/jacob/WDRed8Tb2/FACTSources/" + output_filename + "_background_preprocessed_images.h5")

# Format dataset to fit into tensorflow
def reformat(dataset):
    return dataset.reshape((-1, 186, 186, 1)).astype(np.float32)


def batchYielder(path_runs_to_use):
    paths = []
    #Create paths to the runs to be processed
    with open(path_runs_to_use) as file:
        # Select the paths from each file that are data paths for training
        # Crab had 20 million events in 1344 files, so maybe go for 2000 files for each, randomly chosen from 2013 or later
        for line in file:
            if "/obs/2011" not in line and "/obs/2012" not in line:
                # Storing the path to every run file
                l = line.split('\n')[0]
                paths.append(l)

    #print(paths)
    # Now select a subset of those paths to use
    num_of_files_to_use = 2000
    try:
        used_list = np.random.choice(paths, size=num_of_files_to_use, replace=False)
        used_list = paths[0:num_of_files_to_use]
        not_enough = 0
    except:
        try:
            used_list = np.random.choice(paths, size=int(num_of_files_to_use/2), replace=False)
            used_list = paths[0:int(num_of_files_to_use/2)]
            not_enough = 0
        except:
            try:
                used_list = np.random.choice(paths, size=int(num_of_files_to_use/4), replace=False)
                used_list = paths[0:int(num_of_files_to_use/4)]
                not_enough = 0
            except:
                print("Not Enough Events")
                not_enough = 1
                used_list = np.random.choice(paths, size=1, replace=False)
    # Load mapping-dict to switch from hexagonal to matrix
    id_position = pickle.load(open(path_store_mapping_dict, "rb"))

    batch_size_index = 0
    data = []
    input_matrix = np.zeros([186,186])
    file_index = 0
    #while batch_size_index < 5000:
    try:
        for index, file in enumerate(used_list):
            with gzip.open(used_list[file_index]) as f:
                print(used_list[file_index])
                file_index += 1

                for line in f:
                    line_data = json.loads(line.decode('utf-8'))

                    event_photons = line_data['PhotonArrivals_500ps']
                    night = line_data['Night']
                    run = line_data['Run']
                    event = line_data['Event']
                    zd_deg = line_data['Zd_deg']
                    az_deg = line_data['Az_deg']
                    trigger = line_data['Trigger']
                    time = line_data['UnixTime_s_us'][0] + 1e-6*line_data['UnixTime_s_us'][1]

                    if trigger == 1024: # Code for background only trigger
                        #print("Trigger type is " + str(trigger))
                        input_matrix = np.zeros([186,186])
                        chid_to_pixel = id_position[0]
                        pixel_index_to_grid = id_position[1]
                        for i in range(1440):
                            for element in chid_to_pixel[i]:
                                coords = pixel_index_to_grid[element[0]]
                                input_matrix[coords[0]][coords[1]] += element[1]*len(event_photons[i])
                        #batch_size_index += 1
                        #if batch_size_index >= 5000:
                        #    print("Batch Size Reached")
                        #    # Add to data
                            data.append([np.fliplr(np.rot90(input_matrix, 3)), night, run, event, zd_deg, az_deg, trigger, time])
                        #    plt.imshow(input_matrix)
                        #    plt.show()
                            input_matrix = np.zeros([186,186])
                        #    batch_size_index = 0
            print("Data")
            yield data

    except:
     #   print(e)
        if file_index >= len(used_list):
            print("Overrun events")
            data.append([np.fliplr(np.rot90(input_matrix, 3)), night, run, event, zd_deg, az_deg, trigger, time])
            input_matrix = np.zeros([186,185])
            batch_size_index = 0
            #break
        pass

   # yield data


# Change the datatype to np-arrays
def batchFormatter(batch):
    pic, night, run, event, zd_deg, az_deg, trigger, time = zip(*batch)
    pic = reformat(np.array(pic))
    night = np.array(night)
    run = np.array(run)
    event = np.array(event)
    zd_deg = np.array(zd_deg)
    az_deg = np.array(az_deg)
    trigger = np.array(trigger)
    time = np.array(time)
    return (pic, night, run, event, zd_deg, az_deg, trigger, time)


# Use the batchYielder to concatenate every batch and store it into a single h5 file

for index, source_file in enumerate(source_file_paths):
    print(output_paths[index])
    if not os.path.isfile(output_paths[index]):
        gen = batchYielder(source_file)
        batch = next(gen)
        pic, night, run, event, zd_deg, az_deg, trigger, time = batchFormatter(batch)
        row_count = trigger.shape[0]

        with h5py.File(output_paths[index], 'w') as hdf:
            maxshape_pic = (None,) + pic.shape[1:]
            dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape_pic, chunks=pic.shape, dtype=pic.dtype)
            maxshape_night = (None,) + night.shape[1:]
            dset_night = hdf.create_dataset('Night', shape=night.shape, maxshape=maxshape_night, chunks=night.shape, dtype=night.dtype)
            maxshape_run = (None,) + run.shape[1:]
            dset_run = hdf.create_dataset('Run', shape=run.shape, maxshape=maxshape_run, chunks=run.shape, dtype=run.dtype)
            maxshape_event = (None,) + event.shape[1:]
            dset_event = hdf.create_dataset('Event', shape=event.shape, maxshape=maxshape_event, chunks=event.shape, dtype=event.dtype)
            maxshape_zd_deg = (None,) + zd_deg.shape[1:]
            dset_zd_deg = hdf.create_dataset('Zd_deg', shape=zd_deg.shape, maxshape=maxshape_zd_deg, chunks=zd_deg.shape, dtype=zd_deg.dtype)
            maxshape_az_deg = (None,) + az_deg.shape[1:]
            dset_az_deg = hdf.create_dataset('Az_deg', shape=az_deg.shape, maxshape=maxshape_az_deg, chunks=az_deg.shape, dtype=az_deg.dtype)
            maxshape_trigger = (None,) + trigger.shape[1:]
            dset_trigger = hdf.create_dataset('Trigger', shape=trigger.shape, maxshape=maxshape_trigger, chunks=trigger.shape, dtype=trigger.dtype)
            maxshape_time = (None,) + time.shape[1:]
            dset_time = hdf.create_dataset('Time', shape=time.shape, maxshape=maxshape_time, chunks=time.shape, dtype=time.dtype)

            dset_pic[:] = pic
            dset_night[:] = night
            dset_run[:] = run
            dset_event[:] = event
            dset_zd_deg[:] = zd_deg
            dset_az_deg[:] = az_deg
            dset_trigger[:] = trigger
            dset_time[:] = time

            for batch in gen:
                pic, night, run, event, zd_deg, az_deg, trigger, time = batchFormatter(batch)

                dset_pic.resize(row_count + trigger.shape[0], axis=0)
                dset_night.resize(row_count + trigger.shape[0], axis=0)
                dset_run.resize(row_count + trigger.shape[0], axis=0)
                dset_event.resize(row_count + trigger.shape[0], axis=0)
                dset_zd_deg.resize(row_count + trigger.shape[0], axis=0)
                dset_az_deg.resize(row_count + trigger.shape[0], axis=0)
                dset_trigger.resize(row_count + trigger.shape[0], axis=0)
                dset_time.resize(row_count + trigger.shape[0], axis=0)

                dset_pic[row_count:] = pic
                dset_night[row_count:] = night
                dset_run[row_count:] = run
                dset_event[row_count:] = event
                dset_zd_deg[row_count:] = zd_deg
                dset_az_deg[row_count:] = az_deg
                dset_trigger[row_count:] = trigger
                dset_time[row_count:] = time

                row_count += trigger.shape[0]