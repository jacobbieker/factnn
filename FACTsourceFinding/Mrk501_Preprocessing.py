import gzip
import json
import pickle
import numpy as np
import h5py
import sys
from fact.io import read_h5py
import os

#First input: Path to the raw crab1314_folder
#Second input: Path to the list of runs to use 'Crab1314_runs_to_use.csv'
#Third input: Path to the 'hexagonal_to_quadratic_mapping_dict.p'
#Fourth input: Path to the 'crab1314_preprocessed_images.h5'
#path_raw_crab_folder = sys.argv[1]
#path_runs_to_use = sys.argv[2]
#path_store_mapping_dict = sys.argv[3]
#path_crab_images = sys.argv[4]

path_raw_crab_folder = "/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/obs/Crab/2014/"
#path_store_mapping_dict = sys.argv[2]
path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/rebinned_mapping_dict_4_flipped.p"
#path_mc_images = sys.argv[3]
path_crab_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_mrk501_precut_images.h5"
path_std_analysis = "/run/media/jacob/WDRed8Tb1/dl2_theta/Mrk501_precuts.hdf5"
path_store_runlist = "Mrk501_precuts.p"
path_diffuse = "/run/media/jacob/WDRed8Tb1/dl2_theta/Mrk501_precuts.hdf5"

# Format dataset to fit into tensorflow
def reformat(dataset):
    return dataset.reshape((-1,75, 75, 1)).astype(np.float32)


def batchYielder():
    paths = []
    if not os.path.isfile(path_store_runlist):
        diffuse_df = read_h5py(path_diffuse, key="events", columns=["event_num", "night", "run_id", "source_x_prediction", "source_y_prediction"])
        list_paths = diffuse_df["night"].values
        list_paths2 = diffuse_df['run_id'].values
        list_events = []
        for index, night in enumerate(list_paths):
            # Night and run_id are in same index, so make them pay
            list_events.append(str(night) + "_" + str(list_paths2[index]))

        # Iterate over every file in the subdirs and check if it has the right file extension
        file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(path_raw_crab_folder))
                      for file in fileName if '.json' in file]
        latest_paths = []
        for path in file_paths:
            if any(number in path for number in list_events):
                print(path)
                latest_paths.append(path)
        #Create paths to the runs to be processed

        with open(path_store_runlist, "wb") as path_store:
            pickle.dump(latest_paths, path_store)
    else:
        with open(path_store_runlist, "rb") as path_store:
            latest_paths = pickle.load(path_store)
        diffuse_df = read_h5py(path_diffuse, key="events", columns=["event_num", "night", "run_id", "source_x_prediction", "source_y_prediction"])

    print(latest_paths)
    # Load mapping-dict to switch from hexagonal to matrix
    id_position = pickle.load(open(path_store_mapping_dict, "rb"))

    for file in latest_paths:
        # It is in order of
        try:
            with gzip.open(file) as f:
                print(file)
                data = []

                for line in f:
                    line_data = json.loads(line.decode('utf-8'))
                    run = line_data['Run']
                    event = line_data['Event']
                    night = line_data['Night']
                    testing = diffuse_df.loc[(diffuse_df['event_num'] == event) & (diffuse_df['run_id'] == run) & (diffuse_df['night'] == night)]
                    if not testing.empty:
                        event_photons = line_data['PhotonArrivals_500ps']
                        zd_deg = line_data['Zd_deg']
                        az_deg = line_data['Az_deg']
                        source_x = testing['source_x_prediction']
                        source_y = testing['source_y_prediction']
                        trigger = line_data['Trigger']
                        time = line_data['UnixTime_s_us'][0] + 1e-6*line_data['UnixTime_s_us'][1]
                        input_matrix = np.zeros([75,75])
                        chid_to_pixel = id_position[0]
                        pixel_index_to_grid = id_position[1]
                        for index in range(1440):
                            for element in chid_to_pixel[index]:
                                coords = pixel_index_to_grid[element[0]]
                                input_matrix[coords[0]][coords[1]] += element[1]*len(event_photons[index])
                        data.append([np.fliplr(np.rot90(input_matrix, 3)), night, run, event, zd_deg, az_deg,
                                     trigger, time, source_x, source_y])
            if data != []:
                yield data

        except Exception as e:
            print(e)
            pass


# Change the datatype to np-arrays
def batchFormatter(batch):
    pic, night, run, event, zd_deg, az_deg, trigger, time, source_x, source_y = zip(*batch)
    pic = reformat(np.array(pic))
    night = np.array(night)
    run = np.array(run)
    event = np.array(event)
    zd_deg = np.array(zd_deg)
    az_deg = np.array(az_deg)
    trigger = np.array(trigger)
    time = np.array(time)
    source_x = np.array(source_x)
    source_y = np.array(source_y)
    return (pic, night, run, event, zd_deg, az_deg, trigger, time, source_x, source_y)


# Use the batchYielder to concatenate every batch and store it into a single h5 file

try:
    gen = batchYielder()
    batch = next(gen)
    pic, night, run, event, zd_deg, az_deg, trigger, time, source_x, source_y = batchFormatter(batch)
    row_count = trigger.shape[0]

    with h5py.File(path_crab_images, 'w') as hdf:
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
        maxshape_x = (None,) + source_x.shape[1:]
        dset_x = hdf.create_dataset('Source_X', shape=source_x.shape, maxshape=maxshape_x, chunks=source_x.shape, dtype=source_x.dtype)
        maxshape_y = (None,) + source_y.shape[1:]
        dset_y = hdf.create_dataset('Source_Y', shape=source_y.shape, maxshape=maxshape_y, chunks=source_y.shape, dtype=source_y.dtype)

        dset_pic[:] = pic
        dset_night[:] = night
        dset_run[:] = run
        dset_event[:] = event
        dset_zd_deg[:] = zd_deg
        dset_az_deg[:] = az_deg
        dset_trigger[:] = trigger
        dset_time[:] = time
        dset_x[:] = source_x
        dset_y[:] = source_y


        for batch in gen:
            pic, night, run, event, zd_deg, az_deg, trigger, time, source_x, source_y = batchFormatter(batch)

            dset_pic.resize(row_count + trigger.shape[0], axis=0)
            dset_night.resize(row_count + trigger.shape[0], axis=0)
            dset_run.resize(row_count + trigger.shape[0], axis=0)
            dset_event.resize(row_count + trigger.shape[0], axis=0)
            dset_zd_deg.resize(row_count + trigger.shape[0], axis=0)
            dset_az_deg.resize(row_count + trigger.shape[0], axis=0)
            dset_trigger.resize(row_count + trigger.shape[0], axis=0)
            dset_time.resize(row_count + trigger.shape[0], axis=0)
            dset_x.resize(row_count + trigger.shape[0], axis=0)
            dset_y.resize(row_count + trigger.shape[0], axis=0)

            dset_pic[row_count:] = pic
            dset_night[row_count:] = night
            dset_run[row_count:] = run
            dset_event[row_count:] = event
            dset_zd_deg[row_count:] = zd_deg
            dset_az_deg[row_count:] = az_deg
            dset_trigger[row_count:] = trigger
            dset_time[row_count:] = time
            dset_x[row_count:] = source_x
            dset_y[row_count:] = source_y

            row_count += trigger.shape[0]
except:
    pass
