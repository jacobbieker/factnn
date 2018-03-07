import gzip
import json
import pickle
import numpy as np
import h5py
import sys

path_raw_crab_folder = "/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/obs/"
#path_store_mapping_dict = sys.argv[2]
path_runs_to_use = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/Crab1314_runs_to_use.csv"
path_store_mapping_dict = "/run/media/jbieker/SSD/Development/thesis/thesisTools/output/hexagon_to_cube_mapping.p"
#path_mc_images = sys.argv[3]
path_crab_images = "/run/media/jbieker/WDRed8Tb1/Cube_crab1314_preprocessed_images.h5"

# Format dataset to fit into tensorflow
def reformat(dataset):
    return dataset.reshape((-1, 40, 56, 40)).astype(np.float32)


def batchYielder():
    paths = []
    #Create paths to the runs to be processed
    with open(path_runs_to_use) as file:
        for line in file:
            # Storing the path to every run file
            l = line.split('\t')
            path = path_raw_crab_folder + l[0][:4]+'/' + l[0][4:6]+'/' + l[0][6:8]+'/' + l[0][:4]+l[0][4:6]+l[0][6:8]+'_'+l[1].strip()+'.phs.jsonl.gz'
            paths.append(path)

    # Load mapping-dict to switch from hexagonal to matrix
    id_position = pickle.load(open(path_store_mapping_dict, "rb"))

    for file in paths:
        print(file)
        try:
            with gzip.open(file) as f:
                data = []

                for line in f:
                    line_data = json.loads(line.decode('utf-8'))

                    event_photons = line_data['PhotonArrivals_500ps']
                    night = line_data['Night']
                    run = line_data['Run']
                    event = line_data['Event']
                    zd_deg = line_data['Zd_deg']
                    az_deg = line_data['Az_deg']
                    trigger = line_data['Trigger']

                    input_matrix = np.zeros([40,56,40])
                    for i in range(1440):
                        x, y, z = id_position[i][0]
                        input_matrix[int(x)][int(y)][int(z)] = len(event_photons[i])
                    data.append([input_matrix, night, run, event, zd_deg, az_deg, trigger])
            yield data

        except:
            pass


# Change the datatype to np-arrays
def batchFormatter(batch):
    pic, night, run, event, zd_deg, az_deg, trigger = zip(*batch)
    pic = reformat(np.array(pic))
    night = np.array(night)
    run = np.array(run)
    event = np.array(event)
    zd_deg = np.array(zd_deg)
    az_deg = np.array(az_deg)
    trigger = np.array(trigger)
    return (pic, night, run, event, zd_deg, az_deg, trigger)


# Use the batchYielder to concatenate every batch and store it into a single h5 file

gen = batchYielder()
batch = next(gen)
pic, night, run, event, zd_deg, az_deg, trigger = batchFormatter(batch)
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

    dset_pic[:] = pic
    dset_night[:] = night
    dset_run[:] = run
    dset_event[:] = event
    dset_zd_deg[:] = zd_deg
    dset_az_deg[:] = az_deg
    dset_trigger[:] = trigger

    for batch in gen:
        pic, night, run, event, zd_deg, az_deg, trigger = batchFormatter(batch)

        dset_pic.resize(row_count + trigger.shape[0], axis=0)
        dset_night.resize(row_count + trigger.shape[0], axis=0)
        dset_run.resize(row_count + trigger.shape[0], axis=0)
        dset_event.resize(row_count + trigger.shape[0], axis=0)
        dset_zd_deg.resize(row_count + trigger.shape[0], axis=0)
        dset_az_deg.resize(row_count + trigger.shape[0], axis=0)
        dset_trigger.resize(row_count + trigger.shape[0], axis=0)

        dset_pic[row_count:] = pic
        dset_night[row_count:] = night
        dset_run[row_count:] = run
        dset_event[row_count:] = event
        dset_zd_deg[row_count:] = zd_deg
        dset_az_deg[row_count:] = az_deg
        dset_trigger[row_count:] = trigger

        row_count += trigger.shape[0]
