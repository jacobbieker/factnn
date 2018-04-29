import numpy as np
import pickle
import h5py
import gzip
import json
import sys
import os

from fact.io import read_h5py


#First input: Path to the raw mc_folder
#Second input: Path to the raw_mc_diffuse_folder
#Third input: Path to the 'hexagonal_to_quadratic_mapping_dict.p'
#Fourth input: Path to the 'mc_preprocessed_images.h5'
#path_raw_mc_proton_folder = sys.argv[1]
#path_raw_mc_gamma_folder = sys.argv[2]
#path_store_mapping_dict = sys.argv[3]
#path_mc_diffuse_images = sys.argv[4]

path_raw_mc_proton_folder = "/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/proton/"
path_raw_mc_gamma_folder = "/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/gamma/"
#path_store_mapping_dict = sys.argv[2]
path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/rebinned_mapping_dict_4_flipped.p"
#path_mc_images = sys.argv[3]
path_mc_diffuse_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Gamma_Images.h5"
path_mc_proton_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Proton_Images.h5"
path_diffuse = "/run/media/jacob/Seagate/open_crab_sample_analysis/dl2/gamma.hdf5"
path_proton = "/run/media/jacob/Seagate/open_crab_sample_analysis/dl2/proton.hdf5"


def getMetadata(path_folder, path_type):
    '''
    Gathers the file paths of the training data
    '''
    diffuse_df = read_h5py(path_type, key="events", columns=["event_num", "run_id", "source_position_az",
                                                                "source_position_zd", "pointing_position_az",
                                                                "pointing_position_zd"])

    list_paths = diffuse_df["run_id"].values
    gustav_paths = []
    for num in list_paths:
        gustav_paths.append((str(num)))

    # Iterate over every file in the subdirs and check if it has the right file extension
    file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(path_folder))
                  for file in fileName if '.json' in file]
    file_paths = file_paths
    latest_paths = []
    for path in file_paths:
        if any(number in path for number in gustav_paths):
            latest_paths.append(path)
    print(latest_paths)
    return latest_paths, diffuse_df


def reformat(dataset):
    #Reformat to fit into tensorflow
    dataset = np.array(dataset).reshape((-1, 75, 75, 1)).astype(np.float32)
    return dataset


gamma_file_paths, diffuse_df = getMetadata(path_raw_mc_gamma_folder, path_diffuse)
id_position = pickle.load(open(path_store_mapping_dict, "rb"))


path_mc_gammas = gamma_file_paths

def batchYielder(paths, diffuse_df):

    # Load mapping-dict to switch from hexagonal to matrix
    id_position = pickle.load(open(path_store_mapping_dict, "rb"))

    for file in paths:
        try:
            with gzip.open(file) as f:
                print(file)
                data = []

                for line in f:
                    line_data = json.loads(line.decode('utf-8'))
                    run = line_data['Run']
                    event = line_data['Event']

                    if event in diffuse_df['event_num'].values and run in diffuse_df['run_id'].values:
                        element = diffuse_df[(diffuse_df['run_id'] == run) & (diffuse_df['event_num'] == event)]
                        if not element.empty:
                            event_photons = line_data['PhotonArrivals_500ps']
                            #print(element)
                            az_deg = element["source_position_az"].values
                            zd_deg = element["source_position_zd"].values

                            phi = element["pointing_position_az"].values
                            theta = element["pointing_position_zd"].values

                            input_matrix = np.zeros([75,75])
                            chid_to_pixel = id_position[0]
                            pixel_index_to_grid = id_position[1]
                            for index in range(1440):
                                for element in chid_to_pixel[index]:
                                    coords = pixel_index_to_grid[element[0]]
                                    input_matrix[coords[0]][coords[1]] += element[1]*len(event_photons[index])
                            data.append([np.fliplr(np.rot90(input_matrix, 3)), run, event, zd_deg, az_deg, phi, theta])
            yield data

        except Exception as e:
            print(str(e))


# Use the batchYielder to concatenate every batch and store it into one h5 file
# Change the datatype to np-arrays
def batchFormatter(batch):
    pic, run, event, zd_deg, az_deg, trigger, time = zip(*batch)
    pic = reformat(np.array(pic))
    run = np.array(run)
    event = np.array(event)
    zd_deg = np.array(zd_deg)
    az_deg = np.array(az_deg)
    trigger = np.array(trigger)
    time = np.array(time)
    return (pic, run, event, zd_deg, az_deg, trigger, time)


# Use the batchYielder to concatenate every batch and store it into a single h5 file

gen = batchYielder(gamma_file_paths, diffuse_df)
batch = next(gen)
pic, run, event, zd_deg, az_deg, trigger, time = batchFormatter(batch)
row_count = trigger.shape[0]

with h5py.File(path_mc_diffuse_images, 'w') as hdf:
    maxshape_pic = (None,) + pic.shape[1:]
    dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape_pic, chunks=pic.shape, dtype=pic.dtype)
    maxshape_run = (None,) + run.shape[1:]
    dset_run = hdf.create_dataset('Run', shape=run.shape, maxshape=maxshape_run, chunks=run.shape, dtype=run.dtype)
    maxshape_event = (None,) + event.shape[1:]
    dset_event = hdf.create_dataset('Event', shape=event.shape, maxshape=maxshape_event, chunks=event.shape, dtype=event.dtype)
    maxshape_zd_deg = (None,) + zd_deg.shape[1:]
    dset_zd_deg = hdf.create_dataset('Zd_deg', shape=zd_deg.shape, maxshape=maxshape_zd_deg, chunks=zd_deg.shape, dtype=zd_deg.dtype)
    maxshape_az_deg = (None,) + az_deg.shape[1:]
    dset_az_deg = hdf.create_dataset('Az_deg', shape=az_deg.shape, maxshape=maxshape_az_deg, chunks=az_deg.shape, dtype=az_deg.dtype)
    maxshape_trigger = (None,) + trigger.shape[1:]
    dset_trigger = hdf.create_dataset('Pointing_Az', shape=trigger.shape, maxshape=maxshape_trigger, chunks=trigger.shape, dtype=trigger.dtype)
    maxshape_time = (None,) + time.shape[1:]
    dset_time = hdf.create_dataset('Pointing_Zd', shape=time.shape, maxshape=maxshape_time, chunks=time.shape, dtype=time.dtype)

    dset_pic[:] = pic
    dset_run[:] = run
    dset_event[:] = event
    dset_zd_deg[:] = zd_deg
    dset_az_deg[:] = az_deg
    dset_trigger[:] = trigger
    dset_time[:] = time

    for batch in gen:
        pic, run, event, zd_deg, az_deg, trigger, time = batchFormatter(batch)

        dset_pic.resize(row_count + trigger.shape[0], axis=0)
        dset_run.resize(row_count + trigger.shape[0], axis=0)
        dset_event.resize(row_count + trigger.shape[0], axis=0)
        dset_zd_deg.resize(row_count + trigger.shape[0], axis=0)
        dset_az_deg.resize(row_count + trigger.shape[0], axis=0)
        dset_trigger.resize(row_count + trigger.shape[0], axis=0)
        dset_time.resize(row_count + trigger.shape[0], axis=0)

        dset_pic[row_count:] = pic
        dset_run[row_count:] = run
        dset_event[row_count:] = event
        dset_zd_deg[row_count:] = zd_deg
        dset_az_deg[row_count:] = az_deg
        dset_trigger[row_count:] = trigger
        dset_time[row_count:] = time

        row_count += trigger.shape[0]


proton_file_paths, diffuse_df = getMetadata(path_raw_mc_proton_folder, path_proton)

path_mc_proton = proton_file_paths

gen = batchYielder(proton_file_paths, diffuse_df)
batch = next(gen)
pic, run, event, zd_deg, az_deg, trigger, time = batchFormatter(batch)
row_count = trigger.shape[0]

with h5py.File(path_mc_proton_images, 'w') as hdf:
    maxshape_pic = (None,) + pic.shape[1:]
    dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape_pic, chunks=pic.shape, dtype=pic.dtype)
    maxshape_run = (None,) + run.shape[1:]
    dset_run = hdf.create_dataset('Run', shape=run.shape, maxshape=maxshape_run, chunks=run.shape, dtype=run.dtype)
    maxshape_event = (None,) + event.shape[1:]
    dset_event = hdf.create_dataset('Event', shape=event.shape, maxshape=maxshape_event, chunks=event.shape, dtype=event.dtype)
    maxshape_zd_deg = (None,) + zd_deg.shape[1:]
    dset_zd_deg = hdf.create_dataset('Zd_deg', shape=zd_deg.shape, maxshape=maxshape_zd_deg, chunks=zd_deg.shape, dtype=zd_deg.dtype)
    maxshape_az_deg = (None,) + az_deg.shape[1:]
    dset_az_deg = hdf.create_dataset('Az_deg', shape=az_deg.shape, maxshape=maxshape_az_deg, chunks=az_deg.shape, dtype=az_deg.dtype)
    maxshape_trigger = (None,) + trigger.shape[1:]
    dset_trigger = hdf.create_dataset('Pointing_Az', shape=trigger.shape, maxshape=maxshape_trigger, chunks=trigger.shape, dtype=trigger.dtype)
    maxshape_time = (None,) + time.shape[1:]
    dset_time = hdf.create_dataset('Pointing_Zd', shape=time.shape, maxshape=maxshape_time, chunks=time.shape, dtype=time.dtype)

    dset_pic[:] = pic
    dset_run[:] = run
    dset_event[:] = event
    dset_zd_deg[:] = zd_deg
    dset_az_deg[:] = az_deg
    dset_trigger[:] = trigger
    dset_time[:] = time

    for batch in gen:
        pic, run, event, zd_deg, az_deg, trigger, time = batchFormatter(batch)

        dset_pic.resize(row_count + trigger.shape[0], axis=0)
        dset_run.resize(row_count + trigger.shape[0], axis=0)
        dset_event.resize(row_count + trigger.shape[0], axis=0)
        dset_zd_deg.resize(row_count + trigger.shape[0], axis=0)
        dset_az_deg.resize(row_count + trigger.shape[0], axis=0)
        dset_trigger.resize(row_count + trigger.shape[0], axis=0)
        dset_time.resize(row_count + trigger.shape[0], axis=0)

        dset_pic[row_count:] = pic
        dset_run[row_count:] = run
        dset_event[row_count:] = event
        dset_zd_deg[row_count:] = zd_deg
        dset_az_deg[row_count:] = az_deg
        dset_trigger[row_count:] = trigger
        dset_time[row_count:] = time

        row_count += trigger.shape[0]
