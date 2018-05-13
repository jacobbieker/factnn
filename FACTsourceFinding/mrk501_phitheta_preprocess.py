import numpy as np
import pickle
import h5py
import gzip
import json
import sys
import os
import photon_stream as ps
from astropy.coordinates import PhysicsSphericalRepresentation, AltAz, SkyCoord

from fact.io import read_h5py
from fact.coordinates import horizontal_to_camera


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb2'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/thesis'

path_raw_mc_proton_folder = base_dir + "/ihp-pc41.ethz.ch/public/phs/sim/proton/"
path_raw_mc_gamma_folder = base_dir + "/ihp-pc41.ethz.ch/public/phs/obs/Crab/2014/"
#path_store_mapping_dict = sys.argv[2]
path_store_mapping_dict = thesis_base + "/jan/07_make_FACT/rebinned_mapping_dict_4_flipped.p"
#path_mc_images = sys.argv[3]
path_mc_diffuse_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_Mrk501_500000_Images.h5"
path_to_diffuse = "/run/media/jacob/WDRed8Tb1/dl2_theta/Mrk501_precuts.hdf5"
#path_mc_diffuse_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Phi_Images.h5"
path_store_runlist = "Mrk501_precuts.p"

diffuse_df = read_h5py(path_to_diffuse, key="events", columns=["event_num", "run_id", "night",
                                                               "az_source_calc", "zd_source_calc", "source_position",
                                                               "unix_time_utc", "az_tracking", "zd_tracking"])

diffuse_df = diffuse_df[0:500000]
#diffuse_df = diffuse_df[(diffuse_df["@source"].str.contains("uwe")) | (diffuse_df['@source'].str.contains("yoda"))]
print(diffuse_df)
run_ids = np.array(diffuse_df['run_id'].values)
event_nums = np.array(diffuse_df['event_num'].values)


print(diffuse_df)
# Go through and get all the event_num that belong to a given run_id
events_in_run = {}
for index, run_id in enumerate(run_ids):
    indicies = np.where(run_ids == run_id)[0]
    #print(indicies)
    if indicies.size > 0:
        if run_id not in events_in_run.keys():
            events_in_run[run_id] = []
            for sub_index in indicies:
                events_in_run[run_id].append(event_nums[sub_index])

print(events_in_run)


def getMetadata(path_folder):
    '''
    Gathers the file paths of the training data
    '''
    if not os.path.isfile(path_store_runlist):
        list_paths = diffuse_df["night"].values
        list_paths2 = diffuse_df['run_id'].values
        list_events = []
        for index, night in enumerate(list_paths):
            # Night and run_id are in same index, so make them pay
            list_events.append(str(night) + "_" + str(list_paths2[index]))

        # Iterate over every file in the subdirs and check if it has the right file extension
        file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(path_folder)
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
    #print(latest_paths)
    return latest_paths


def reformat(dataset):
    #Reformat to fit into tensorflow
    dataset = np.array(dataset).reshape((-1, 75, 75, 1)).astype(np.float32)
    return dataset


gamma_file_paths = getMetadata(path_raw_mc_gamma_folder)
#proton_file_paths = getMetadata(path_raw_mc_proton_folder)
id_position = pickle.load(open(path_store_mapping_dict, "rb"))


path_mc_gammas = gamma_file_paths

def batchYielder(paths):

    # Load mapping-dict to switch from hexagonal to matrix
    id_position = pickle.load(open(path_store_mapping_dict, "rb"))

    for index, file in enumerate(paths):
        print(file)
        try:

            sim_reader = ps.EventListReader(file)
            data = []
            for event in sim_reader:
                df_event = diffuse_df.loc[(diffuse_df['event_num'] == event.observation_info.event) & (diffuse_df['night'] == event.observation_info.night) & (diffuse_df['run_id'] == event.observation_info.run)]
                if not df_event.empty:
                    # In the event chosen from the file
                    # Each event is the same as each line below
                    source_pos_x = df_event['source_position_1'].values[0]
                    source_pos_y = df_event['source_position_0'].values[0]
                    energy = df_event['unix_time_utc_0'].values[0] * 1e-6  + df_event['unix_time_utc_1'].values[0]
                    event_photons = event.photon_stream.list_of_lists
                    zd_deg = event.zd
                    az_deg = event.az
                    sky_source_az = df_event['az_source_calc'].values[0]
                    sky_source_zd = df_event['zd_source_calc'].values[0]
                    zd_deg1 = df_event['az_tracking'].values[0]
                    az_deg1 = df_event['zd_tracking'].values[0]
                    input_matrix = np.zeros([75,75])
                    chid_to_pixel = id_position[0]
                    pixel_index_to_grid = id_position[1]
                    for index in range(1440):
                        for element in chid_to_pixel[index]:
                            coords = pixel_index_to_grid[element[0]]
                            input_matrix[coords[0]][coords[1]] += element[1]*len(event_photons[index])

                    data.append([np.fliplr(np.rot90(input_matrix, 3)), energy, zd_deg, az_deg, source_pos_x, source_pos_y, sky_source_zd, sky_source_az, zd_deg1, az_deg1])
            #exit(1)
            yield data

        except Exception as e:
            print(str(e))


# Use the batchYielder to concatenate every batch and store it into one h5 file
# Change the datatype to np-arrays
def batchFormatter(batch):
    pic, run, event, zd_deg, source_position_x, source_pos_y, sky_source_zd, sky_source_az, zd_deg1, az_deg1 = zip(*batch)
    pic = reformat(np.array(pic))
    run = np.array(run)
    event = np.array(event)
    zd_deg = np.array(zd_deg)
    source_position_x = np.array(source_position_x)
    source_pos_y = np.array(source_pos_y)
    sky_source_zd = np.array(sky_source_zd)
    sky_source_az = np.array(sky_source_az)
    zd_deg1 = np.array(zd_deg1)
    az_deg1 = np.array(az_deg1)
    return (pic, run, event, zd_deg, source_position_x, source_pos_y, sky_source_zd, sky_source_az, zd_deg1, az_deg1)


# Use the batchYielder to concatenate every batch and store it into a single h5 file

gen = batchYielder(path_mc_gammas)
batch = next(gen)
pic, energy, zd_deg, az_deg, phi, theta, zd, az, zd_deg1, az_deg1 = batchFormatter(batch)
row_count = az_deg.shape[0]
print(row_count)

with h5py.File(path_mc_diffuse_images, 'w') as hdf:
    maxshape_pic = (None,) + pic.shape[1:]
    dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape_pic, chunks=pic.shape, dtype=pic.dtype)
    maxshape_run = (None,) + zd_deg.shape[1:]
    dset_run = hdf.create_dataset('Time', shape=energy.shape, maxshape=maxshape_run, chunks=energy.shape, dtype=energy.dtype)
    maxshape_event = (None,) + zd_deg.shape[1:]
    dset_zd_deg = hdf.create_dataset('Zd_deg', shape=zd_deg.shape, maxshape=maxshape_event, chunks=zd_deg.shape, dtype=zd_deg.dtype)
    maxshape_az_deg = (None,) + zd_deg.shape[1:]
    dset_az_deg = hdf.create_dataset('Az_deg', shape=az_deg.shape, maxshape=maxshape_az_deg, chunks=az_deg.shape, dtype=az_deg.dtype)
    maxshape_phi = (None,) + zd_deg.shape[1:]
    dset_phi = hdf.create_dataset('Source_X', shape=phi.shape, maxshape=maxshape_phi, chunks=phi.shape, dtype=phi.dtype)
    maxshape_theta = (None,) + zd_deg.shape[1:]
    dset_theta = hdf.create_dataset('Source_Y', shape=theta.shape, maxshape=maxshape_theta, chunks=theta.shape, dtype=theta.dtype)
    maxshape_zd = (None,) + zd_deg.shape[1:]
    dset_zd = hdf.create_dataset('Source_Zd', shape=zd_deg.shape, maxshape=maxshape_zd, chunks=zd.shape, dtype=zd.dtype)
    maxshape_az = (None,) + zd_deg.shape[1:]
    dset_az = hdf.create_dataset('Source_Az', shape=az_deg.shape, maxshape=maxshape_az, chunks=az.shape, dtype=az.dtype)
    maxshape_event1 = (None,) + zd_deg.shape[1:]
    dset_zd_deg1 = hdf.create_dataset('Pointing_Zd', shape=zd_deg1.shape, maxshape=maxshape_event1, chunks=zd_deg1.shape, dtype=zd_deg1.dtype)
    maxshape_az_deg1 = (None,) + zd_deg.shape[1:]
    dset_az_deg1 = hdf.create_dataset('Pointing_Az', shape=az_deg1.shape, maxshape=maxshape_az_deg1, chunks=az_deg1.shape, dtype=az_deg1.dtype)

    dset_pic[:] = pic
    dset_run[:] = energy
    dset_phi[:] = phi
    dset_zd_deg[:] = zd_deg
    dset_az_deg[:] = az_deg
    dset_theta[:] = theta
    dset_zd[:] = zd
    dset_az[:] = az
    dset_zd_deg1[:] = zd_deg1
    dset_az_deg1[:] = az_deg1

    for batch in gen:
        pic, energy, zd_deg, az_deg, phi, theta, zd, az, zd_deg1, az_deg1 = batchFormatter(batch)

        dset_pic.resize(row_count + theta.shape[0], axis=0)
        dset_run.resize(row_count + theta.shape[0], axis=0)
        dset_phi.resize(row_count + theta.shape[0], axis=0)
        dset_theta.resize(row_count + theta.shape[0], axis=0)
        dset_zd_deg.resize(row_count + theta.shape[0], axis=0)
        dset_az_deg.resize(row_count + theta.shape[0], axis=0)
        dset_zd.resize(row_count + theta.shape[0], axis=0)
        dset_az.resize(row_count + theta.shape[0], axis=0)
        dset_zd_deg1.resize(row_count + theta.shape[0], axis=0)
        dset_az_deg1.resize(row_count + theta.shape[0], axis=0)

        dset_pic[row_count:] = pic
        dset_run[row_count:] = energy
        dset_phi[row_count:] = phi
        dset_theta[row_count:] = theta
        dset_zd_deg[row_count:] = zd_deg
        dset_az_deg[row_count:] = az_deg
        dset_zd[row_count:] = zd
        dset_az[row_count:] = az
        dset_zd_deg1[row_count:] = zd_deg1
        dset_az_deg1[row_count:] = az_deg1

        row_count += phi.shape[0]