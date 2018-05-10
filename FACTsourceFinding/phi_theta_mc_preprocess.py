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


#First input: Path to the raw mc_folder
#Second input: Path to the raw_mc_diffuse_folder
#Third input: Path to the 'hexagonal_to_quadratic_mapping_dict.p'
#Fourth input: Path to the 'mc_preprocessed_images.h5'
#path_raw_mc_proton_folder = sys.argv[1]
#path_raw_mc_gamma_folder = sys.argv[2]
#path_store_mapping_dict = sys.argv[3]
#path_mc_diffuse_images = sys.argv[4]

architecture = 'manjaro'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/thesis'

path_raw_mc_proton_folder = base_dir + "/ihp-pc41.ethz.ch/public/phs/sim/proton/"
path_raw_mc_gamma_folder = base_dir + "/ihp-pc41.ethz.ch/public/phs/sim/gamma/"
#path_store_mapping_dict = sys.argv[2]
path_store_mapping_dict = thesis_base + "/jan/07_make_FACT/rebinned_mapping_dict_4_flipped.p"
#path_mc_images = sys.argv[3]
path_mc_diffuse_images = "/run/media/jacob/WDRed8Tb2/Rebinned_5_MC_Diffuse_CORSIKATracking_Images.h5"
path_to_diffuse = "/run/media/jacob/WDRed8Tb1/open_crab_sample_analysis/dl2/gamma_diffuse.hdf5"
#path_mc_diffuse_images = "/run/media/jacob/WDRed8Tb1/Rebinned_5_MC_Phi_Images.h5"

diffuse_df = read_h5py(path_to_diffuse, key="events", columns=["event_num", "@source", "source_position",
                                                               "az_source_calc", "zd_source_calc",
                                                               "az_tracking", "zd_tracking",
                                                               "corsika_evt_header_total_energy", "corsika_evt_header_az"])
#diffuse_df = diffuse_df[(diffuse_df["@source"].str.contains("uwe")) | (diffuse_df['@source'].str.contains("yoda"))]
print(diffuse_df)
run_ids_long = np.array(diffuse_df['@source'].values)
event_nums = np.array(diffuse_df['event_num'].values)

source_x = diffuse_df['source_position_1'].values
source_y = diffuse_df['source_position_0'].values


run_ids = []
for id in run_ids_long:
    tmp = id.split("/")[12]
    #print(tmp)
    tmp = tmp.split("/")[0]
    run_ids.append(int(tmp))

run_ids = np.asarray(run_ids)

diffuse_df['run_id'] = run_ids

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
    run_ids = []
    for id in run_ids_long:
        tmp = id.split("/")[12]
        tmp = tmp.split("/")[0]
        run_ids.append(tmp)
    # Iterate over every file in the subdirs and check if it has the right file extension
    file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(path_folder)
                  for file in fileName if '.json' in file]
    latest_paths = []
    for path in file_paths:
        if any(number in path for number in run_ids):
                #print(path)
                latest_paths.append(path)
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
        mc_truth = file.split(".phs")[0] + ".ch.gz"
        print(mc_truth)
        try:
            sim_reader = ps.SimulationReader(
                photon_stream_path=file,
                mmcs_corsika_path=mc_truth
            )
            data = []
            for event in sim_reader:
                df_event = diffuse_df.loc[(np.isclose(diffuse_df['corsika_evt_header_total_energy'], event.simulation_truth.air_shower.energy)) & (diffuse_df['run_id'] == event.simulation_truth.run)]
                if not df_event.empty:
                    # In the event chosen from the file
                    # Each event is the same as each line below
                    source_pos_x = df_event['source_position_1'].values
                    print(source_pos_x.shape)
                    source_pos_y = df_event['source_position_0'].values
                    energy = event.simulation_truth.air_shower.energy
                    event_photons = event.photon_stream.list_of_lists
                    zd_deg = event.zd
                    print("CORSIKA ZD Track vs FACT")
                    print(zd_deg)
                    print(df_event['zd_tracking'])
                    az_deg = event.az
                    print("CORSIKA Az track vs FACT")
                    print(az_deg)
                    print(df_event['az_tracking'])
                    act_phi = event.simulation_truth.air_shower.phi
                    act_theta = event.simulation_truth.air_shower.theta
                    print("CORSIKA Phi vs FACT Zd")
                    print(act_phi)
                    print(df_event['zd_source_calc'])
                    print("CORSIKA Theta vs FACT Az")
                    print(act_theta)
                    print(df_event['az_source_calc'])
                    sky_source_az = df_event['az_source_calc'].values
                    sky_source_zd = df_event['zd_source_calc'].values
                    #zd_deg = df_event['az_tracking'].values
                    #az_deg = df_event['zd_tracking'].values
                    input_matrix = np.zeros([75,75])
                    chid_to_pixel = id_position[0]
                    pixel_index_to_grid = id_position[1]
                    for index in range(1440):
                        for element in chid_to_pixel[index]:
                            coords = pixel_index_to_grid[element[0]]
                            input_matrix[coords[0]][coords[1]] += element[1]*len(event_photons[index])

                    data.append([np.fliplr(np.rot90(input_matrix, 3)), energy, zd_deg, az_deg, source_pos_x, source_pos_y, act_phi, act_theta])
            #exit(1)
            yield data

        except Exception as e:
            print(str(e))


# Use the batchYielder to concatenate every batch and store it into one h5 file
# Change the datatype to np-arrays
def batchFormatter(batch):
    pic, run, event, zd_deg, source_position_x, source_pos_y, act_phi, act_theta = zip(*batch)
    pic = reformat(np.array(pic))
    run = np.array(run)
    event = np.array(event)
    zd_deg = np.array(zd_deg)
    source_position_x = np.array(source_position_x)
    source_pos_y = np.array(source_pos_y)
    act_phi = np.array(act_phi)
    act_theta = np.array(act_theta)
    return (pic, run, event, zd_deg, source_position_x, source_pos_y, act_phi, act_theta)


# Use the batchYielder to concatenate every batch and store it into a single h5 file

gen = batchYielder(path_mc_gammas)
batch = next(gen)
pic, energy, zd_deg, az_deg, phi, theta, act_phi, act_theta = batchFormatter(batch)
row_count = az_deg.shape[0]
print(row_count)

with h5py.File(path_mc_diffuse_images, 'w') as hdf:
    maxshape_pic = (None,) + pic.shape[1:]
    dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape_pic, chunks=pic.shape, dtype=pic.dtype)
    maxshape_run = (None,) + energy[1:]
    dset_run = hdf.create_dataset('Energy', shape=energy.shape, maxshape=maxshape_run, chunks=energy.shape, dtype=energy.dtype)
    maxshape_event = (None,) + zd_deg[1:]
    dset_zd_deg = hdf.create_dataset('Zd_deg', shape=zd_deg.shape, maxshape=maxshape_event, chunks=zd_deg.shape, dtype=zd_deg.dtype)
    maxshape_az_deg = (None,) + energy[1:]
    dset_az_deg = hdf.create_dataset('Az_deg', shape=az_deg.shape, maxshape=maxshape_az_deg, chunks=az_deg.shape, dtype=az_deg.dtype)
    maxshape_phi = (None,) + energy[1:]
    dset_phi = hdf.create_dataset('Source_X', shape=phi.shape, maxshape=maxshape_phi, chunks=phi.shape, dtype=phi.dtype)
    maxshape_theta = (None,) + energy[1:]
    dset_theta = hdf.create_dataset('Source_Y', shape=theta.shape, maxshape=maxshape_theta, chunks=theta.shape, dtype=theta.dtype)
    maxshape_phia = (None,) + energy[1:]
    dset_phia = hdf.create_dataset('Phi', shape=act_phi.shape, maxshape=maxshape_phia, chunks=act_phi.shape, dtype=act_phi.dtype)
    maxshape_thetaa = (None,) + energy[1:]
    dset_thetaa = hdf.create_dataset('Theta', shape=act_theta.shape, maxshape=maxshape_thetaa, chunks=act_theta.shape, dtype=act_theta.dtype)

    dset_pic[:] = pic
    dset_run[:] = energy
    dset_phi[:] = phi
    dset_zd_deg[:] = zd_deg
    dset_az_deg[:] = az_deg
    dset_theta[:] = theta
    dset_phia[:] = act_phi
    dset_thetaa[:] = act_theta

    for batch in gen:
        pic, energy, zd_deg, az_deg, phi, theta, act_phi, act_theta = batchFormatter(batch)

        dset_pic.resize(row_count + theta.shape[0], axis=0)
        dset_run.resize(row_count + theta.shape[0], axis=0)
        dset_phi.resize(row_count + theta.shape[0], axis=0)
        dset_theta.resize(row_count + theta.shape[0], axis=0)
        dset_zd_deg.resize(row_count + theta.shape[0], axis=0)
        dset_az_deg.resize(row_count + theta.shape[0], axis=0)
        dset_phia.resize(row_count + theta.shape[0], axis=0)
        dset_thetaa.resize(row_count + theta.shape[0], axis=0)

        dset_pic[row_count:] = pic
        dset_run[row_count:] = energy
        dset_phi[row_count:] = phi
        dset_theta[row_count:] = theta
        dset_zd_deg[row_count:] = zd_deg
        dset_az_deg[row_count:] = az_deg
        dset_phia[row_count:] = act_phi
        dset_thetaa[row_count:] = act_theta

        row_count += phi.shape[0]