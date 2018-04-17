from fact.io import read_h5py, to_h5py
from fact.instrument import get_pixel_coords, get_pixel_dataframe
import sys
import numpy as np
import gzip
import json
import h5py
import pickle
import yaml

from fact.factdb import connect_database, RunInfo, get_ontime_by_source_and_runtype, get_ontime_by_source, Source, RunType, AnalysisResultsRunISDC, AnalysisResultsRunLP
from fact.credentials import get_credentials
import os
import datetime

import matplotlib.pyplot as plt
import fact.plotting as factplot
from scipy import spatial

np.random.seed(0)
#path_store_mapping_dict = sys.argv[2]
path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/hexagonal_to_quadratic_mapping_dict.p"
#path_mc_images = "D:\Development\thesis\jan\07_make_FACT\hexagonal_to_quadratic_mapping_dict.p"

source_file_paths = []
output_paths = []

architecture = yaml.load("../envs.yaml")['arch']

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/thesis'

source_file_paths.append("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/std_analysis/crab_1314_std_analysis_v1.0.0.hdf5")
output_paths.append("/run/media/jacob/WDRed8Tb1/FACTSources/crab_1314_std_analysis_v1.0.0_preprocessed_source.hdf5")
# Build list of source hdf5 files to go through to get runlist and source prediction points
for subdir, dirs, files in os.walk("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/std_analysis/"):
    for file in files:
        path = os.path.join(subdir, file)
        source_file_paths.append(path)
        output_filename = file.split(".hdf5")[0]
        output_paths.append(base_dir + "/FACTSources/" + output_filename + "_preprocessed_source.hdf5")

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

pixel_mapping_df = get_pixel_dataframe()



# Format dataset to fit into tensorflow
def reformat(dataset):
    return dataset.reshape((-1, 46, 45, 1)).astype(np.float32)



# Change the datatype to np-arrays
def batchFormatter(batch):
    pic, night, run, event, zd_deg, az_deg, trigger, mapping = zip(*batch)
    pic = reformat(np.array(pic))
    night = np.array(night)
    run = np.array(run)
    event = np.array(event)
    zd_deg = np.array(zd_deg)
    az_deg = np.array(az_deg)
    trigger = np.array(trigger)
    mapping = reformat(np.array(mapping))
    return (pic, night, run, event, zd_deg, az_deg, trigger, mapping)


# Use the batchYielder to concatenate every batch and store it into a single h5 f`ile
q = 0
for index, source_file in enumerate(source_file_paths):
    print(output_paths[index])
    if os.path.isfile(output_paths[index]):
        os.remove(output_paths[index])
    list_of_used_sources = []
    all_data_from_source = []
    print(source_file)
    run_df = read_h5py(file_path=source_file, key='runs', columns=['night', 'run_id', 'source'])
    if "0.17.2" in source_file:
        info_df = read_h5py(file_path=source_file, key='events', columns=['source_position'])
    else:
        info_df = read_h5py(file_path=source_file, key='events', columns=['source_position_x', 'source_position_y'])
    nights = list(run_df['night'].values)
    #print(nights)
    run_ids = list(run_df['run_id'].values)
    #print(run_ids)
    # Now have all the information needed
    current_source = run_df['source'][0]
    print(current_source)
    with open(os.path.join(thesis_base + "/FACTsourceFinding/runs", current_source + ".csv")) as source_list:
        data = []
        k = 0
        for path in source_list:
            try:
                with gzip.open(path.split("\n")[0]) as f:
                    print(path)
                    data = []
                    line_data = json.loads(f.readline().decode('utf-8'))
                    night = line_data['Night']
                    run = line_data['Run']

                    if night not in nights and run not in run_ids:
                        print("Skipping")
                        continue
                    else:
                        for line in f:
                            line_data = json.loads(line.decode('utf-8'))
                            night = line_data['Night']
                            run = line_data['Run']
                            event = line_data['Event']
                            trigger = line_data['Trigger']
                            event_photons = line_data['PhotonArrivals_500ps']
                            zd_deg = line_data['Zd_deg']
                            az_deg = line_data['Az_deg']
                            eventTime = line_data['UnixTime_s_us']

                            if night in nights and run in run_ids:
                                # Need it for the observations
                                # Transform it with the pixel mapping, and a second part for the labels
                                list_of_used_sources.append(path)
                                # Now, use the source_x_prediction and source_y_prediction to locate the source in the pixel mapping
                                # Get index of specific source
                                night_index = nights.index(night)
                                source_info_df = info_df.iloc[night_index]
                                pixel_mapping_df['source_position'] = 0 # 0 is no source, 1 is it is the source
                                # Sets the predicted source position as the source
                                if "0.17.2" in source_file:
                                    float_y = float(source_info_df['source_position_0']) # source_position_0 seems to be y, and the other x
                                    float_x = float(source_info_df['source_position_1'])
                                else:
                                    float_y = float(source_info_df['source_position_y']) # source_position_0 seems to be y, and the other x
                                    float_x = float(source_info_df['source_position_x'])
                                x_y = get_pixel_coords()
                                new_list = np.stack((x_y[0], x_y[1]), axis=1)
                                output = new_list[spatial.KDTree(new_list).query([float_x, float_y], k=7)[1]] # Return the 7 nearest neighbors, so not just a single pixel per
                                for element in output:
                                    pixel_mapping_df.loc[(pixel_mapping_df['x'] == element[0]) & (pixel_mapping_df['y'] == element[1]), "source_position"] = 1
                                id_position = pickle.load(open(path_store_mapping_dict, "rb"))

                                mapping_matrix = np.zeros([46, 45])
                                input_matrix = np.zeros([46, 45])
                                for i in range(1440):
                                    x, y = id_position[i]
                                    mapping_matrix[int(x)][int(y)] = pixel_mapping_df['source_position'][i]
                                    input_matrix[int(x)][int(y)] = len(event_photons[i])
                                k += 1
                                #print("Appended Data " + str(k))
                                data.append([input_matrix, night, run, event, zd_deg, az_deg, trigger, mapping_matrix]) # end of each event, add it to data
            except:
                print("Failed at something" + str(k))
                #pass=
            if data != []:
                print("Append Data to All Data \n")
                all_data_from_source.append(data) # End of each path, add it to all data
                q += 1
                if (q % 5) == 0:
                    print("Writing Stuff\n")
                    pic, night, run, event, zd_deg, az_deg, trigger, mapping = batchFormatter(all_data_from_source[-1])
                    row_count = trigger.shape[0]
                    with h5py.File(output_paths[index] + "_" + str(q), 'w') as hdf:
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
                        maxshape_mapping = (None,) + mapping.shape[1:]
                        dset_map = hdf.create_dataset('Source_Position', shape=mapping.shape, maxshape=maxshape_mapping, chunks=mapping.shape, dtype=mapping.dtype)

                        dset_pic[:] = pic
                        dset_night[:] = night
                        dset_run[:] = run
                        dset_event[:] = event
                        dset_zd_deg[:] = zd_deg
                        dset_az_deg[:] = az_deg
                        dset_trigger[:] = trigger
                        dset_map[:] = mapping

                        for batch in all_data_from_source:
                            pic, night, run, event, zd_deg, az_deg, trigger, mapping = batchFormatter(batch)

                            dset_pic.resize(row_count + trigger.shape[0], axis=0)
                            dset_night.resize(row_count + trigger.shape[0], axis=0)
                            dset_run.resize(row_count + trigger.shape[0], axis=0)
                            dset_event.resize(row_count + trigger.shape[0], axis=0)
                            dset_zd_deg.resize(row_count + trigger.shape[0], axis=0)
                            dset_az_deg.resize(row_count + trigger.shape[0], axis=0)
                            dset_trigger.resize(row_count + trigger.shape[0], axis=0)
                            dset_map.resize(row_count + trigger.shape[0], axis=0)

                            dset_pic[row_count:] = pic
                            dset_night[row_count:] = night
                            dset_run[row_count:] = run
                            dset_event[row_count:] = event
                            dset_zd_deg[row_count:] = zd_deg
                            dset_az_deg[row_count:] = az_deg
                            dset_trigger[row_count:] = trigger
                            dset_map[row_count:] = mapping

                            row_count += trigger.shape[0]

        # End of all paths in source_list
        for element in all_data_from_source:
            q += 1
            # For each element is the data from a single run

            pic, night, run, event, zd_deg, az_deg, trigger, mapping = batchFormatter(element)
            row_count = trigger.shape[0]

            with h5py.File(output_paths[index] + "_" + str(q), 'w') as hdf:
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
                maxshape_mapping = (None,) + mapping.shape[1:]
                dset_map = hdf.create_dataset('Source_Position', shape=mapping.shape, maxshape=maxshape_mapping, chunks=mapping.shape, dtype=mapping.dtype)

                dset_pic[:] = pic
                dset_night[:] = night
                dset_run[:] = run
                dset_event[:] = event
                dset_zd_deg[:] = zd_deg
                dset_az_deg[:] = az_deg
                dset_trigger[:] = trigger
                dset_map[:] = mapping

                for batch in all_data_from_source:
                    pic, night, run, event, zd_deg, az_deg, trigger, mapping = batchFormatter(batch)

                    dset_pic.resize(row_count + trigger.shape[0], axis=0)
                    dset_night.resize(row_count + trigger.shape[0], axis=0)
                    dset_run.resize(row_count + trigger.shape[0], axis=0)
                    dset_event.resize(row_count + trigger.shape[0], axis=0)
                    dset_zd_deg.resize(row_count + trigger.shape[0], axis=0)
                    dset_az_deg.resize(row_count + trigger.shape[0], axis=0)
                    dset_trigger.resize(row_count + trigger.shape[0], axis=0)
                    dset_map.resize(row_count + trigger.shape[0], axis=0)

                    dset_pic[row_count:] = pic
                    dset_night[row_count:] = night
                    dset_run[row_count:] = run
                    dset_event[row_count:] = event
                    dset_zd_deg[row_count:] = zd_deg
                    dset_az_deg[row_count:] = az_deg
                    dset_trigger[row_count:] = trigger
                    dset_map[row_count:] = mapping

                    row_count += trigger.shape[0]