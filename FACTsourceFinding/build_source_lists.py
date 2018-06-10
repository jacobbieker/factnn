from fact.analysis import split_on_off_source_independent
from fact.analysis.binning import bin_runs
from fact.io import read_h5py, to_h5py
import sys
import numpy as np
import gzip
import json
import pickle

import pandas as pd

from fact.factdb import connect_database, RunInfo, read_into_dataframe, get_ontime_by_source_and_runtype, get_ontime_by_source, Source, RunType, AnalysisResultsRunISDC, AnalysisResultsRunLP
from fact.credentials import get_credentials
import os
import datetime

os.environ["FACT_PASSWORD"] = PASSWORD

#connect_database()
#print("Successfully Connected")

# Args should be: base_input_df, outputdir, start, end, step_size, output_file_name_base

# For loop to produce the different cuts
# First arg needs to be the path to the file
path_raw_crab_folder = "/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/obs/"
#path_store_mapping_dict = sys.argv[2]
path_runs_to_use = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/Crab1314_runs_to_use.csv"
path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/hexagonal_to_quadratic_mapping_dict.p"
#path_mc_images = sys.argv[3]
path_crab_images = "/run/media/jacob/WDRed8Tb1/00_crab1314_preprocessed_images.h5"

'''
with open(path_runs_to_use) as file:
    paths = []

    for line in file:
        # Storing the path to every run file
        l = line.split('\t')
        path = path_raw_crab_folder + l[0][:4]+'/' + l[0][4:6]+'/' + l[0][6:8]+'/' + l[0][:4]+l[0][4:6]+l[0][6:8]+'_'+l[1].strip()+'.phs.jsonl.gz'
        paths.append(path)

    for file in paths:
        with gzip.open(file) as f:
            print(file)
            data = []

            for line in f:
                print("\n")
                #line_data = json.loads(line.decode('utf-8'))

                #event_photons = line_data['PhotonArrivals_500ps']
                #night = line_data['Night']
                #run = line_data['Run']
                #event = line_data['Event']
                #zd_deg = line_data['Zd_deg']
                #az_deg = line_data['Az_deg']
                #trigger = line_data['Trigger']
                #eventTime = line_data['UnixTime_s_us'][0] + 1e-6 * line_data['UnixTime_s_us'][1]
                
                info = RunInfo.select().where(RunInfo.frunid == run and RunInfo.fnight == night)
                for element in info:
                    print(element.fsourcekey)
                    if element.frunstart > datetime.datetime.utcfromtimestamp(eventTime) < element.frunstop:
                        print(element.frunstart)
                        print(datetime.datetime.utcfromtimestamp(eventTime))
                        print(eventTime)
                        sources = Source.select().where(Source.fsourcekey == element.fsourcekey)
                        for source in sources:
                            print(source.fsourcename)
                

                #print(info)

                #print("Event: " + str(event))
'''

def read_in_source_csvs(directory):
    used_paths = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if ".csv" in file:
                path = os.path.join(subdir, file)
                with open(path, "r") as source_csv:
                    content = source_csv.readlines()
                for line in content:
                    used_paths.append(line.strip("\n"))
    return used_paths


# go through all the ones on the WDRed8Tb1

disk_one = "/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/obs/"

# Go through all the ones on the WDRed8Tb2

disk_two = "/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/obs/"

# And ones on Seagate

disk_three = "/run/media/jacob/Seagate/ihp-pc41.ethz.ch/public/phs/obs/"
# And ones on HDD

disk_four = "/run/media/jacob/HDD/2014/"

disks = [disk_one, disk_two, disk_three, disk_four]
file_index = 0
disk_index = 0

list_path = "list_order.pkl"

used_paths = read_in_source_csvs(".")

run_info = read_h5py("RunInfo.hdf5", key="info")
run_type = read_h5py("RunType.hdf5", key="info")
source_stuff = read_h5py("Source.hdf5", key="info")
source_type = read_h5py("SourceType.hdf5", key="info")

print(run_info.columns.values)
print("Run Type:")
print(run_type.columns.values)
print("Source Type:")
print(source_type.columns.values)
print("Source Stuff:")
print(source_stuff.columns.values)

for disk in disks:
    for subdir, dirs, files in os.walk(disk):
        for file in files:
            if "phs.jsonl.gz" in file:
                path = os.path.join(subdir, file)
                print(path)
                with gzip.open(path) as f:
                    try:
                        line_data = json.loads(f.readline().decode('utf-8'))
                        night = line_data['Night']
                        run = line_data['Run']
                        event = line_data['Event']
                        trigger = line_data['Trigger']
                        eventTime = line_data['UnixTime_s_us'][0] + 1e-6 * line_data['UnixTime_s_us'][1]
                        element = run_info[(run_info["fRunID"] == run) & (run_info['fNight'] == night)]
                        #info = RunInfo.select().where(RunInfo.frunid == run and RunInfo.fnight == night)
                        if pd.to_datetime(element["fRunStart"].iloc[0]) < datetime.datetime.utcfromtimestamp(eventTime) < pd.to_datetime(element["fRunStop"].iloc[0]):
                            print(str(pd.to_datetime(element["fRunStart"].iloc[0])) + " < " + str(datetime.datetime.utcfromtimestamp(eventTime)) + " < " + str(pd.to_datetime(element["fRunStop"].iloc[0])))
                            # Now in the right run for the event, should be the right source
                            run_type_key = element["fRunTypeKey"].iloc[0]
                            runtype = run_type[(run_type["fRunTypeKEY"] == run_type_key)]
                            #runtypes = RunType.select().where(RunType.fruntypekey == element.fruntypekey)
                            print(runtype["fRunTypeName"].iloc[0])
                            if runtype["fRunTypeName"].iloc[0] == "data":
                                source_name = element["fSourceKEY"].iloc[0]
                                source = source_stuff[(source_stuff["fSourceKEY"] == source_name)]
                                #sources = Source.select().where(Source.fsourcekey == element.fsourcekey)
                                print(source["fSourceName"].iloc[0])
                                if os.path.exists(str(source["fSourceName"].iloc[0]) + ".csv"):
                                    append_write = 'a' # append if already exists
                                else:
                                    append_write = 'w' # make a new file if not
                                # Write the path ot the file that has data on the source one
                                with open(str(source["fSourceName"].iloc[0]) + ".csv", append_write) as source_csv:
                                    source_csv.write(path)
                                    source_csv.write("\n")
                    except Exception as e:
                        print(e)
                        pass
'''
for disk in disks:
    for subdir, dirs, files in os.walk(disk):
        for file in files:
            if "phs.jsonl.gz" in file:
                path = os.path.join(subdir, file)
                print(path)
                with gzip.open(path) as f:
                        try:
                            line_data = json.loads(f.readline().decode('utf-8'))
                            #for line in f:
                            #   line_data = json.loads(line.decode('utf-8'))
                            night = line_data['Night']
                            run = line_data['Run']
                            event = line_data['Event']
                            trigger = line_data['Trigger']
                            eventTime = line_data['UnixTime_s_us'][0] + 1e-6 * line_data['UnixTime_s_us'][1]
                            element = run_info[(run_info["fRunID"] == run) & (run_info['fNight'] == night)]
                            #info = RunInfo.select().where(RunInfo.frunid == run and RunInfo.fnight == night)
                            if pd.to_datetime(element["fRunStart"].iloc[0]) < datetime.datetime.utcfromtimestamp(eventTime) < pd.to_datetime(element["fRunStop"].iloc[0]):
                                print(str(pd.to_datetime(element["fRunStart"].iloc[0])) + " < " + str(datetime.datetime.utcfromtimestamp(eventTime)) + " < " + str(pd.to_datetime(element["fRunStop"].iloc[0])))
                                # Now in the right run for the event, should be the right source
                                run_type_key = element["fRunTypeKey"].iloc[0]
                                runtype = run_type[(run_type["fRunTypeKEY"] == run_type_key)]
                                #runtypes = RunType.select().where(RunType.fruntypekey == element.fruntypekey)
                                print(runtype["fRunTypeName"].iloc[0])
                                if runtype["fRunTypeName"].iloc[0] == "data":
                                    source_name = element["fSourceKEY"].iloc[0]
                                    source = source_stuff[(source_stuff["fSourceKEY"] == source_name)]
                                    #sources = Source.select().where(Source.fsourcekey == element.fsourcekey)
                                    print(source["fSourceName"].iloc[0] == "Mrk 501")
                                    if os.path.exists(str(source["fSourceName"].iloc[0]) + "_14.csv"):
                                        append_write = 'a' # append if already exists
                                    else:
                                        append_write = 'w' # make a new file if not
                                    # Write the path ot the file that has data on the source one
                                    with open(str(source["fSourceName"].iloc[0]) + "_14.csv", append_write) as source_csv:
                                        source_csv.write(path)
                                        source_csv.write("\n")
                        except Exception as e:
                            print(e)
                            pass
'''