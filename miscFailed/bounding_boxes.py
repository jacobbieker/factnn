from fact.io import read_h5py, to_h5py
from fact.instrument import get_pixel_coords, get_pixel_dataframe
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
import fact.plotting as factplot
from scipy import spatial
import yaml

with open("../envs.yaml", 'r') as yaml_file:
    architecture = yaml.load(yaml_file)['arch']

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
    std_base = '/dl2_theta/orecuts/std_analysis/'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/thesis'
    std_base = base_dir + "/FACTSources/std_analysis/"

np.random.seed(0)

'''
The goal is to get bounding boxes around a point that gives the largest amount of photons inside of it vs outside of it, the Li and Ma significance

'''

