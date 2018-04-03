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


with h5py.File("/run/media/jacob/WDRed8Tb1/Crab_preprocessed_images.h5", "r") as hdf:
    images = []
    for index in range(0, 30):
        image = hdf["Image"][index]
        images.append(image)
    images = np.asarray(images)
    print(np.max(images))
    new_image = np.sum(images, axis=0)
    print(np.max(new_image))