#import keras
import numpy as np
import h5py
import fact
from fact.io import read_h5py
from fact.coordinates.utils import arrays_to_equatorial, equatorial_to_camera, camera_to_equatorial
import pickle
from astropy.coordinates import SkyCoord
import pandas as pd

from fact.instrument import get_pixel_dataframe

from fact_plots.skymap import plot_skymap
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import yaml



# Open the std_analysis thing
# Extract the info from the Night, Run_ID, and Event to open the phs files directly
# read in JSON data until the event num shows up
# Run DBSCAN on it
# Save to standard image format
# Save to Skymap image format
# Train on each event, where the truth is the ra_predict and dec_predict, but also the Az and Zd, since it depends on place of observer
# Train on Mrk501 and Crab, validate on Mrk421 and IE whatever
# Essentially get disp method through that
# Then, take that trained and validate on Az and Zd of different sources, see what combo works best
# Or, also train on ones not in dataset, see if still get similar results

columns = [
    'ra_prediction',
    'dec_predition',
    'az_source_calc',
    'zd_source_calc',
    'run_id',
    'event_num',
    'disp',
    'disp_prediction',
    'gamma_prediction',
    'night',

]

source_one_df = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/crab_precuts.hdf5", key="events")
source_two_df = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/Mrk501_precuts.hdf5", key="events")

# No DBSCAN for now, just use itself

# Create list of files to use
path_raw_crab_folder = "/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/obs/"
