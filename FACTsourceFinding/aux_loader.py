import fact
import photon_stream as ps
import numpy as np
import h5py
import os
import yaml
from fact.io import read_h5py
from astropy.coordinates import AltAz, SkyCoord



'''
The goal of this one is to load in auxiliary data from the std_analysis files for use in different NN
Need to make it flexible

Right now: Get the disp predictions, pointing, source positions in both camera and az, dec, cog, length, and width of events
As well as theta, and theta on/off regions


'''

try:
    with open("/run/media/jacob/SSD/Development/thesis/envs.yaml", 'r') as yaml_file:
        architecture = yaml.load(yaml_file)['arch']
except:
    architecture = 'intel'

if architecture == 'manjaro':
    base_dir = '/run/media/jacob/WDRed8Tb1'
    thesis_base = '/run/media/jacob/SSD/Development/thesis'
else:
    base_dir = '/projects/sventeklab/jbieker'
    thesis_base = base_dir + '/thesis'



def get_column(source, col_name):
    '''
    Return the column for the source file
    :param source:
    :param col_name:
    :return:
    '''

    df = read_h5py(source, key='events')
    return df[col_name].values

def get_source(source_file, event):
    '''
    Return the source in both dec, az, and camera coordinates
    :param source_file:
    :param event:
    :return:
    '''


