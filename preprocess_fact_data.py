import os
import gzip
import json
import pickle
import numpy as np
from multiprocessing import Pool
import h5py
import time

import photon_stream as ps
import matplotlib.pyplot as plt
import yaml

envs = yaml.load("envs.yaml")

#sim_path = envs['local']['sim']
#obs_path = envs['local']['obs']

reader = ps.EventListReader("/run/media/jbieker/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/obs/2013/01/02/20130102_035.phs.jsonl.gz")

    #os.path.join("/run/media/jbieker/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/obs/", "2014", "01", "02", "20130102_035.phs.jsonl.gz"))

event = next(reader)
print(event)
event.plot()


sim_reader = ps.SimulationReader(
    photon_stream_path="/run/media/jbieker/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/sim/gamma/gustav/010910.phs.jsonl.gz",
    mmcs_corsika_path="/run/media/jbieker/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/sim/gamma/gustav/010910.ch.gz"
)

for event in sim_reader:
    print(event)
    pass
#plt.show()
