import os
import gzip
import photon_stream as ps
from astropy.coordinates import SkyCoord
import pickle
import fact.plotting as factplot
import matplotlib.pyplot as plt
import numpy as np

sim_reader = ps.SimulationReader(
    photon_stream_path='/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/proton/yoda/028908.phs.jsonl.gz',
    mmcs_corsika_path='/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/proton/yoda/028908.ch.gz'
)

with open("/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/Mrk501_precuts.p", 'rb') as foi:
    test = pickle.load(foi)
print(test)
for event in sim_reader:
    print(event)
    print(event.simulation_truth.run)
    print(event.simulation_truth.event)
    event_photons = event.photon_stream.list_of_lists
    print(event.simulation_truth)
    for index in range(1440):
        event_photons[index] = len(event_photons[index])
    event_photons = (event_photons - np.mean(event_photons)) / np.std(event_photons)
    factplot.camera(event_photons)
    plt.show()
    plt.clf()
    break

sim_reader = ps.SimulationReader(
    mmcs_corsika_path='/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/gamma/gustav/010910.ch.gz',
    photon_stream_path='/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/gamma/gustav/010910.phs.jsonl.gz'
)

for event in sim_reader:
    print(event)
    print(event.simulation_truth.run)
    event_photons = event.photon_stream.list_of_lists
    print(event.simulation_truth)
    for index in range(1440):
        event_photons[index] = len(event_photons[index])
    event_photons = (event_photons - np.mean(event_photons)) / np.std(event_photons)
    factplot.camera(event_photons)
    plt.show()
    plt.clf()
    break
