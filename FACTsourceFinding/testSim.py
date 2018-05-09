import os
import gzip
import photon_stream as ps
from astropy.coordinates import SkyCoord

sim_reader = ps.SimulationReader(
    photon_stream_path='/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/proton/yoda/028908.phs.jsonl.gz',
    mmcs_corsika_path='/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/proton/yoda/028908.ch.gz'
)


for event in sim_reader:
    print(event)
    print(event.simulation_truth.run)
    print(event.simulation_truth.event)
    print(event.photon_stream)
    print(event.simulation_truth)