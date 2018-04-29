import os
import gzip
import photon_stream as ps
from astropy.coordinates import SkyCoord

sim_reader = ps.SimulationReader(
    photon_stream_path='/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/proton/yoda/028908.phs.jsonl.gz',
    mmcs_corsika_path='/run/media/jacob/WDRed8Tb1/ihp-pc41.ethz.ch/public/phs/sim/proton/yoda/028908.ch.gz'
)


for event in sim_reader:
    print(event.az)
    print(event.simulation_truth.air_shower.theta)
    SkyCoord(representation_type="spherical")