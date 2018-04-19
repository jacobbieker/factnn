import fact
import numpy as np
import matplotlib.pyplot as plt
import os
import photon_stream as ps
from photon_stream import plot as ps_plot
from photon_stream import  photon_cluster as ps_cluster
import pandas as pd

sim_reader = ps.SimulationReader(
    photon_stream_path="/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/sim/gamma/gustav/012909.phs.jsonl.gz",
    mmcs_corsika_path="/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/sim/gamma/gustav/012909.ch.gz"
)

for event in sim_reader:
    print(event)
    new_cluster = ps_cluster.PhotonStreamCluster(event.photon_stream)
    print(new_cluster)
    pass

thrown_events = pd.DataFrame(sim_reader.thrown_events())

print(thrown_events)