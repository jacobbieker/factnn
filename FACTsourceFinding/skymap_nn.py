import fact_plots
import os
import fact
import yaml
from fact.io import read_h5py


df = read_h5py("/run/media/jacob/WDRed8Tb1/dl2_theta/precuts/std_analysis/crab_1314_std_analysis_v1.0.0.hdf5", key='events')

print("Read in file")


fact_plots.plot_angular_resolution(df=df)
