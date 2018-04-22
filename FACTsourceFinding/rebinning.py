#import keras
import numpy as np
import h5py
import fact
from fact.io import read_h5py
from fact.coordinates.utils import arrays_to_equatorial, equatorial_to_camera, camera_to_equatorial
import pickle
from astropy.coordinates import SkyCoord
import pandas as pd

from fact.instrument import get_pixel_dataframe, get_pixel_coords

from fact_plots.skymap import plot_skymap
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord

from fact.io import read_h5py
import fact.plotting as factplot

from fact_plots.plotting import add_preliminary
from fact_plots.time import read_timestamp

df = get_pixel_dataframe()

x = (df['x'].values - np.min(df['x'].values))
y = (df['y'].values - np.min(df['y'].values))
print(np.max(x))
print(np.min(x))
print(np.max(y))
print(np.min(y))
#factplot.camera(df['CHID'])
#plt.ylim(0,50)
#plt.xlim(-100,100)
#plt.show()
#plt.show()


new_x = []
new_y = []

x, y = get_pixel_coords()
for i in range(1440):
    if i != 0:
        for j in range(i):
            new_x.append(x[i])
            new_y.append(y[i])
    else:
        new_x.append(x[0])
        new_y.append(y[0])


rebin_grid = np.zeros((371, 371))

# This is rebinning it at 1 bin per mm, or 9.5 bins per hexagon, since mm per pixel is 9.5 for the inner radius
# Divind by 9 or even 10 should work, 38,38 or something to that effect

# (40,45) seems to work, except for a recurring hole in the center

rebinned_grid, xedges, yedges = np.histogram2d(new_x, new_y, bins=(40,45))

plt.imshow(np.rot90(rebinned_grid), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.show()

