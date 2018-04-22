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

from scipy.stats import binned_statistic_2d

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

#plt.imshow(np.rot90(rebinned_grid), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#plt.show()


print(np.rot90(rebinned_grid).shape)

# Need to figure out how to get mapping from CHID to this,

# Try doing it by number of things in each bin, the number should be, except for 0 and 1, unique
# For 1, just get the pixel coordinates, see where it is in the rebinned grid,
# For 0, same
# Just do it by the pixel coords in unrot90 grid, then rot90 grid and work out

# save one for both though
# Incase it is needed with phs the unrotated grid

x_digitized = np.digitize(x, xedges, right=True)
y_digitized = np.digitize(y, yedges, right=True)

# now combine them to create unique ones, plot that to make sure

position_dict = {}
for index, element in enumerate(x_digitized):
    # Should swap 90 degrees, I think
    position_dict[index] = [y_digitized[index], x_digitized[index]]


# Do it with the larger amount, to make sure swapped in right direction
x_lerge_digitized = np.digitize(new_x, xedges, right=True)
y_lerge_digitized = np.digitize(new_y, yedges, right=True)

# now combine them to create unique ones, plot that to make sure

position_dict_lerge = {}
for index, element in enumerate(x_lerge_digitized):
    # Should swap 90 degrees, I think
    position_dict_lerge[index] = [x_lerge_digitized[index], y_lerge_digitized[index]]

# Now create a graph from the digitized ones
'''
test = np.zeros((46,41))

for i in range(1440):
    element = position_dict[i]
    print(element)
    test[element[0]][element[1]] = 1

plt.imshow(test)
plt.title("Normal Ones")
plt.show()

test = np.zeros((41,46))

for i in range(1440):
    element = position_dict_lerge[i]
    print(element)
    test[element[0]][element[1]] = 1

plt.imshow(test)
plt.title("Lerge Ones")
plt.show()
'''

counts, xedges1, yedges1, binNum = binned_statistic_2d(x, y, range(0,1440), 'count', bins=(40,45), expand_binnumbers=True)

print(counts)
#plt.imshow(np.rot90(counts), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#plt.show()

print(binNum)
print(binNum.shape)
print(counts.shape)
print(xedges.shape)
print(yedges.shape)

test = np.zeros((45, 40))

position_dict = {}
for index in range(1439, -1, -1):
    # Should swap 90 degrees, I think
    print(index)
    x_pos = binNum[0][index]
    y_pos = binNum[1][index]
    # Now get the center of the bin in x and y for the actual thing

    position_dict[index] = [y_pos-1, x_pos-1]


for i in range(1440):
    element = position_dict[i]
    print(element)
    test[element[0]][element[1]] = i

#test = np.flip(test, axis=0)
#plt.imshow(np.flip(test, 0))
#plt.title("Normal Ones")
#plt.show()

path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/holy_squished_mapping_dict.p"
pickle.dump(position_dict, open(path_store_mapping_dict, 'wb'))

import shapely
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.affinity import translate
from shapely.strtree import STRtree
from descartes import PolygonPatch



p = Point(0.0, 0.0)
PIXEL_EDGE = 9.51/np.sqrt(3)
# Top one
p1 = Point(0.0, PIXEL_EDGE)
# Bottom one
p2 = Point(0.0, -PIXEL_EDGE)
# Bottom right
p3 = Point(-PIXEL_EDGE*(np.sqrt(3)/2), -PIXEL_EDGE*.5)
# Bottom left
p4 = Point(PIXEL_EDGE*(np.sqrt(3)/2), PIXEL_EDGE*.5)
# right
p5 = Point(PIXEL_EDGE*(np.sqrt(3)/2), -PIXEL_EDGE*.5)
#  left
p6 = Point(-PIXEL_EDGE*(np.sqrt(3)/2), PIXEL_EDGE*.5)

hexagon = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull


square_start = 186
square_size = 5
square = Polygon([(-square_start,square_start), (-square_start+square_size,square_start),
                  (-square_start+square_size,square_start-square_size), (-square_start, square_start-square_size),
                  (-square_start,square_start)])

grid = False
list_of_squares = [square]
m = 0

steps = int(np.ceil(np.abs(square_start*2) / square_size))
print(steps)
pixel_index_to_grid = {}
pix_index = 0
# Generate tessellation of grid
for x_step in range(steps):
    for y_step in range(steps):
        new_square = translate(square, xoff=x_step*square_size, yoff=-square_size*y_step)
        pixel_index_to_grid[pix_index] = [x_step, y_step]
        pix_index += 1
        list_of_squares.append(new_square)


x, y = get_pixel_coords()
list_hexagons = []
for index, x_coor in enumerate(x):
    list_hexagons.append(translate(hexagon, x_coor, y[index]))
'''
fig = plt.figure(1, dpi=90)

# 1
ax = fig.add_subplot(111)
for patch in list_hexagons:
    ax.add_patch(PolygonPatch(patch))

for patch in list_of_squares:
    ax.add_patch(PolygonPatch(patch))

#ax.add_patch(PolygonPatch(square))
ax.set_ylim(top=186, bottom=-186)
ax.set_xlim(left=-186, right=186)
plt.show()
'''
#factplot.camera(df['CHID'])
#plt.show()

# Now take squares and build grid over the whole thing

# Have grid, now convert to get intersection between each square and every hexagon, saving the intersection.area / hex.area
# to data structure
# DS has to have grid where each square is a pixel, every pixel has a CHID num plus fraction in tuple, kinda hard to do but oh well

list_pixels_and_fractions = {}
for i in range(len(list_of_squares)):
    list_pixels_and_fractions[i] = []

chid_to_pixel = {}
for i in range(1440):
    chid_to_pixel[i] = []

for pixel_index, pixel in enumerate(list_of_squares):
    for chid, hexagon in enumerate(list_hexagons):
        # Do the dirty work, hexagons should be in CHID order because translate in that order and append
        if pixel.intersects(hexagon):
            intersection = pixel.intersection(hexagon)
            fraction_whole = intersection.area/hexagon.area
            #print(fraction_whole)
            #print(chid)
            if not np.isclose(fraction_whole,0.0):
                # so not close to zero overlap, add to list for that pixel
                list_pixels_and_fractions[pixel_index].append((chid, fraction_whole))
                chid_to_pixel[chid].append((pixel_index, fraction_whole))

#print(list_pixels_and_fractions)

hex_to_grid = [chid_to_pixel, pixel_index_to_grid]

path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/rebinned_mapping_dict_5.p"
pickle.dump(hex_to_grid, open(path_store_mapping_dict, 'wb'))

# To get back to original orientation, need to do a fliplr, after a rot90, 3



# Test with CHID mapping
test_rebin = np.zeros((steps,steps))

for index in range(1440):
    print(index)
    for element in chid_to_pixel[index]:
        coords = pixel_index_to_grid[element[0]]
        test_rebin[coords[0]][coords[1]] += element[1]*index

plt.imshow(np.fliplr(np.rot90(test_rebin, 3)))
plt.title("Testing Rebin at " + str(square_size))
plt.savefig("Rebin_size_5.png", dpi=300)
# HAVE IT!!!