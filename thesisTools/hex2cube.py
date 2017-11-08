import numpy as np
import pickle
import pandas as pd
import sys
import matplotlib.pyplot as plt

size = 1
df = pd.read_csv("/run/media/jbieker/SSD/Development/thesis/camera_bild.csv")
print(len(df))

df.plot.scatter(x='x', y='y', c='data', cmap=plt.cm.Reds, marker='h')
#plt.show()

# These are the vectors for moving from any hex to one of its neighbors.
SE = np.array((1, 0, -1))
SW = np.array((0, 1, -1))
W = np.array((-1, 1, 0))
NW = np.array((-1, 0, 1))
NE = np.array((0, -1, 1))
E = np.array((1, -1, 0))
ALL_DIRECTIONS = np.array([NW, NE, E, SE, SW, W, ])



def oddr_to_cube(hex):
    '''
    Takes Hex point and returns the cube representation value for that point
    :param hex: [row, col]
    :return:
    '''
    x = hex[1] - (hex[0] - (hex[0] & 1)) / 2
    z = hex.row
    y = -x-z
    return (x, y, z)

def hex_to_pixel(hex):
    '''
    Takes Hex point and converts to pixel
    :param hex: [row, col]
    :return:
    '''
    x = size * np.sqrt(3) * (hex.q + hex.r/2)
    y = size * 3/2 * hex.r
    return (x,y)

