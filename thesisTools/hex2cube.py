import numpy as np
import pickle
import pandas as pd
import sys
import matplotlib.pyplot as plt

# Matrix for converting axial coordinates to pixel coordinates
axial_to_pixel_mat = np.array([[np.sqrt(3), np.sqrt(3) / 2], [0, 3 / 2.]])

# Matrix for converting pixel coordinates to axial coordinates
pixel_to_axial_mat = np.linalg.inv(axial_to_pixel_mat)

# These are the vectors for moving from any hex to one of its neighbors.
SE = np.array((1, 0, -1))
SW = np.array((0, 1, -1))
W = np.array((-1, 1, 0))
NW = np.array((-1, 0, 1))
NE = np.array((0, -1, 1))
E = np.array((1, -1, 0))
ALL_DIRECTIONS = np.array([NW, NE, E, SE, SW, W, ])



size = 1
df = pd.read_csv("/run/media/jbieker/SSD/Development/thesis/camera_bild.csv")
print(len(df))

#df.plot.scatter(x='x', y='y', c='data', cmap=plt.cm.Reds, marker='h')
#plt.show()

# These are the vectors for moving from any hex to one of its neighbors.

def cube_to_oddr(cube):
    col = cube[0] + (cube[1] - (cube[1] & 1)) / 2
    row = cube[1]
    return [col, row]

def oddr_to_cube(hex):
    x = hex[0] - (hex[1] - (int(hex[1]) & 1)) / 2
    z = hex[1]
    y = -x-z
    return [x, y, z]

from fact.instrument import get_pixel_coords, get_pixel_dataframe

position_dict = {}
x = []
y = []
z = []

#plt.show()

pm = get_pixel_dataframe()

for id, values in enumerate(pm['x']):
    print(id)
    hex = [pm['x'][id], pm['y'][id]]
    position_dict[id] = oddr_to_cube(hex)
    print(oddr_to_cube(hex))
    x.append(oddr_to_cube(hex)[0])
    y.append(oddr_to_cube(hex)[1])
    z.append(oddr_to_cube(hex)[2])
    print(oddr_to_cube(hex)[0] + oddr_to_cube(hex)[1] + oddr_to_cube(hex)[2])

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c='r', marker='h')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

x_cor, y_cor = get_pixel_coords()

y_cor=np.round(y_cor/8.2175,0)
x_cor=x_cor/4.75+y_cor
mapping = np.arange(0, 1440, 1)
plt.scatter(x_cor, y_cor, c=mapping, cmap=plt.cm.Reds, marker='s')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('FACT Camera skewed to quadratic pixels')

plt.show()


#TODO Need to convert the pixel corrdinates to the hexagon coordinates first, then hexagon to cubes

#Haveto convert the  coordinates toe hexagonal