import numpy as np
import pickle
import pandas as pd
import sys
import matplotlib.pyplot as plt

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

position_dict = {}
x = []
y = []
z = []

for id, values in enumerate(df.values):
    hex = [values[0], values[1]]
    position_dict[id] = oddr_to_cube(hex)
    print(oddr_to_cube(hex))
    x.append(oddr_to_cube(hex)[0])
    y.append(oddr_to_cube(hex)[1])
    z.append(oddr_to_cube(hex)[2])
    print(id)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


