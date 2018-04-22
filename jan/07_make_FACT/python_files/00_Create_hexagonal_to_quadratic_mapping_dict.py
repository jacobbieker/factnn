import pandas as pd
import pickle
import sys
from copy import deepcopy
from fact.instrument import get_pixel_coords, get_pixel_dataframe
import matplotlib.pyplot as plt
import fact.plotting as factplot
import numpy as np
#First input: Path to the 'Camera_Image_FACT.csv'
#Second input: Path to the 'hexagonal_to_quadratic_mapping_dict.p'
path_image = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/Camera_Image_FACT.csv"
path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/quadratic_to_quadratic_mapping_dict.p"

df = get_pixel_dataframe()

# Get

# Translating and scaling the values to represent natural numbers
# Rounding is needed, because some hexagonal pixel layers are not equally spaced
factplot.camera(df['CHID'])
plt.show()
df['y'] =round(df['y']/8.2175,0)+22
df['x'] =(df['x']/4.75+39+df['y']-16)/2


x = df['x'].values
y = df['y'].values
print(np.max(x))
print(np.min(x))
print(np.max(y))
print(np.min(y))
factplot.camera(df['CHID'])
plt.ylim(0,50)
plt.xlim(-100,100)
plt.show()

position_dict = {}
for id, values in enumerate(df['x'].values):
    position_dict[id] = [df['x'][id], df['y'][id]]

pickle.dump(position_dict, open(path_store_mapping_dict, 'wb'))