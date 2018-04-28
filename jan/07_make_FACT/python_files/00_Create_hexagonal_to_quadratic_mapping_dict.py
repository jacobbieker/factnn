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
path_image = "/home/jacob/Development/thesis/jan/07_make_FACT/Camera_Image_FACT.csv"
path_store_mapping_dict = "/run/media/jacob/SSD/Development/thesis/jan/07_make_FACT/quadratic_to_quadratic_mapping_dict.p"

df = pd.read_csv(path_image)
df.y=round(df.y/8.2175,0)+22
df.x=(df.x/4.75+39+df.y-16)/2

position_dict = {}
for id, values in enumerate(df.values):
    position_dict[id] = [values[0], values[1]]

test = np.zeros((46,45))
l = 0
for i in range(1440):
    x1, y1 = position_dict[i]
    test[int(x1)][int(y1)] = i

plt.imshow(test, cmap="Greys")
plt.savefig("SkewedFACT.png", dpi=300)
pickle.dump(position_dict, open(path_store_mapping_dict, 'wb'))