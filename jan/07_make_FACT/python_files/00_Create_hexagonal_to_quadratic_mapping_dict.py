import pandas as pd
import pickle
import sys

#First input: Path to the 'Camera_Image_FACT.csv'
#Second input: Path to the 'hexagonal_to_quadratic_mapping_dict.p'
path_image = sys.argv[1]
path_store_mapping_dict = sys.argv[2]

df = pd.read_csv(path_image)

# Translating and scaling the values to represent natural numbers
# Rounding is needed, because some hexagonal pixel layers are not equally spaced
df.y=round(df.y/8.2175,0)+22
df.x=(df.x/4.75+39+df.y-16)/2

position_dict = {}
for id, values in enumerate(df.values):
    position_dict[id] = [values[0], values[1]]

pickle.dump(position_dict, open(path_store_mapping_dict, 'wb'))