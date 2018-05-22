import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys

#First input: Path to the 'Camera_Image_FACT.csv'
#Second input: Path to the 'hexagonal_to_quadratic_mapping_dict.p'
#Third input: Path to the 'Unskewed_Camera_Image_FACT_with_hexagonal_pixel.pdf'
#Fourth input: Path to the 'Skewed_Camera_Image_FACT_with_quadratic_pixel.pdf'
path_image = sys.argv[1]
path_store_mapping_dict = sys.argv[2]
path_unskewed_image = sys.argv[3]
path_skewed_image = sys.argv[4]

df = pd.read_csv(path_image)

#Plot the unskewed camera image (hexagonal)
df.plot.scatter(x='x', y='y', c='data', cmap=plt.cm.Reds, marker='h')
plt.title('Camera Image FACT with hexagonal pixel')
plt.xlabel('x Position')
plt.ylabel('y Position')
plt.tight_layout()
plt.savefig(path_unskewed_image)


positions = pickle.load(open(path_store_mapping_dict, 'rb'))

data = []
for id, pos in positions.items():
    data.append([id, pos[0], pos[1]])
df = pd.DataFrame(data, columns=['id', 'x', 'y'])

#Plot the skewed camera image (quadratic)
df.plot.scatter(x='x', y='y', c='id', cmap=plt.cm.Greys, marker='h')
plt.title('Skewed Camera Image FACT with quadratic pixel')
plt.xlabel('x Position')
plt.ylabel('y Position')
plt.tight_layout()
plt.savefig(path_skewed_image)