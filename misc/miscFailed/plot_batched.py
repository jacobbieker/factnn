import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.colors import LogNorm

with h5py.File("/run/media/jacob/WDRed8Tb1/FACTSources/1ES 1218+304_prebatched_preprocessed_images.h5", "r") as hdf:
    image = np.array(hdf['Image'][0])
    image = image.reshape((46,45))

print(image.shape)
print(np.min(image))
print(np.max(image))
print(np.mean(image))
plt.imshow(image, norm=LogNorm(vmin=0.01, vmax=1))
plt.show()