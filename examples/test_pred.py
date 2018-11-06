import pickle
import os
import numpy as np
if os.path.isfile("crab_predictions.p"):
    with open("crab_predictions.p", "rb") as savedfile:
        predictions = np.asarray(pickle.load(savedfile))
        predictions.reshape((-1,))
        print(predictions.shape)
        print(np.mean(predictions))
        print(np.std(predictions))
        print(predictions)
exit()