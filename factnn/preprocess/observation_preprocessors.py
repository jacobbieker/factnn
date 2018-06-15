import numpy as np
import pickle
import h5py
import os
import photon_stream as ps
from fact.io import read_h5py
from fact.coordinates import horizontal_to_camera

from factnn.preprocess.base_preprocessor import BasePreprocessor


class ObservationPreprocessor(BasePreprocessor):

    def create_dataset(self):
        return None
