import numpy as np
from keras.utils import Sequence
from sklearn.utils import shuffle
from factnn.utils.augment import *


class EventFileGenerator(Sequence):

    def __init__(self, paths, num_elements, batch_size, preprocessor=None, proton_preprocessor=None, proton_paths=None,
                 as_channels=False,
                 final_slices=5, slices=(30, 70), augment=False, training_type=None):
        self.paths = paths
        self.num_elements = num_elements
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.proton_preprocessor = proton_preprocessor
        self.as_channels = as_channels
        self.final_slices = final_slices
        self.slices = slices
        self.augment = augment
        self.training_type = training_type
        self.proton_paths = proton_paths

    def __getitem__(self, index):
        """
        Go through each set of files and augment them as needed
        :param index:
        :return:
        """
        batch_files = self.paths[index * self.batch_size:(index + 1) * self.batch_size]
        if self.proton_paths is not None:
            proton_batch_files = self.proton_paths[index * self.batch_size:(index + 1) * self.batch_size]
            proton_images, proton_data_format = self.proton_preprocessor.on_files_processor(paths=proton_batch_files,
                                                                                            size=self.batch_size,
                                                                                            augment=self.augment)
        else:
            proton_images = None
        images, data_format = self.preprocessor.on_files_processor(paths=batch_files, size=self.batch_size,
                                                                   augment=self.augment)
        images, labels = augment_image_batch(images, proton_images=proton_images,
                                             type_training=self.training_type,
                                             augment=self.augment,
                                             swap=self.augment,
                                             shape=[-1,self.slices[1]-self.slices[0], self.preprocessor.shape[1], self.preprocessor.shape[2],1],
                                             as_channels=self.as_channels)

        return images, labels

    def __len__(self):
        """
        Returns the length of the list of paths, as the number of events is not known, gives min if using both proton and
        not proton
        :return:
        """
        if self.proton_paths is not None:
            return np.min(int(np.ceil(len(self.paths) / float(self.batch_size))),
                          int(np.ceil(len(self.proton_paths) / float(self.batch_size))))
        else:
            return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.augment:
            self.paths = shuffle(self.paths)
            self.proton_paths = shuffle(self.proton_paths)
