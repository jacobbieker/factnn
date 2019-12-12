import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence

from factnn.utils.augment import augment_pointcloud_batch


class PointCloudGenerator(Sequence):
    """

    """

    def __init__(self, paths, batch_size, preprocessor=None, proton_preprocessor=None, proton_paths=None,
                 slices=(30, 70), final_points=2048, replacement=False, augment=False, training_type=None,
                 truncate=True, return_features=False, rotate=True, jitter=None):
        self.paths = paths
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.proton_preprocessor = proton_preprocessor
        self.slices = slices
        self.augment = augment
        self.training_type = training_type
        self.proton_paths = proton_paths
        self.truncate = truncate
        self.replacement = replacement
        self.final_points = final_points
        self.rotate = rotate
        self.jitter = jitter

        # These three are for if multiple inputs need to be returned,
        self.features = return_features
        if return_features:
            self.multiple = True
        else:
            self.multiple = False

    def __getitem__(self, index):
        batch_files = self.paths[index * self.batch_size:(index + 1) * self.batch_size]
        if self.proton_paths is not None:
            proton_batch_files = self.proton_paths[index * self.batch_size:(index + 1) * self.batch_size]
            proton_images = self.proton_preprocessor.on_files_processor(paths=proton_batch_files,
                                                                        final_points=self.final_points,
                                                                        replacement=self.replacement,
                                                                        truncate=self.truncate,
                                                                        return_features=self.features)
        else:
            proton_images = None
        images = self.preprocessor.on_files_processor(paths=batch_files,
                                                      final_points=self.final_points,
                                                      replacement=self.replacement,
                                                      truncate=self.truncate,
                                                      return_features=self.features)
        images, labels = augment_pointcloud_batch(images, proton_images=proton_images,
                                                  type_training=self.training_type,
                                                  augment=self.augment,
                                                  swap=self.augment,
                                                  jitter=self.jitter,
                                                  rotate=self.rotate,
                                                  return_features=self.features)

        return images, labels

    def __len__(self):
        """
        Returns the length of the list of paths, as the number of events is not known, gives min if using both proton and
        not proton, so only up to the number of proton events will be used, keeping a 1 to 1 ratio for validation and training
        :return:
        """
        if self.proton_paths is not None:
            min_paths = np.min([len(self.proton_paths), len(self.paths)])
            return int(np.ceil(min_paths / float(self.batch_size)))
        else:
            return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.augment:
            self.paths = shuffle(self.paths)
            if self.proton_paths is not None:
                self.proton_paths = shuffle(self.proton_paths)
