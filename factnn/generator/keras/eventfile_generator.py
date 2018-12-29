import numpy as np
from keras.utils import Sequence
from sklearn.utils import shuffle
from factnn.utils.augment import augment_image_batch


class EventFileGenerator(Sequence):

    def __init__(self, paths, batch_size, preprocessor=None, proton_preprocessor=None, proton_paths=None,
                 as_channels=False,
                 final_slices=5, slices=(30, 70), augment=False, training_type=None,
                 normalize=False, truncate=True, dynamic_resize=True, equal_slices=False, multiple=False, collapsed=False, features=False):
        self.paths = paths
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.proton_preprocessor = proton_preprocessor
        self.as_channels = as_channels
        self.final_slices = final_slices
        self.slices = slices
        self.augment = augment
        self.training_type = training_type
        self.proton_paths = proton_paths
        self.normalize = normalize
        self.truncate = truncate
        self.dynamic_resize = dynamic_resize
        self.equal_slices = equal_slices

        # These three are for if multiple inputs need to be returned,
        self.features = features
        self.multiple = multiple
        self.collapsed = collapsed
        #failed_paths = self.proton_preprocessor.check_files(self.paths, "Gamma")
        #self.paths = [x for x in self.paths if x not in failed_paths]
        #sfailed_paths = self.proton_preprocessor.check_files(self.proton_paths, "Proton")
        #self.proton_paths = [x for x in self.proton_paths if x not in sfailed_paths]


    def __getitem__(self, index):
        """
        Go through each set of files and augment them as needed
        :param index:
        :return:
        """
        batch_files = self.paths[index * self.batch_size:(index + 1) * self.batch_size]

        if self.multiple:
            # Return multiple inputs, the basic one, optionally the collapsed image, and the features extracted from the image
            if self.proton_paths is not None:
                proton_batch_files = self.proton_paths[index * self.batch_size:(index + 1) * self.batch_size]
                proton_images = self.proton_preprocessor.on_files_processor(paths=proton_batch_files,
                                                                            final_slices=self.final_slices,
                                                                            normalize=self.normalize,
                                                                            dynamic_resize=self.dynamic_resize,
                                                                            truncate=self.truncate,
                                                                            equal_slices=self.equal_slices)
            else:
                proton_images = None
            images = self.preprocessor.on_files_processor(paths=batch_files, final_slices=self.final_slices,
                                                          normalize=self.normalize, dynamic_resize=self.dynamic_resize,
                                                          truncate=self.truncate, equal_slices=self.equal_slices)
            images, labels = augment_image_batch(images, proton_images=proton_images,
                                                 type_training=self.training_type,
                                                 augment=self.augment,
                                                 swap=self.augment,
                                                 shape=[-1,self.final_slices, self.preprocessor.shape[1], self.preprocessor.shape[2],1],
                                                 as_channels=self.as_channels)
        else:
            if self.proton_paths is not None:
                proton_batch_files = self.proton_paths[index * self.batch_size:(index + 1) * self.batch_size]
                proton_images = self.proton_preprocessor.on_files_processor(paths=proton_batch_files,
                                                                            final_slices=self.final_slices,
                                                                            normalize=self.normalize,
                                                                            dynamic_resize=self.dynamic_resize,
                                                                            truncate=self.truncate,
                                                                            equal_slices=self.equal_slices)
            else:
                proton_images = None
            images = self.preprocessor.on_files_processor(paths=batch_files, final_slices=self.final_slices,
                                                          normalize=self.normalize, dynamic_resize=self.dynamic_resize,
                                                          truncate=self.truncate, equal_slices=self.equal_slices)
            images, labels = augment_image_batch(images, proton_images=proton_images,
                                                 type_training=self.training_type,
                                                 augment=self.augment,
                                                 swap=self.augment,
                                                 shape=[-1,self.final_slices, self.preprocessor.shape[1], self.preprocessor.shape[2],1],
                                                 as_channels=self.as_channels)

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
