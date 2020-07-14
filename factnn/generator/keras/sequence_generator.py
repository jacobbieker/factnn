import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence


class BaseSequence(Sequence):
    def __init__(
        self,
        paths,
        num_elements,
        batch_size,
        preprocessor=None,
        proton_preprocessor=None,
        as_channels=False,
        final_slices=5,
        slices=(30, 70),
        augment=False,
    ):
        """
        Idea for this is to be simpler than the base_generator, less options and less have everything and the kitchen sink
        Instead, each BaseSequence will do one thing, given the paths assigned to it, take a uniform sample of the images in each file,
        get their labels/ augment the images, and return them for use

        For prediction purposes, the use of the original generator is perferred, as it will go through all the events in
        all the files given to it, unlike this one where that is not currently guaranteed

        Same for validation purposes, but that has to be fixed to really work, for now just use the resoivoir sampling

        Could have the on_batch_preprocess return both the data, data_format, and whether the file was exhausted or not,
        if not, on the next call, that file is continued until it is exhausted....

        Downside is having to work through the whole file to get to the part that hasn't been done
        Same with the iterators going through the entire file each time

        Downside to tracking how far into the file you are is that there is no guarantee that that worker will get the
        same file as before, most likely it will not and so the same events will be read, so the reservoir sampling makes
        more sense to do, although not great for testing. Also means validation data is not the same everytime.

        Best solution would be to change each event file into the one image per file structure and using that

        Could do that the index is ignored as long as there are more events in a file.

        Have second SequenceGenerator that is based off each event is its own file, with preprocessors that write out the
        image and necessary aux data to individual files for use in this later. Need to open many more files, but then
        gets the benefit of real randomness, better shuffling, and no need to resevoir sample.

        Resevooir sampling will be done for this one, makes it slower but better distributed, each pass through opens each
        file once, samples it, and then moves on. Meaning different data each time for validation and training, but overall
        should be comsistent over time

        Second sequence generator takes the paths to files in individual directories, assumes one image per file, and
        goes from there

        So options would then be the current generators, this SquenceGenerator for less duplication in training and validation

        Validation data might be better served by using the other generator class to iterate completely through the dataset

        Otherwise, the validation data could be set to use the first few elements from each file instead, the batch size,
        but this makes the validation set much smaller than otherwise usually used


        Final setup then would be:

        BaseGenerator for HDF5 files, streaming in files with infinite loops if no need for multiprocesssing
        SequenceGenerator for streaming in files, with each file used once per iteration, regardless of how much of it is used
        EventGenerator for streaming in files where there is one event per file, still a form of Sequence Generator,
        but designed for a single event per file setup

        Other things needed:
        Preprocessor on_batch_preprocess for SequenceGenerator
        Way to generate the one event per file setup
        New class of DataFormatter?

        Dataset class that holds the parts for creating the HDF5 dataset, one file per event conversion, and

        New Design:
            models -> holds most useful models
            docs -> docs
            resources -> resources
            tests -> tests
            utils -> augment, plotting parts
            data -> data.preprocess holds preprocessors
            data -> data.dataset holds creating HDF5 converthing to different storage formats
            generator -> generator.keras holds Sequence-based generators
            generator -> generator.generator holds BaseGenerator-based generators


        :param paths:
        :param num_elements:
        :param batch_size: Maximum size of batch, might be smaller depending on how many events are left in each file
        :param preprocessor:
        :param proton_preprocessor:
        :param as_channels:
        :param final_slices:
        :param slices:
        :param augment:
        """
        self.paths = paths
        self.num_elements = num_elements
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.proton_preprocessor = proton_preprocessor
        self.as_channels = as_channels
        self.final_slices = final_slices
        self.slices = slices
        self.augment = augment

    def __getitem__(self, index):
        """
        Use index to get the file to use, then use ressivoir sampling on each file to get the random files to get
        Involves adding to the preprocessors a on_batch_preprocessor that returns a batch number of images and aux data
        Those are then sent to the augment ones to get the actually augmented results, if needed
        :param index:
        :return:
        """
        if self.augment:
            # Augmenting, so do a resoivor sample batch_size large of the whole file, the run it through the augment
            # Much slower, but with multiprocessing hopefully fine enough
            images, data_format = self.preprocessor.on_batch_preprocess(
                datafile=self.paths[index], size=self.batch_size, augment=self.augment
            )
            pass
        else:
            # Not augment, so take the first batch_size elements in the file
            self.preprocessor.on_batch_preprocess(
                datafile=self.paths[index], size=self.batch_size, augment=self.augment
            )
            pass

    def __len__(self):
        """
        Returns the length of the list of paths, as the number of events is not known
        :return:
        """
        return len(self.paths)

    def on_epoch_end(self):
        if self.augment:
            self.paths = shuffle(self.paths)

    def iter_sample_fast(self, iterable, samplesize):
        results = []
        iterator = iter(iterable)
        # Fill in the first samplesize elements:
        for _ in range(samplesize):
            results.append(next(iterator))
        np.random.shuffle(results)  # Randomize their positions
        for i, v in enumerate(iterator, samplesize):
            r = np.random.randint(0, i)
            if r < samplesize:
                results[r] = v  # at a decreasing rate, replace random items

        if len(results) < samplesize:
            raise ValueError("Sample larger than population.")
        return results
