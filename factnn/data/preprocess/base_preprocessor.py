import numpy as np
from fact.instrument import get_pixel_coords
from fact.instrument.constants import PIXEL_SPACING_MM
import pickle
import os

class BasePreprocessor(object):

    def __init__(self, config):
        if 'directories' in config:
            self.directories = config['directories']
        if 'paths' in config:
            self.paths = config['paths']
        else:
            # Get paths from the directories
            self.paths = []
            for directory in self.directories:
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith("phs.jsonl.gz"):
                            self.paths.append(os.path.join(root, file))

        if 'dl2_file' in config:
            self.dl2_file = config['dl2_file']
        else:
            self.dl2_file = None

        if 'rebin_size' in config:
            if config['rebin_size'] <= 10:
                try:
                    with open(os.path.join("..","factnn", "resources", "rebinning_" + str(config['rebin_size']) + ".p"), "rb") as rebinning_file:
                        self.rebinning = pickle.load(rebinning_file)
                except Exception as e:
                    self.rebinning = self.generate_rebinning(config['rebin_size'])
            else:
                self.rebinning = self.generate_rebinning(config['rebin_size'])
        else:
            self.rebinning = self.generate_rebinning(5)

        if 'gaussian' in config:
            if config['gaussian']:
                self.rebinning = self.generate_rebin_fractions()

        if 'shape' in config:
            self.start = config['shape'][0]
            self.end = config['shape'][1]
        else:
            # Get it from the rebinning
            self.end = 100
            self.start = 0

        self.shape = [-1, int(np.ceil(np.abs(186 * 2) / config['rebin_size'])), int(np.ceil(np.abs(186 * 2) / config['rebin_size'])), self.end - self.start]

        self.dataset = None
        if 'output_file' in config:
            self.output_file = config['output_file']
        else:
            self.output_file = None

        self.num_events = -1

        if 'as_channels' in config:
            self.as_channels = config['as_channels']
        else:
            self.as_channels = False

        self.init()

    def init(self):
        """
        Recalcs the file paths if called based on self.directories
        :return:
        """
        return NotImplemented

    def generate_rebin_fractions(self):
        """
        Generates the fraction where each CHID pixel needs to go for the Gaussian rebin
        :return: Fractions for x and y pixel values, can be then used with any grid size
        """
        x, y = get_pixel_coords()

        # Need to know the fractional dependence in each direction, and the difference between the x and y directions

        range_x = np.abs(np.min(x) - np.max(x))
        range_y = np.abs(np.min(y) - np.max(y))

        # Ratio is 1.02... X/Y so to create correct image on square grid, need to change center of Gaussian so that
        # X/Y so multiply Y coordinates by 1.02... to make a square grid

        # So now need to get the fractional point, scale Y fraction by 1.02... and then put in the square grid

        x += np.min(x)
        y += np.min(y)

        # Now can scale just by fraction of whole

        x /= np.max(x)
        y /= np.max(y)

        ratio = (range_x/range_y)
        y *= ratio

        pixel_fractions = []

        for index in range(1440):
            pixel_fractions.append((x[index], y[index]))

        pixel_fractions = np.asarray(pixel_fractions)
        return pixel_fractions

    def generate_rebinning(self, size):

        from shapely.geometry import Point, Polygon, MultiPoint
        from shapely.affinity import translate

        p = Point(0.0, 0.0)
        PIXEL_EDGE = 9.51 / np.sqrt(3)
        # Top one
        p1 = Point(0.0, PIXEL_EDGE)
        # Bottom one
        p2 = Point(0.0, -PIXEL_EDGE)
        # Bottom right
        p3 = Point(-PIXEL_EDGE * (np.sqrt(3) / 2), -PIXEL_EDGE * .5)
        # Bottom left
        p4 = Point(PIXEL_EDGE * (np.sqrt(3) / 2), PIXEL_EDGE * .5)
        # right
        p5 = Point(PIXEL_EDGE * (np.sqrt(3) / 2), -PIXEL_EDGE * .5)
        #  left
        p6 = Point(-PIXEL_EDGE * (np.sqrt(3) / 2), PIXEL_EDGE * .5)

        hexagon = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull

        square_start = 186
        square_size = size
        square = Polygon([(-square_start, square_start), (-square_start + square_size, square_start),
                          (-square_start + square_size, square_start - square_size),
                          (-square_start, square_start - square_size),
                          (-square_start, square_start)])

        list_of_squares = [square]
        steps = int(np.ceil(np.abs(square_start * 2) / square_size))

        pixel_index_to_grid = {}
        pix_index = 0
        # Generate tessellation of grid
        for x_step in range(steps):
            for y_step in range(steps):
                new_square = translate(square, xoff=x_step * square_size, yoff=-square_size * y_step)
                pixel_index_to_grid[pix_index] = [x_step, y_step]
                pix_index += 1
                list_of_squares.append(new_square)

        x, y = get_pixel_coords()
        list_hexagons = []
        for index, x_coor in enumerate(x):
            list_hexagons.append(translate(hexagon, x_coor, y[index]))
        list_pixels_and_fractions = {}
        for i in range(len(list_of_squares)):
            list_pixels_and_fractions[i] = []

        chid_to_pixel = {}
        for i in range(1440):
            chid_to_pixel[i] = []

        for pixel_index, pixel in enumerate(list_of_squares):
            for chid, hexagon in enumerate(list_hexagons):
                # Do the dirty work, hexagons should be in CHID order because translate in that order and append
                if pixel.intersects(hexagon):
                    intersection = pixel.intersection(hexagon)
                    fraction_whole = intersection.area / hexagon.area
                    if not np.isclose(fraction_whole, 0.0):
                        # so not close to zero overlap, add to list for that pixel
                        list_pixels_and_fractions[np.abs(pixel_index)].append((chid, fraction_whole))
                        chid_to_pixel[np.abs(1439 - chid)].append((pixel_index, fraction_whole))

        hex_to_grid = [chid_to_pixel, pixel_index_to_grid]
        return hex_to_grid

    def batch_processor(self):
        return NotImplemented

    def single_processor(self, normalize=False, collapse_time=False, final_slices=5):
        return NotImplemented

    def on_batch_processor(self, filepath, size, sample=False, normalize=False, collapse_time=False, final_slices=5):
        """
        Returns at most size-elements from the file at filepath, if sample=True, then does resevoir sampling of the entire
        file, otherwise takes the first size-elements
        :param size: Number of elements to return, usually batch size
        :param normalize: Whether to normalize the images or not
        :param collapse_time: Whether to collapse the time axis to final_slices number of slices
        :param final_slices: Number of slices to collapse the time axis to, defaults to 5
        :return:
        """
        return NotImplementedError

    def event_processor(self, directory):
        """
        Goes through each event in all the files specified in self.paths and returns each event individually, including the
        default photon-stream representation, and auxiliary data and saves it to a new file based on the
        :return:
        """

    def count_events(self):
        """
        Ideally to count the number of events in the files for the streaming data
        :return:
        """
        return NotImplementedError

    def normalize_image(self, image):
        """
        Assumes Image in the format given by reformat, and designed for single processor
        :param image:
        :return:
        """
        # Now have the whole data image, go through an normalize each slice
        temp_matrix = []
        for data_cube in image:
            for image_slice in data_cube:
                # Each time slice you normalize
                mean = np.mean(image_slice)
                stddev = np.std(image_slice)
                denom = np.max([stddev, 1.0/np.sqrt(image_slice.size)])
                image_slice = (image_slice - mean) / denom
                temp_matrix.append(image_slice)
        # Should be normalized now
        temp_matrix = np.array(temp_matrix)
        temp_matrix = temp_matrix.reshape(1, temp_matrix.shape[0], temp_matrix.shape[1], temp_matrix.shape[2])
        return temp_matrix

    def collapse_image_time(self, image, final_slices, as_channels=False):
        """
        Partially flattens an image cube to a smaller set, e.g. (1,40,75,75) with final_slices=3 becomes
        (1,3,75,75) with each new slice being a sum of the fraction of slices of the whole

        If as_channels is True, then the time_slices are moved to the channels, so the previous example
        would end up with the final shape (1,75,75,3)

        :param image: The image in (1, time_slices, width, height, channel) order
        :param final_slices: Number of slices to use
        :param as_channels: Boolean, if the time dimension should be moved to the channels
        :return: Converted image cube with the proper dimensions
        """
        temp_matrix = []
        num_slices_per_final_slice = int(np.floor(image.shape[1] / final_slices))
        for data_cube in image:
            # Need to now sum up along each smaller section
            for time_slice in range(final_slices):
                if time_slice < (final_slices - 1):
                    image_slice = data_cube[time_slice*num_slices_per_final_slice:((time_slice+1)*num_slices_per_final_slice), ::]
                else:
                    # To use all the available slices
                    image_slice = data_cube[time_slice*num_slices_per_final_slice:, ::]
                image_slice = np.sum(image_slice, axis=0)
                temp_matrix.append(image_slice)
        # Should be normalized now
        temp_matrix = np.array(temp_matrix)
        # Now to convert to chennel format if needed
        if as_channels:
            temp_matrix = np.swapaxes(temp_matrix, 0, 2)
            # Second one is to keep the order of the width/height
            temp_matrix = np.swapaxes(temp_matrix, 0, 1)
            temp_matrix = temp_matrix.reshape(1, temp_matrix.shape[0], temp_matrix.shape[1], temp_matrix.shape[2])
        else:
            # Else keep same format as before
            temp_matrix = temp_matrix.reshape(1, temp_matrix.shape[0], temp_matrix.shape[1], temp_matrix.shape[2])
        return temp_matrix

    def reformat(self, image):
        """
        Reformats image to what is needed for LSTM with time, width, height, channel order
        :param image:
        :return:
        """
        dataset = np.swapaxes(image, 1, 3)
        dataset = np.array(dataset).reshape((self.shape[0], self.shape[3], self.shape[2], self.shape[1])).astype(np.float32)
        return dataset

    def convert_to_gaussian_image(self, phs_photons, sigma, size, delta=PIXEL_SPACING_MM/2, normalize=None, as_channels=False):
        """
        Converts a list of lists of number of photons to create a final gaussian image
        :param phs_photons: e.g. for final slices = 2, would be similar to [[2,3,4,...],[5,1,2,...]]
        So each final slice add another sublist to the lists, while a final slice of 1 is = [[1,2,3,5,...]]
        :param sigma: Width of the gaussian, default should be delta/2
        :param delta: Pixel physcal distance/pixel size
        :param size: Size of one side of square grid, i.e. 100 => 100x100
        :param normalize: Whether to normalize to each grid or not, one of None, 'slice' for per_slice normalization,
        and 'full' for normalization over the full image
        :param as_channels: Whether to return with time_slice as channels or not
        :return: The Gaussian image made up of final_slices slices and possibly normalized
        """

        image = np.zeros(shape=(size, size, len(phs_photons)))
        for slice_index, single_slice in enumerate(phs_photons):
            pass

        return NotImplementedError

    def format(self, batch):
        return NotImplemented
