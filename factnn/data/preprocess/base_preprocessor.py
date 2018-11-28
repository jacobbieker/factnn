import numpy as np
import fact
from fact.instrument import get_pixel_coords
from fact.instrument.constants import PIXEL_SPACING_MM
from sklearn.cluster import DBSCAN
import pickle
import os
import pkg_resources as res


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
            if config['rebin_size'] <= 300:
                try:
                    with open(res.resource_filename('factnn.data.resources', "rebinning_" + str(config['rebin_size']) + ".p"), "rb") as rebinning_file:
                        self.rebinning = pickle.load(rebinning_file)
                except Exception as e:
                    self.rebinning = self.generate_rebinning(config['rebin_size'])
            else:
                self.rebinning = self.generate_rebinning(config['rebin_size'])
        else:
            self.rebinning = self.generate_rebinning(50)

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

        self.shape = [-1, config['rebin_size'], config['rebin_size'], self.end - self.start]

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

        steps = size # Now size of 100 should make a 100x100 grid

        square_size = np.abs(square_start * 2 / steps) # Now this is the size of the grid

        square = Polygon([(-square_start, square_start), (-square_start + square_size, square_start),
                          (-square_start + square_size, square_start - square_size),
                          (-square_start, square_start - square_size),
                          (-square_start, square_start)])

        list_of_squares = [square]

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

    def batch_processor(self, clean_images=False):
        return NotImplemented

    def single_processor(self, normalize=False, collapse_time=False, final_slices=5, clean_images=False):
        return NotImplemented

    def on_batch_processor(self, filepath, size, sample=False, normalize=False, collapse_time=False, final_slices=5, clean_images=False):
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

    def event_processor(self, directory, clean_images=False):
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

    def normalize_image(self, image, per_slice=True):
        """
        Assumes Image in the format given by reformat, and designed for single processor
        :param per_slice: Whether to nrom along each time slice or through the whole data cube
        :param image:
        :return:
        """
        # Now have the whole data image, go through an normalize each slice
        temp_matrix = []
        # Should be just one datacube per image
        for data_cube in image:
            if per_slice:
                for image_slice in data_cube:
                    # Each time slice you normalize
                    mean = np.mean(image_slice)
                    stddev = np.std(image_slice)
                    denom = np.max([stddev, 1.0/np.sqrt(image_slice.size)])
                    image_slice = (image_slice - mean) / denom
                    temp_matrix.append(image_slice)
            else:
                # Do it over the whole timeslice/channels
                mean = np.mean(data_cube)
                stddev = np.std(data_cube)
                denom = np.max([stddev, 1.0/np.sqrt(data_cube.size)])
                data_cube = (data_cube - mean) / denom
                temp_matrix = data_cube
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

    def select_clustered_photons(self, dbscan, point_cloud, debug=True):
        """
        Take DBSCAN output on the point cloud and translate it back to a list of lists of photons

        Useful for using the clustering to reduce the noise in the image stacks later, image cleaning

        The raw photons are stored in order of CHID, with 255 separating the different pixels and the
        numbers indicating the arrival time of the photon

        Can be found by using the core samples from DBSCAN aand only keeping those photons
        Core smaples define the clusters, those that are not core samples are on the fringe of the sample
        and we can test discarding them to clean the image further, so only dbscan.core_sample_indicies_ is
        needed really

        Raw to point cloud is given below where cx and cy are the geometry of the image, but not needed here

        def raw_phs_to_point_cloud(raw_phs, cx, cy):
            number_photons = len(raw_phs) - NUMBER_OF_PIXELS
            cloud = np.zeros(shape=(number_photons, 3))
            pixel_chid = 0
            p = 0
            for s in raw_phs:
                if s == io.binary.LINEBREAK:
                    pixel_chid += 1
                else:
                    cloud[p, 0] = cx[pixel_chid]
                    cloud[p, 1] = cy[pixel_chid]
                    cloud[p, 2] = s*TIME_SLICE_DURATION_S
                    p += 1
            return cloud

        :param dbscan:
        :param raw_photons:
        :return: New raw photon event, or None if no clumps are found
        """
        TIME_SLICE_DURATION_S = 0.5e-9 # Taken from FACT magic constants

        core_sample = dbscan.core_sample_indices_
        number = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        # Now go backwards through the raw photons and gather those ones in a new raw photon format
        # That format will be passed to raw_photons_to_list_of_lists to create a new list_of_lists repr

        pixels = fact.instrument.get_pixel_dataframe()
        pixels.sort_values('CHID', inplace=True)

        x_angle = np.deg2rad(pixels.x_angle.values)
        y_angle = np.deg2rad(pixels.y_angle.values)
        new_list_of_list = [[] for _ in range(1440)]
        list_of_slices = []
        for element in core_sample:
            # Now have each index into the point cloud, so start backwards
            current_photon = point_cloud[element]
            for index in range(1440):
                if np.isclose(current_photon[0], x_angle[index]) and np.isclose(current_photon[1], y_angle[index]):
                    time_slice = int(np.round(current_photon[2] / TIME_SLICE_DURATION_S))
                    list_of_slices.append(time_slice)
                    # Now add to new_raw
                    new_list_of_list[index].append(time_slice)

        # Convert back to raw
        new_raw = []
        for sublist in new_list_of_list:
            # Go through each sublist in order, adding 255 for each one
            if not sublist:
                new_raw.append(255)
                # List is empty
            else:
                # Go through sublist, adding its stuff to it
                for item in sublist:
                    new_raw.append(item)
                # Done with sublist, so add end one
                new_raw.append(255)
        if number == 0:
            # No clumps, so returns None
            return None
        if debug:
            print("Start: {}, End: {}, Mean: {}, Std: {} Clumps: {}".format(np.min(list_of_slices), np.max(list_of_slices), np.mean(list_of_slices), np.std(list_of_slices), number))
        return new_raw

    def clean_image(self, event, min_samples=20, eps=0.1):
        """
        Clean the image with various methods, currently only DBSCAN

        DBSCAN code is taken almost directly from pyfact


        :param event: PhotonStream Event
        :param min_samples: Min samples for DBSCAN
        :param eps: maximal distance between two samples to be considered same neighborhood
        :return: The same Photon Stream Event with only photons in a cluster
        """

        point_cloud = event.photon_stream.point_cloud

        deg_over_s = 0.35e9
        xyt = event.photon_stream.point_cloud.copy()
        xyt[:, 2] *= np.deg2rad(deg_over_s)

        fov_radius = np.deg2rad(fact.instrument.camera.FOV_RADIUS)
        abs_eps = eps * (2.0*fov_radius)

        dbscan = DBSCAN(eps=abs_eps, min_samples=min_samples).fit(xyt)

        event.photon_stream.raw = self.select_clustered_photons(dbscan, point_cloud)

        return event

    def dynamic_size(self, photon_stream):
        """
        Takes a photon stream list of lists representation and finds the start and end of the photons in that and returns the indexes
        :param photon_stream:
        :return: (start,end)
        """

        length = len(sorted(photon_stream,key=len, reverse=True)[0])
        arr = np.array([xi+[None]*(length-len(xi)) for xi in photon_stream])

        start = np.min(arr)
        end = np.max(arr)

        return (start, end)

    def format(self, batch):
        return NotImplemented
