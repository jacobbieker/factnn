import numpy as np
import h5py
import fact
import photon_stream as ps
from fact.io import read_h5py
from factnn.data.preprocess.base_preprocessor import BasePreprocessor
from sklearn.utils import shuffle
import pickle
import os


class ProtonPreprocessor(BasePreprocessor):

    def event_processor(self, directory):
        for index, file in enumerate(self.paths):
            mc_truth = file.split(".phs")[0] + ".ch.gz"
            try:
                sim_reader = ps.SimulationReader(
                    photon_stream_path=file,
                    mmcs_corsika_path=mc_truth
                )
                counter = 0
                for event in sim_reader:
                    counter += 1
                    # In the event chosen from the file
                    # Each event is the same as each line below
                    energy = event.simulation_truth.air_shower.energy
                    event_photons = event.photon_stream.list_of_lists
                    zd_deg = event.zd
                    az_deg = event.az
                    act_phi = event.simulation_truth.air_shower.phi
                    act_theta = event.simulation_truth.air_shower.theta
                    data_dict = [[event_photons, energy, zd_deg, az_deg, act_phi, act_theta],
                                 {'Image': 0, 'Energy': 1, 'Zd_Deg': 2, 'Az_Deg': 3, 'COG_Y': 4, 'Phi': 5,
                                  'Theta': 6, }]
                    with open(os.path.join(directory, str(index) + "_" + str(counter)), "wb") as event_file:
                        pickle.dump(data_dict, event_file)
            except Exception as e:
                print(str(e))
                pass

    def batch_processor(self):
        for index, file in enumerate(self.paths):
            mc_truth = file.split(".phs")[0] + ".ch.gz"
            try:
                sim_reader = ps.SimulationReader(
                    photon_stream_path=file,
                    mmcs_corsika_path=mc_truth
                )
                data = []
                for event in sim_reader:
                    # In the event chosen from the file
                    # Each event is the same as each line below
                    energy = event.simulation_truth.air_shower.energy
                    event_photons = event.photon_stream.list_of_lists
                    zd_deg = event.zd
                    az_deg = event.az
                    act_phi = event.simulation_truth.air_shower.phi
                    act_theta = event.simulation_truth.air_shower.theta
                    input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
                    chid_to_pixel = self.rebinning[0]
                    pixel_index_to_grid = self.rebinning[1]
                    for index in range(1440):
                        for element in chid_to_pixel[index]:
                            coords = pixel_index_to_grid[element[0]]
                            for value in event_photons[index]:
                                if self.end > value > self.start:
                                    input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1
                    data.append([np.fliplr(np.rot90(input_matrix, 3)), energy, zd_deg, az_deg, act_phi, act_theta])
                yield data

            except Exception as e:
                print(str(e))

    def single_processor(self, normalize=False, collapse_time=False, final_slices=5, as_channels=False):
        while True:
            self.paths = shuffle(self.paths)
            print("\nNew Proton")
            for index, file in enumerate(self.paths):
                mc_truth = file.split(".phs")[0] + ".ch.gz"
                try:
                    sim_reader = ps.SimulationReader(
                        photon_stream_path=file,
                        mmcs_corsika_path=mc_truth
                    )
                    for event in sim_reader:
                        data = []
                        # In the event chosen from the file
                        # Each event is the same as each line below
                        energy = event.simulation_truth.air_shower.energy
                        event_photons = event.photon_stream.list_of_lists
                        zd_deg = event.zd
                        az_deg = event.az
                        act_phi = event.simulation_truth.air_shower.phi
                        act_theta = event.simulation_truth.air_shower.theta
                        input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
                        chid_to_pixel = self.rebinning[0]
                        pixel_index_to_grid = self.rebinning[1]
                        for index in range(1440):
                            for element in chid_to_pixel[index]:
                                coords = pixel_index_to_grid[element[0]]
                                for value in event_photons[index]:
                                    if self.end > value > self.start:
                                        input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1
                        data.append([np.fliplr(np.rot90(input_matrix, 3)), energy, zd_deg, az_deg, act_phi, act_theta])
                        data_format = {'Image': 0, 'Energy': 1, 'Zd_Deg': 2, 'Az_Deg': 3, 'COG_Y': 4, 'Phi': 5,
                                       'Theta': 6, }
                        data = self.format(data)
                        if normalize:
                            data = list(data)
                            data[0] = self.normalize_image(data[0])
                            data = tuple(data)
                        if collapse_time:
                            data = list(data)
                            data[0] = self.collapse_image_time(data[0], final_slices, as_channels)
                            data = tuple(data)
                        yield data, data_format

                except Exception as e:
                    print(str(e))

    def count_events(self):
        if self.num_events < 0:
            count = 0
            for index, file in enumerate(self.paths):
                mc_truth = file.split(".phs")[0] + ".ch.gz"
                try:
                    sim_reader = ps.SimulationReader(
                        photon_stream_path=file,
                        mmcs_corsika_path=mc_truth
                    )
                    count += sum(1 for _ in sim_reader)
                except Exception as e:
                    print(str(e))
            print(count)
            print('\n')
            self.num_events = count
            return count
        else:
            return self.num_events

    def format(self, batch):
        pic, energy, zd_deg, az_deg, act_phi, act_theta = zip(*batch)
        pic = self.reformat(np.array(pic))
        energy = np.array(energy)
        zd_deg = np.array(zd_deg)
        az_deg = np.array(az_deg)
        act_phi = np.array(act_phi)
        act_theta = np.array(act_theta)
        return pic, energy, zd_deg, az_deg, act_phi, act_theta


class GammaPreprocessor(BasePreprocessor):

    def from_point_cloud_to_list_of_lists(self, dbscan, point_cloud, raw_photons):
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
        :return:
        """
        TIME_SLICE_DURATION_S = 0.5e-9 # Taken from FACT magic constants

        core_sample = dbscan.core_sample_indices_
        print(core_sample)
        # Now go backwards through the raw photons and gather those ones in a new raw photon format
        # That format will be passed to raw_photons_to_list_of_lists to create a new list_of_lists repr

        pixels = fact.instrument.get_pixel_dataframe()
        pixels.sort_values('CHID', inplace=True)

        x_angle = np.deg2rad(pixels.x_angle.values)
        y_angle = np.deg2rad(pixels.y_angle.values)
        fov_radius = np.deg2rad(fact.instrument.camera.FOV_RADIUS)
        new_list_of_list = [[] for _ in range(1440)]
        for element in core_sample:
            # Now have each index into the point cloud, so start backwards
            current_photon = point_cloud[element]
            for index in range(1440):
                if np.isclose(current_photon[0], x_angle[index]) and np.isclose(current_photon[1], y_angle[index]):
                    time_slice = int(np.round(current_photon[2] / TIME_SLICE_DURATION_S))
                    # Now add to new_raw
                    new_list_of_list[index].append(time_slice)

        # Convert back to raw
        new_new_raw = []
        for sublist in new_list_of_list:
            # Go through each sublist in order, adding 255 for each one
            if not sublist:
                new_new_raw.append(255)
                # List is empty
            else:
                # Go through sublist, adding its stuff to it
                for item in sublist:
                    new_new_raw.append(item)
                # Done with sublist, so add end one
                new_new_raw.append(255)
        print(len(new_new_raw) - 1440)
        print("Len Core Sample: ", len(core_sample))


        return NotImplementedError

    def event_processor(self, directory):
        import matplotlib.pyplot as plt
        from sklearn.cluster import DBSCAN
        for index, file in enumerate(self.paths):
            mc_truth = file.split(".phs")[0] + ".ch.gz"
            try:
                sim_reader = ps.SimulationReader(
                    photon_stream_path=file,
                    mmcs_corsika_path=mc_truth
                )
                counter = 0
                for event in sim_reader:
                    counter += 1

                    print(len(event.photon_stream.raw) - 1440)

                    hist_photons = event.photon_stream.image_sequence

                    point_cloud = event.photon_stream.point_cloud
                    print(len(point_cloud))

                    eps = 0.1
                    min_samples = 20
                    deg_over_s = 0.35e9
                    xyt = event.photon_stream.point_cloud.copy()
                    xyt[:, 2] *= np.deg2rad(deg_over_s)

                    fov_radius = np.deg2rad(fact.instrument.camera.FOV_RADIUS)
                    abs_eps = eps * (2.0*fov_radius)

                    dbscan = DBSCAN(eps=abs_eps, min_samples=min_samples).fit(xyt)
                    print("Example point: ", len(dbscan.core_sample_indices_))
                    number = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
                    print("DBSCAN Indicies: ", number)
                    print("Hist Shape: ", hist_photons)
                    self.from_point_cloud_to_list_of_lists(dbscan, point_cloud, raw_photons=event.photon_stream.raw)
                    p
                    plt.show()
                    exit()
                    # In the event chosen from the file
                    # Each event is the same as each line below
                    energy = event.simulation_truth.air_shower.energy
                    event_photons = event.photon_stream.list_of_lists
                    zd_deg = event.zd
                    az_deg = event.az
                    act_phi = event.simulation_truth.air_shower.phi
                    act_theta = event.simulation_truth.air_shower.theta
                    data_dict = [[event_photons, energy, zd_deg, az_deg, act_phi, act_theta],
                                 {'Image': 0, 'Energy': 1, 'Zd_Deg': 2, 'Az_Deg': 3, 'COG_Y': 4, 'Phi': 5,
                                  'Theta': 6, }]
                    with open(os.path.join(directory, str(index) + "_" + str(counter)), "wb") as event_file:
                        pickle.dump(data_dict, event_file)
            except Exception as e:
                print(str(e))
                pass

    def batch_processor(self):
        for index, file in enumerate(self.paths):
            mc_truth = file.split(".phs")[0] + ".ch.gz"
            try:
                sim_reader = ps.SimulationReader(
                    photon_stream_path=file,
                    mmcs_corsika_path=mc_truth
                )
                data = []
                for event in sim_reader:
                    # In the event chosen from the file
                    # Each event is the same as each line below
                    energy = event.simulation_truth.air_shower.energy
                    event_photons = event.photon_stream.list_of_lists
                    zd_deg = event.zd
                    az_deg = event.az
                    act_phi = event.simulation_truth.air_shower.phi
                    act_theta = event.simulation_truth.air_shower.theta
                    input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
                    chid_to_pixel = self.rebinning[0]
                    pixel_index_to_grid = self.rebinning[1]
                    for index in range(1440):
                        for element in chid_to_pixel[index]:
                            coords = pixel_index_to_grid[element[0]]
                            for value in event_photons[index]:
                                if self.end > value > self.start:
                                    input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1
                    data.append([np.fliplr(np.rot90(input_matrix, 3)), energy, zd_deg, az_deg, act_phi, act_theta])
                yield data

            except Exception as e:
                print(str(e))

    def single_processor(self, normalize=False, collapse_time=False, final_slices=5, as_channels=False):
        while True:
            self.paths = shuffle(self.paths)
            print("\nNew Gamma")
            for index, file in enumerate(self.paths):
                mc_truth = file.split(".phs")[0] + ".ch.gz"
                try:
                    sim_reader = ps.SimulationReader(
                        photon_stream_path=file,
                        mmcs_corsika_path=mc_truth
                    )
                    for event in sim_reader:
                        data = []
                        # In the event chosen from the file
                        # Each event is the same as each line below
                        energy = event.simulation_truth.air_shower.energy
                        event_photons = event.photon_stream.list_of_lists
                        zd_deg = event.zd
                        az_deg = event.az
                        act_phi = event.simulation_truth.air_shower.phi
                        act_theta = event.simulation_truth.air_shower.theta
                        input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
                        chid_to_pixel = self.rebinning[0]
                        pixel_index_to_grid = self.rebinning[1]
                        for index in range(1440):
                            for element in chid_to_pixel[index]:
                                coords = pixel_index_to_grid[element[0]]
                                for value in event_photons[index]:
                                    if self.end > value > self.start:
                                        input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 100
                        data.append([np.fliplr(np.rot90(input_matrix, 3)), energy, zd_deg, az_deg, act_phi, act_theta])
                        data_format = {'Image': 0, 'Energy': 1, 'Zd_Deg': 2, 'Az_Deg': 3, 'COG_Y': 4, 'Phi': 5,
                                       'Theta': 6, }
                        data = self.format(data)
                        if normalize:
                            data = list(data)
                            data[0] = self.normalize_image(data[0])
                            data = tuple(data)
                        if collapse_time:
                            data = list(data)
                            data[0] = self.collapse_image_time(data[0], final_slices, as_channels)
                            data = tuple(data)
                        yield data, data_format

                except Exception as e:
                    print(str(e))

    def count_events(self):
        if self.num_events < 0:
            count = 0
            for index, file in enumerate(self.paths):
                mc_truth = file.split(".phs")[0] + ".ch.gz"
                try:
                    sim_reader = ps.SimulationReader(
                        photon_stream_path=file,
                        mmcs_corsika_path=mc_truth
                    )
                    count += sum(1 for _ in sim_reader)
                except Exception as e:
                    print(str(e))
            print(count)
            print('\n')
            self.num_events = count
            return count
        else:
            return self.num_events

    def format(self, batch):
        pic, energy, zd_deg, az_deg, act_phi, act_theta = zip(*batch)
        pic = self.reformat(np.array(pic))
        energy = np.array(energy)
        zd_deg = np.array(zd_deg)
        az_deg = np.array(az_deg)
        act_phi = np.array(act_phi)
        act_theta = np.array(act_theta)
        return pic, energy, zd_deg, az_deg, act_phi, act_theta


class GammaDiffusePreprocessor(BasePreprocessor):

    def init(self):
        self.dl2_file = read_h5py(self.dl2_file, key="events",
                                  columns=["event_num", "source_position_x", "source_position_y", "cog_x", "cog_y",
                                           "delta",
                                           "source_position_az", "source_position_zd",
                                           "aux_pointing_position_az", "aux_pointing_position_zd",
                                           "corsika_event_header_total_energy", "corsika_event_header_az", "run_id"])

    def event_processor(self, directory):
        for index, file in enumerate(self.paths):
            mc_truth = file.split(".phs")[0] + ".ch.gz"
            try:
                sim_reader = ps.SimulationReader(
                    photon_stream_path=file,
                    mmcs_corsika_path=mc_truth
                )
                counter = 0
                for event in sim_reader:
                    df_event = self.dl2_file.loc[(np.isclose(self.dl2_file['corsika_event_header_total_energy'],
                                                             event.simulation_truth.air_shower.energy)) &
                                                 (self.dl2_file['run_id'] == event.simulation_truth.run)]
                    if not df_event.empty:
                        counter += 1
                        # In the event chosen from the file
                        # Each event is the same as each line below
                        cog_x = df_event['cog_x'].values[0]
                        cog_y = df_event['cog_y'].values[0]
                        act_sky_source_zero = df_event['source_position_x'].values[0]
                        act_sky_source_one = df_event['source_position_y'].values[0]
                        event_photons = event.photon_stream.list_of_lists
                        zd_deg = event.zd
                        az_deg = event.az
                        delta = df_event['delta'].values[0]
                        energy = event.simulation_truth.air_shower.energy
                        sky_source_zd = df_event['source_position_zd'].values[0]
                        sky_source_az = df_event['source_position_az'].values[0]
                        zd_deg1 = df_event['aux_pointing_position_az'].values[0]
                        az_deg1 = df_event['aux_pointing_position_zd'].values[0]
                        data_dict = [[event_photons, act_sky_source_zero, act_sky_source_one,
                                      cog_x, cog_y, zd_deg, az_deg, sky_source_zd, sky_source_az, delta,
                                      energy, zd_deg1, az_deg1],
                                     {'Image': 0, 'Source_X': 1, 'Source_Y': 2, 'COG_X': 3, 'COG_Y': 4, 'Zd_Deg': 5,
                                      'Az_Deg': 6,
                                      'Source_Zd': 7, 'Source_Az': 8, 'Delta': 9, 'Energy': 10, 'Pointing_Zd': 11,
                                      'Pointing_Az': 12}]
                        with open(os.path.join(directory, str(index) + "_" + str(counter)), "wb") as event_file:
                            pickle.dump(data_dict, event_file)
            except Exception as e:
                print(str(e))
                pass

    def batch_processor(self):
        for index, file in enumerate(self.paths):
            mc_truth = file.split(".phs")[0] + ".ch.gz"
            try:
                sim_reader = ps.SimulationReader(
                    photon_stream_path=file,
                    mmcs_corsika_path=mc_truth
                )
                data = []
                for event in sim_reader:
                    df_event = self.dl2_file.loc[(np.isclose(self.dl2_file['corsika_event_header_total_energy'],
                                                             event.simulation_truth.air_shower.energy)) &
                                                 (self.dl2_file['run_id'] == event.simulation_truth.run)]
                    if not df_event.empty:
                        # In the event chosen from the file
                        # Each event is the same as each line below
                        cog_x = df_event['cog_x'].values[0]
                        cog_y = df_event['cog_y'].values[0]
                        act_sky_source_zero = df_event['source_position_x'].values[0]
                        act_sky_source_one = df_event['source_position_y'].values[0]
                        event_photons = event.photon_stream.list_of_lists
                        zd_deg = event.zd
                        az_deg = event.az
                        delta = df_event['delta'].values[0]
                        energy = event.simulation_truth.air_shower.energy
                        sky_source_zd = df_event['source_position_zd'].values[0]
                        sky_source_az = df_event['source_position_az'].values[0]
                        zd_deg1 = df_event['aux_pointing_position_az'].values[0]
                        az_deg1 = df_event['aux_pointing_position_zd'].values[0]
                        input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
                        chid_to_pixel = self.rebinning[0]
                        pixel_index_to_grid = self.rebinning[1]
                        for index in range(1440):
                            for element in chid_to_pixel[index]:
                                # Now get the first 60 event photons
                                coords = pixel_index_to_grid[element[0]]
                                for value in event_photons[index]:
                                    if self.end > value > self.start:
                                        input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1

                        data.append([np.fliplr(np.rot90(input_matrix, 3)), act_sky_source_zero, act_sky_source_one,
                                     cog_x, cog_y, zd_deg, az_deg, sky_source_zd, sky_source_az, delta,
                                     energy, zd_deg1, az_deg1])
                yield data

            except Exception as e:
                print(str(e))

    def single_processor(self, normalize=False, collapse_time=False, final_slices=5, as_channels=False):
        while True:
            self.paths = shuffle(self.paths)
            for index, file in enumerate(self.paths):
                mc_truth = file.split(".phs")[0] + ".ch.gz"
                try:
                    sim_reader = ps.SimulationReader(
                        photon_stream_path=file,
                        mmcs_corsika_path=mc_truth
                    )
                    for event in sim_reader:
                        data = []
                        df_event = self.dl2_file.loc[(np.isclose(self.dl2_file['corsika_event_header_total_energy'],
                                                                 event.simulation_truth.air_shower.energy)) &
                                                     (self.dl2_file['run_id'] == event.simulation_truth.run)]
                        if not df_event.empty:
                            # In the event chosen from the file
                            # Each event is the same as each line below
                            cog_x = df_event['cog_x'].values[0]
                            cog_y = df_event['cog_y'].values[0]
                            act_sky_source_zero = df_event['source_position_x'].values[0]
                            act_sky_source_one = df_event['source_position_y'].values[0]
                            event_photons = event.photon_stream.list_of_lists
                            zd_deg = event.zd
                            az_deg = event.az
                            delta = df_event['delta'].values[0]
                            energy = event.simulation_truth.air_shower.energy
                            sky_source_zd = df_event['source_position_zd'].values[0]
                            sky_source_az = df_event['source_position_zd'].values[0]
                            zd_deg1 = df_event['aux_pointing_position_az'].values[0]
                            az_deg1 = df_event['aux_pointing_position_zd'].values[0]
                            input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
                            chid_to_pixel = self.rebinning[0]
                            pixel_index_to_grid = self.rebinning[1]
                            for index in range(1440):
                                for element in chid_to_pixel[index]:
                                    # Now get the first 60 event photons
                                    coords = pixel_index_to_grid[element[0]]
                                    for value in event_photons[index]:
                                        if self.end > value > self.start:
                                            input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1
                            data.append([np.fliplr(np.rot90(input_matrix, 3)), act_sky_source_zero, act_sky_source_one,
                                         cog_x, cog_y, zd_deg, az_deg, sky_source_zd, sky_source_az, delta,
                                         energy, zd_deg1, az_deg1])
                        # Add an associated structure that gives the name?
                        data_format = {'Image': 0, 'Source_X': 1, 'Source_Y': 2, 'COG_X': 3, 'COG_Y': 4, 'Zd_Deg': 5,
                                       'Az_Deg': 6,
                                       'Source_Zd': 7, 'Source_Az': 8, 'Delta': 9, 'Energy': 10, 'Pointing_Zd': 11,
                                       'Pointing_Az': 12}

                        if len(data) != 0:
                            data = self.format(data)
                            if normalize:
                                data = list(data)
                                data[0] = self.normalize_image(data[0])
                                data = tuple(data)
                            if collapse_time:
                                data = list(data)
                                data[0] = self.collapse_image_time(data[0], final_slices, as_channels)
                                data = tuple(data)
                            yield data, data_format

                except Exception as e:
                    print(str(e))

    def count_events(self):
        if self.num_events < 0:
            count = 0
            for index, file in enumerate(self.paths):
                mc_truth = file.split(".phs")[0] + ".ch.gz"
                try:
                    sim_reader = ps.SimulationReader(
                        photon_stream_path=file,
                        mmcs_corsika_path=mc_truth
                    )
                    for event in sim_reader:
                        df_event = self.dl2_file.loc[(np.isclose(self.dl2_file['corsika_event_header_total_energy'],
                                                                 event.simulation_truth.air_shower.energy)) &
                                                     (self.dl2_file['run_id'] == event.simulation_truth.run)]
                        if not df_event.empty:
                            count += 1
                except Exception as e:
                    print(str(e))
            self.num_events = count
            return count
        else:
            return self.num_events

    def format(self, batch):
        pic, act_sky_source_zero, act_sky_source_one, cog_x, cog_y, zd_deg, az_deg, sky_source_zd, sky_source_az, delta, energy, zd_deg1, az_deg1 = zip(
            *batch)
        pic = self.reformat(np.array(pic))
        act_sky_source_zero = np.array(act_sky_source_zero)
        act_sky_source_one = np.array(act_sky_source_one)
        cog_x = np.array(cog_x)
        cog_y = np.array(cog_y)
        zd_deg = np.array(zd_deg)
        az_deg = np.array(az_deg)
        sky_source_zd = np.array(sky_source_zd)
        sky_source_az = np.array(sky_source_az)
        delta = np.array(delta)
        energy = np.array(energy)
        zd_deg1 = np.array(zd_deg1)
        az_deg1 = np.array(az_deg1)
        return pic, act_sky_source_zero, act_sky_source_one, cog_x, cog_y, zd_deg, az_deg, sky_source_zd, \
               sky_source_az, delta, energy, zd_deg1, az_deg1
