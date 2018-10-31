import numpy as np
import h5py
import photon_stream as ps
from fact.io import read_h5py
from factnn.preprocess.base_preprocessor import BasePreprocessor
from sklearn.utils import shuffle


class ProtonPreprocessor(BasePreprocessor):

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

    def single_processor(self, normalize=False):
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
                        data_format = {'Image': 0, 'Energy': 1, 'Zd_Deg': 2, 'Az_Deg': 3, 'COG_Y': 4, 'Phi': 5, 'Theta': 6,}
                        data = self.format(data)
                        if normalize:
                            data = list(data)
                            data[0] = self.normalize_image(data[0])
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

    def create_dataset(self):
        gen = self.batch_processor()
        batch = next(gen)
        pic, energy, zd_deg, az_deg, act_phi, act_theta = self.format(batch)
        row_count = az_deg.shape[0]
        print(row_count)

        with h5py.File(self.output_file, 'w') as hdf:
            maxshape_pic = (None,) + pic.shape[1:]
            dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape_pic, chunks=pic.shape,
                                          dtype=pic.dtype)
            maxshape_run = (None,) + zd_deg.shape[1:]
            dset_energy = hdf.create_dataset('Energy', shape=energy.shape, maxshape=maxshape_run, chunks=energy.shape,
                                             dtype=energy.dtype)
            maxshape_event = (None,) + zd_deg.shape[1:]
            dset_zd_deg = hdf.create_dataset('Zd_deg', shape=zd_deg.shape, maxshape=maxshape_event, chunks=zd_deg.shape,
                                             dtype=zd_deg.dtype)
            maxshape_az_deg = (None,) + zd_deg.shape[1:]
            dset_az_deg = hdf.create_dataset('Az_deg', shape=az_deg.shape, maxshape=maxshape_az_deg,
                                             chunks=az_deg.shape, dtype=az_deg.dtype)
            maxshape_phia = (None,) + zd_deg.shape[1:]
            dset_phia = hdf.create_dataset('Phi', shape=act_phi.shape, maxshape=maxshape_phia, chunks=act_phi.shape,
                                           dtype=act_phi.dtype)
            maxshape_thetaa = (None,) + zd_deg.shape[1:]
            dset_thetaa = hdf.create_dataset('Theta', shape=act_theta.shape, maxshape=maxshape_thetaa,
                                             chunks=act_theta.shape, dtype=act_theta.dtype)

            dset_pic[:] = pic
            dset_energy[:] = energy
            dset_zd_deg[:] = zd_deg
            dset_az_deg[:] = az_deg
            dset_phia[:] = act_phi
            dset_thetaa[:] = act_theta

            for batch in gen:
                pic, energy, zd_deg, az_deg, phi, theta = self.format(batch)

                shape = theta.shape[0]

                dset_pic.resize(row_count + shape, axis=0)
                dset_energy.resize(row_count + shape, axis=0)
                dset_zd_deg.resize(row_count + shape, axis=0)
                dset_az_deg.resize(row_count + shape, axis=0)
                dset_phia.resize(row_count + shape, axis=0)
                dset_thetaa.resize(row_count + shape, axis=0)

                dset_pic[row_count:] = pic
                dset_energy[row_count:] = energy
                dset_zd_deg[row_count:] = zd_deg
                dset_az_deg[row_count:] = az_deg
                dset_phia[row_count:] = phi
                dset_thetaa[row_count:] = theta

                row_count += phi.shape[0]


class GammaPreprocessor(BasePreprocessor):

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

    def single_processor(self, normalize=False):
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
                        data_format = {'Image': 0, 'Energy': 1, 'Zd_Deg': 2, 'Az_Deg': 3, 'COG_Y': 4, 'Phi': 5, 'Theta': 6,}
                        data = self.format(data)
                        if normalize:
                            data = list(data)
                            data[0] = self.normalize_image(data[0])
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

    def create_dataset(self):
        gen = self.batch_processor()
        batch = next(gen)
        pic, energy, zd_deg, az_deg, act_phi, act_theta = self.format(batch)
        row_count = az_deg.shape[0]
        print(row_count)

        with h5py.File(self.output_file, 'w') as hdf:
            maxshape_pic = (None,) + pic.shape[1:]
            dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape_pic, chunks=pic.shape,
                                          dtype=pic.dtype)
            maxshape = (None,) + zd_deg.shape[1:]
            shape = energy.shape
            dset_energy = hdf.create_dataset('Energy', shape=shape, maxshape=maxshape, chunks=shape,dtype=energy.dtype)
            dset_zd_deg = hdf.create_dataset('Zd_deg', shape=shape, maxshape=maxshape, chunks=shape,dtype=zd_deg.dtype)
            dset_az_deg = hdf.create_dataset('Az_deg', shape=shape, maxshape=maxshape,chunks=shape, dtype=az_deg.dtype)
            dset_phia = hdf.create_dataset('Phi', shape=shape, maxshape=maxshape, chunks=shape,dtype=act_phi.dtype)
            dset_thetaa = hdf.create_dataset('Theta', shape=shape, maxshape=maxshape,chunks=shape, dtype=act_theta.dtype)

            dset_pic[:] = pic
            dset_energy[:] = energy
            dset_zd_deg[:] = zd_deg
            dset_az_deg[:] = az_deg
            dset_phia[:] = act_phi
            dset_thetaa[:] = act_theta

            for batch in gen:
                pic, energy, zd_deg, az_deg, phi, theta = self.format(batch)

                shape = energy.shape[0]

                dset_pic.resize(row_count + shape, axis=0)
                dset_energy.resize(row_count + shape, axis=0)
                dset_zd_deg.resize(row_count + shape, axis=0)
                dset_az_deg.resize(row_count + shape, axis=0)
                dset_phia.resize(row_count + shape, axis=0)
                dset_thetaa.resize(row_count + shape, axis=0)

                dset_pic[row_count:] = pic
                dset_energy[row_count:] = energy
                dset_zd_deg[row_count:] = zd_deg
                dset_az_deg[row_count:] = az_deg
                dset_phia[row_count:] = phi
                dset_thetaa[row_count:] = theta

                row_count += shape


class GammaDiffusePreprocessor(BasePreprocessor):

    def init(self):
        self.dl2_file = read_h5py(self.dl2_file, key="events",
                                  columns=["event_num", "source_position_x", "source_position_y", "cog_x", "cog_y", "delta",
                                           "source_position_az", "source_position_zd",
                                           "aux_pointing_position_az", "aux_pointing_position_zd",
                                           "corsika_event_header_total_energy", "corsika_event_header_az", "run_id"])

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

    def single_processor(self, normalize=False):
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
                        data_format = {'Image': 0, 'Source_X': 1, 'Source_Y': 2, 'COG_X': 3, 'COG_Y': 4, 'Zd_Deg': 5, 'Az_Deg': 6,
                                       'Source_Zd': 7, 'Source_Az': 8, 'Delta': 9, 'Energy': 10, 'Pointing_Zd': 11, 'Pointing_Az': 12}

                        if len(data) != 0:
                            data = self.format(data)
                            if normalize:
                                data = list(data)
                                data[0] = self.normalize_image(data[0])
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

    def create_dataset(self):
        gen = self.batch_processor()
        batch = next(gen)
        pic, act_sky_source_zero, act_sky_source_one, cog_x, cog_y, zd_deg, az_deg, sky_source_zd, \
        sky_source_az, delta, energy, zd_deg1, az_deg1 = self.format(batch)
        row_count = az_deg.shape[0]
        print(row_count)

        with h5py.File(self.output_file, 'w') as hdf:
            maxshape = (None,) + pic.shape[1:]
            shape = energy.shape
            dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape, chunks=pic.shape,
                                          dtype=pic.dtype)
            maxshape = (None, ) + energy.shape[1:]
            dset_source_pos_x = hdf.create_dataset('Source_X', shape=energy.shape, maxshape=maxshape, chunks=shape,
                                                   dtype=act_sky_source_zero.dtype)
            dset_source_pos_y = hdf.create_dataset('Source_Y', shape=shape, maxshape=maxshape, chunks=shape,
                                                   dtype=act_sky_source_one.dtype)
            dset_zd_deg = hdf.create_dataset('Zd_deg', shape=shape, maxshape=maxshape, chunks=shape, dtype=zd_deg.dtype)
            dset_az_deg = hdf.create_dataset('Az_deg', shape=shape, maxshape=maxshape, chunks=shape, dtype=az_deg.dtype)
            dset_cog_x = hdf.create_dataset('COG_X', shape=shape, maxshape=maxshape, chunks=shape, dtype=cog_x.dtype)
            dset_cog_y = hdf.create_dataset('COG_Y', shape=shape, maxshape=maxshape, chunks=shape, dtype=cog_y.dtype)
            dset_delta = hdf.create_dataset('Delta', shape=shape, maxshape=maxshape, chunks=shape, dtype=delta.dtype)
            dset_source_zd = hdf.create_dataset('Source_Zd', shape=shape, maxshape=maxshape, chunks=shape,
                                                dtype=sky_source_zd.dtype)
            dset_source_az = hdf.create_dataset('Source_Az', shape=shape, maxshape=maxshape, chunks=shape,
                                                dtype=sky_source_az.dtype)
            dset_energy = hdf.create_dataset('Energy', shape=shape, maxshape=maxshape, chunks=shape, dtype=energy.dtype)
            dset_zd_deg1 = hdf.create_dataset('Pointing_Zd', shape=shape, maxshape=maxshape, chunks=zd_deg1.shape,
                                              dtype=zd_deg1.dtype)
            dset_az_deg1 = hdf.create_dataset('Pointing_Az', shape=shape, maxshape=maxshape, chunks=az_deg1.shape,
                                              dtype=az_deg1.dtype)

            dset_pic[:] = pic
            dset_source_pos_x[:] = act_sky_source_zero
            dset_source_pos_y[:] = act_sky_source_one
            dset_cog_x[:] = cog_x
            dset_zd_deg[:] = zd_deg
            dset_az_deg[:] = az_deg
            dset_cog_y[:] = cog_y
            dset_delta[:] = delta
            dset_source_zd[:] = sky_source_zd
            dset_source_az[:] = sky_source_az
            dset_energy[:] = energy
            dset_zd_deg1[:] = zd_deg1
            dset_az_deg1[:] = az_deg1

            for batch in gen:
                pic, act_sky_source_zero, act_sky_source_one, cog_x, cog_y, zd_deg, az_deg, sky_source_zd, \
                sky_source_az, delta, energy, zd_deg1, az_deg1 = self.format(batch)

                shape = energy.shape[0]

                dset_pic.resize(row_count + shape, axis=0)
                dset_source_pos_x.resize(row_count + shape, axis=0)
                dset_cog_x.resize(row_count + shape, axis=0)
                dset_cog_y.resize(row_count + shape, axis=0)
                dset_zd_deg.resize(row_count + shape, axis=0)
                dset_az_deg.resize(row_count + shape, axis=0)
                dset_delta.resize(row_count + shape, axis=0)
                dset_source_pos_y.resize(row_count + shape, axis=0)
                dset_source_zd.resize(row_count + shape, axis=0)
                dset_source_az.resize(row_count + shape, axis=0)
                dset_energy.resize(row_count + shape, axis=0)
                dset_zd_deg1.resize(row_count + shape, axis=0)
                dset_az_deg1.resize(row_count + shape, axis=0)

                dset_pic[row_count:] = pic
                dset_source_pos_x[row_count:] = act_sky_source_zero
                dset_source_pos_y[row_count:] = act_sky_source_one
                dset_cog_x[row_count:] = cog_x
                dset_zd_deg[row_count:] = zd_deg
                dset_az_deg[row_count:] = az_deg
                dset_cog_y[row_count:] = cog_y
                dset_delta[row_count:] = delta
                dset_source_zd[row_count:] = sky_source_zd
                dset_source_az[row_count:] = sky_source_az
                dset_energy[row_count:] = energy
                dset_zd_deg1[row_count:] = zd_deg1
                dset_az_deg1[row_count:] = az_deg1

                row_count += shape
