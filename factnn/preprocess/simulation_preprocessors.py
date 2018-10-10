import numpy as np
import h5py
import photon_stream as ps
from fact.io import read_h5py
from factnn.preprocess.base_preprocessor import BasePreprocessor


class ProtonPreprocessor(BasePreprocessor):

    def batch_processor(self):
        for index, file in enumerate(self.paths):
            mc_truth = file.split(".phs")[0] + ".ch.gz"
            print(mc_truth)
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
                    input_matrix = np.zeros([self.shape[1],self.shape[2],self.shape[3]])
                    chid_to_pixel = self.rebinning[0]
                    pixel_index_to_grid = self.rebinning[1]
                    for index in range(1440):
                        for element in chid_to_pixel[index]:
                            coords = pixel_index_to_grid[element[0]]
                            for value in event_photons[index]:
                                input_matrix[coords[0]][coords[1]][value-30] += element[1]*1
                    data.append([np.fliplr(np.rot90(input_matrix, 3)), energy, zd_deg, az_deg, act_phi, act_theta])
                yield data

            except Exception as e:
                print(str(e))

    def single_processor(self):
        return NotImplemented

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
            dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape_pic, chunks=pic.shape, dtype=pic.dtype)
            maxshape_run = (None,) + zd_deg.shape[1:]
            dset_run = hdf.create_dataset('Energy', shape=energy.shape, maxshape=maxshape_run, chunks=energy.shape, dtype=energy.dtype)
            maxshape_event = (None,) + zd_deg.shape[1:]
            dset_zd_deg = hdf.create_dataset('Zd_deg', shape=zd_deg.shape, maxshape=maxshape_event, chunks=zd_deg.shape, dtype=zd_deg.dtype)
            maxshape_az_deg = (None,) + zd_deg.shape[1:]
            dset_az_deg = hdf.create_dataset('Az_deg', shape=az_deg.shape, maxshape=maxshape_az_deg, chunks=az_deg.shape, dtype=az_deg.dtype)
            maxshape_phia = (None,) + zd_deg.shape[1:]
            dset_phia = hdf.create_dataset('Phi', shape=act_phi.shape, maxshape=maxshape_phia, chunks=act_phi.shape, dtype=act_phi.dtype)
            maxshape_thetaa = (None,) + zd_deg.shape[1:]
            dset_thetaa = hdf.create_dataset('Theta', shape=act_theta.shape, maxshape=maxshape_thetaa, chunks=act_theta.shape, dtype=act_theta.dtype)

            dset_pic[:] = pic
            dset_run[:] = energy
            dset_zd_deg[:] = zd_deg
            dset_az_deg[:] = az_deg
            dset_phia[:] = act_phi
            dset_thetaa[:] = act_theta

            for batch in gen:
                pic, energy, zd_deg, az_deg, phi, theta = self.format(batch)

                dset_pic.resize(row_count + theta.shape[0], axis=0)
                dset_run.resize(row_count + theta.shape[0], axis=0)
                dset_zd_deg.resize(row_count + theta.shape[0], axis=0)
                dset_az_deg.resize(row_count + theta.shape[0], axis=0)
                dset_phia.resize(row_count + theta.shape[0], axis=0)
                dset_thetaa.resize(row_count + theta.shape[0], axis=0)

                dset_pic[row_count:] = pic
                dset_run[row_count:] = energy
                dset_zd_deg[row_count:] = zd_deg
                dset_az_deg[row_count:] = az_deg
                dset_phia[row_count:] = act_phi
                dset_thetaa[row_count:] = act_theta

                row_count += phi.shape[0]


class GammaPreprocessor(BasePreprocessor):

    def init(self):
        self.dl2_file = read_h5py(self.dl2_file, key="events", columns=["event_num", "@source", "source_position", "cog_x", "cog_y", "delta",
                                                                        "az_source_calc", "zd_source_calc",
                                                                        "az_tracking", "zd_tracking",
                                                                        "corsika_evt_header_total_energy", "corsika_evt_header_az"])

    def batch_processor(self):

        for index, file in enumerate(self.paths):
            mc_truth = file.split(".phs")[0] + ".ch.gz"
            print(mc_truth)
            try:
                sim_reader = ps.SimulationReader(
                    photon_stream_path=file,
                    mmcs_corsika_path=mc_truth
                )
                data = []
                for event in sim_reader:
                    df_event = self.dl2_file.loc[(np.isclose(self.dl2_file['corsika_evt_header_total_energy'],
                                                             event.simulation_truth.air_shower.energy)) &
                                                 (self.dl2_file['run_id'] == event.simulation_truth.run)]
                    if not df_event.empty:
                        # In the event chosen from the file
                        # Each event is the same as each line below
                        source_pos_x = df_event['cog_x'].values[0]
                        source_pos_y = df_event['cog_y'].values[0]
                        energy = df_event['source_position_0'].values[0]
                        event_photons = event.photon_stream.list_of_lists
                        zd_deg = event.zd
                        az_deg = event.az
                        act_phi = df_event['delta'].values[0]
                        act_theta = df_event['source_position_1'].values[0]
                        sky_source_az = event.simulation_truth.air_shower.energy
                        sky_source_zd = df_event['zd_source_calc'].values[0]
                        zd_deg1 = df_event['az_tracking'].values[0]
                        az_deg1 = df_event['zd_tracking'].values[0]
                        input_matrix = np.zeros([self.shape[1],self.shape[2],self.shape[3]])
                        chid_to_pixel = self.rebinning[0]
                        pixel_index_to_grid = self.rebinning[1]
                        for index in range(1440):
                            for element in chid_to_pixel[index]:
                                # Now get the first 60 event photons
                                coords = pixel_index_to_grid[element[0]]
                                for value in event_photons[index]:
                                    input_matrix[coords[0]][coords[1]][value-30] += element[1]*1

                        data.append([np.fliplr(np.rot90(input_matrix, 3)), energy, zd_deg, az_deg, source_pos_x,
                                     source_pos_y, act_phi, act_theta, sky_source_zd, sky_source_az, zd_deg1, az_deg1])
                yield data

            except Exception as e:
                print(str(e))

    def single_processor(self):
        return NotImplemented

    def format(self, batch):
        pic, run, event, zd_deg, source_position_x, source_pos_y, act_phi, act_theta, sky_source_zd, sky_source_az, zd_deg1, az_deg1 = zip(*batch)
        pic = self.reformat(np.array(pic))
        run = np.array(run)
        event = np.array(event)
        zd_deg = np.array(zd_deg)
        source_position_x = np.array(source_position_x)
        source_pos_y = np.array(source_pos_y)
        act_phi = np.array(act_phi)
        act_theta = np.array(act_theta)
        sky_source_zd = np.array(sky_source_zd)
        sky_source_az = np.array(sky_source_az)
        zd_deg1 = np.array(zd_deg1)
        az_deg1 = np.array(az_deg1)
        return pic, run, event, zd_deg, source_position_x, source_pos_y, act_phi, act_theta, sky_source_zd, \
               sky_source_az, zd_deg1, az_deg1

    def create_dataset(self):
        gen = self.batch_processor()
        batch = next(gen)
        pic, energy, zd_deg, az_deg, phi, theta, act_phi, act_theta, zd, az, zd_deg1, az_deg1 = self.format(batch)
        row_count = az_deg.shape[0]
        print(row_count)

        with h5py.File(self.output_file, 'w') as hdf:
            maxshape_pic = (None,) + pic.shape[1:]
            dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape_pic, chunks=pic.shape, dtype=pic.dtype)
            maxshape_run = (None,) + zd_deg.shape[1:]
            dset_run = hdf.create_dataset('Source_X', shape=energy.shape, maxshape=maxshape_run, chunks=energy.shape, dtype=energy.dtype)
            maxshape_event = (None,) + zd_deg.shape[1:]
            dset_zd_deg = hdf.create_dataset('Zd_deg', shape=zd_deg.shape, maxshape=maxshape_event, chunks=zd_deg.shape, dtype=zd_deg.dtype)
            maxshape_az_deg = (None,) + zd_deg.shape[1:]
            dset_az_deg = hdf.create_dataset('Az_deg', shape=az_deg.shape, maxshape=maxshape_az_deg, chunks=az_deg.shape, dtype=az_deg.dtype)
            maxshape_phi = (None,) + zd_deg.shape[1:]
            dset_phi = hdf.create_dataset('COG_X', shape=phi.shape, maxshape=maxshape_phi, chunks=phi.shape, dtype=phi.dtype)
            maxshape_theta = (None,) + zd_deg.shape[1:]
            dset_theta = hdf.create_dataset('COG_Y', shape=theta.shape, maxshape=maxshape_theta, chunks=theta.shape, dtype=theta.dtype)
            maxshape_phia = (None,) + zd_deg.shape[1:]
            dset_phia = hdf.create_dataset('Delta', shape=act_phi.shape, maxshape=maxshape_phia, chunks=act_phi.shape, dtype=act_phi.dtype)
            maxshape_thetaa = (None,) + zd_deg.shape[1:]
            dset_thetaa = hdf.create_dataset('Source_Y', shape=act_theta.shape, maxshape=maxshape_thetaa, chunks=act_theta.shape, dtype=act_theta.dtype)
            maxshape_zd = (None,) + zd_deg.shape[1:]
            dset_zd = hdf.create_dataset('Source_Zd', shape=zd_deg.shape, maxshape=maxshape_zd, chunks=zd.shape, dtype=zd.dtype)
            maxshape_az = (None,) + zd_deg.shape[1:]
            dset_az = hdf.create_dataset('Energy', shape=az_deg.shape, maxshape=maxshape_az, chunks=az.shape, dtype=az.dtype)
            maxshape_event1 = (None,) + zd_deg.shape[1:]
            dset_zd_deg1 = hdf.create_dataset('Pointing_Zd', shape=zd_deg1.shape, maxshape=maxshape_event1, chunks=zd_deg1.shape, dtype=zd_deg1.dtype)
            maxshape_az_deg1 = (None,) + zd_deg.shape[1:]
            dset_az_deg1 = hdf.create_dataset('Pointing_Az', shape=az_deg1.shape, maxshape=maxshape_az_deg1, chunks=az_deg1.shape, dtype=az_deg1.dtype)

            dset_pic[:] = pic
            dset_run[:] = energy
            dset_phi[:] = phi
            dset_zd_deg[:] = zd_deg
            dset_az_deg[:] = az_deg
            dset_theta[:] = theta
            dset_phia[:] = act_phi
            dset_thetaa[:] = act_theta
            dset_zd[:] = zd
            dset_az[:] = az
            dset_zd_deg1[:] = zd_deg1
            dset_az_deg1[:] = az_deg1

            for batch in gen:
                pic, energy, zd_deg, az_deg, phi, theta, act_phi, act_theta, zd, az, zd_deg1, az_deg1 = self.format(batch)

                dset_pic.resize(row_count + theta.shape[0], axis=0)
                dset_run.resize(row_count + theta.shape[0], axis=0)
                dset_phi.resize(row_count + theta.shape[0], axis=0)
                dset_theta.resize(row_count + theta.shape[0], axis=0)
                dset_zd_deg.resize(row_count + theta.shape[0], axis=0)
                dset_az_deg.resize(row_count + theta.shape[0], axis=0)
                dset_phia.resize(row_count + theta.shape[0], axis=0)
                dset_thetaa.resize(row_count + theta.shape[0], axis=0)
                dset_zd.resize(row_count + theta.shape[0], axis=0)
                dset_az.resize(row_count + theta.shape[0], axis=0)
                dset_zd_deg1.resize(row_count + theta.shape[0], axis=0)
                dset_az_deg1.resize(row_count + theta.shape[0], axis=0)

                dset_pic[row_count:] = pic
                dset_run[row_count:] = energy
                dset_phi[row_count:] = phi
                dset_theta[row_count:] = theta
                dset_zd_deg[row_count:] = zd_deg
                dset_az_deg[row_count:] = az_deg
                dset_phia[row_count:] = act_phi
                dset_thetaa[row_count:] = act_theta
                dset_zd[row_count:] = zd
                dset_az[row_count:] = az
                dset_zd_deg1[row_count:] = zd_deg1
                dset_az_deg1[row_count:] = az_deg1

                row_count += phi.shape[0]