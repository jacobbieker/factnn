import numpy as np
import h5py
import photon_stream as ps
from fact.io import read_h5py
import datetime

from factnn.preprocess.base_preprocessor import BasePreprocessor


class ObservationPreprocessor(BasePreprocessor):

    def init(self):
        self.dl2_file = read_h5py(self.dl2_file, key="events", columns=["event_num", "run_id", "night",
                                                                        "source_position_az", "source_position_zd",
                                                                        "source_position_x", "source_position_y",
                                                                        "cog_x", "cog_y",
                                                                        "timestamp", "pointing_position_az",
                                                                        "pointing_position_zd"])

    def batch_processor(self):
        self.init()
        for index, file in enumerate(self.paths):
            print(file)
            try:
                sim_reader = ps.EventListReader(file)
                data = []
                for event in sim_reader:
                    df_event = self.dl2_file.loc[(self.dl2_file['event_num'] == event.observation_info.event) &
                                                 (self.dl2_file['night'] == event.observation_info.night) &
                                                 (self.dl2_file['run_id'] == event.observation_info.run)]
                    if not df_event.empty:
                        # In the event chosen from the file
                        # Each event is the same as each line below
                        source_pos_x = df_event['source_position_x'].values[0]
                        source_pos_y = df_event['source_position_y'].values[0]
                        energy = df_event['timestamp'].values[0].astype(datetime.datetime)
                        event_photons = event.photon_stream.list_of_lists
                        zd_deg = event.zd
                        az_deg = event.az
                        cog_x = df_event['cog_x'].values[0]
                        cog_y = df_event['cog_y'].values[0]
                        sky_source_az = df_event['source_position_az'].values[0]
                        sky_source_zd = df_event['source_position_zd'].values[0]
                        zd_deg1 = df_event['pointing_position_zd'].values[0]
                        az_deg1 = df_event['pointing_position_az'].values[0]
                        event_num = event.observation_info.event
                        night = event.observation_info.night
                        run = event.observation_info.run
                        input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
                        chid_to_pixel = self.rebinning[0]
                        pixel_index_to_grid = self.rebinning[1]
                        for index in range(1440):
                            for element in chid_to_pixel[index]:
                                coords = pixel_index_to_grid[element[0]]
                                for value in event_photons[index]:
                                    if self.end > value > self.start:
                                        input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1

                        data.append([np.fliplr(np.rot90(input_matrix, 3)), energy, zd_deg, az_deg, source_pos_x,
                                     source_pos_y, sky_source_zd, sky_source_az, zd_deg1, az_deg1,
                                     event_num, night, run, cog_x, cog_y])
                yield data

            except Exception as e:
                print(str(e))

    def single_processor(self):
        for index, file in enumerate(self.paths):
            print(file)
            try:
                sim_reader = ps.EventListReader(file)
                for event in sim_reader:
                    data = []
                    df_event = self.dl2_file.loc[(self.dl2_file['event_num'] == event.observation_info.event) &
                                                 (self.dl2_file['night'] == event.observation_info.night) &
                                                 (self.dl2_file['run_id'] == event.observation_info.run)]
                    if not df_event.empty:
                        # In the event chosen from the file
                        # Each event is the same as each line below
                        source_pos_x = df_event['source_position_x'].values[0]
                        source_pos_y = df_event['source_position_y'].values[0]
                        energy = df_event['timestamp'].values[0].astype(datetime.datetime)
                        event_photons = event.photon_stream.list_of_lists
                        zd_deg = event.zd
                        az_deg = event.az
                        cog_x = df_event['cog_x'].values[0]
                        cog_y = df_event['cog_y'].values[0]
                        sky_source_az = df_event['source_position_az'].values[0]
                        sky_source_zd = df_event['source_position_zd'].values[0]
                        zd_deg1 = df_event['pointing_position_zd'].values[0]
                        az_deg1 = df_event['pointing_position_az'].values[0]
                        event_num = event.observation_info.event
                        night = event.observation_info.night
                        run = event.observation_info.run
                        input_matrix = np.zeros([self.shape[1], self.shape[2], self.shape[3]])
                        chid_to_pixel = self.rebinning[0]
                        pixel_index_to_grid = self.rebinning[1]
                        for index in range(1440):
                            for element in chid_to_pixel[index]:
                                coords = pixel_index_to_grid[element[0]]
                                for value in event_photons[index]:
                                    if self.end > value > self.start:
                                        input_matrix[coords[0]][coords[1]][value - self.start] += element[1] * 1

                        data.append([np.fliplr(np.rot90(input_matrix, 3)), energy, zd_deg, az_deg, source_pos_x,
                                     source_pos_y, sky_source_zd, sky_source_az, zd_deg1, az_deg1,
                                     event_num, night, run, cog_x, cog_y])
                    # need to do the format thing here, and add auxiliary structure
                    yield self.format(data)

            except Exception as e:
                print(str(e))

    def format(self, batch):
        pic, energy, zd_deg, az_deg, source_pos_x, source_pos_y, sky_source_zd, sky_source_az, zd_deg1, az_deg1, \
        event_num, night, run, cog_x, cog_y = zip(*batch)
        pic = self.reformat(np.array(pic))
        energy = np.array(energy)
        zd_deg = np.array(zd_deg)
        az_deg = np.array(az_deg)
        source_pos_x = np.array(source_pos_x)
        source_pos_y = np.array(source_pos_y)
        sky_source_zd = np.array(sky_source_zd)
        sky_source_az = np.array(sky_source_az)
        zd_deg1 = np.array(zd_deg1)
        az_deg1 = np.array(az_deg1)
        event_num = np.array(event_num)
        night = np.array(night)
        run = np.array(run)
        cog_x = np.array(cog_x)
        cog_y = np.array(cog_y)
        return pic, energy, zd_deg, az_deg, source_pos_x, source_pos_y, sky_source_zd, sky_source_az, zd_deg1, az_deg1, \
               event_num, night, run, cog_x, cog_y

    def create_dataset(self):
        gen = self.batch_processor()
        batch = next(gen)
        pic, energy, zd_deg, az_deg, source_pos_x, source_pos_y, sky_source_zd, sky_source_az, zd_deg1, az_deg1, \
        event_num, night, run, cog_x, cog_y = self.format(batch)
        row_count = az_deg.shape[0]
        print(row_count)

        with h5py.File(self.output_file, 'w') as hdf:
            maxshape_pic = (None,) + pic.shape[1:]
            dset_pic = hdf.create_dataset('Image', shape=pic.shape, maxshape=maxshape_pic, chunks=pic.shape,
                                          dtype=pic.dtype)
            maxshape_run = (None,) + zd_deg.shape[1:]
            dset_time = hdf.create_dataset('Time', shape=energy.shape, maxshape=maxshape_run, chunks=energy.shape,
                                           dtype=energy.dtype)
            dset_run = hdf.create_dataset('Run', shape=run.shape, maxshape=maxshape_run, chunks=energy.shape,
                                          dtype=run.dtype)
            dset_night = hdf.create_dataset('Night', shape=night.shape, maxshape=maxshape_run, chunks=night.shape,
                                            dtype=night.dtype)
            dset_eventnum = hdf.create_dataset('EventNum', shape=event_num.shape, maxshape=maxshape_run,
                                               chunks=event_num.shape,
                                               dtype=event_num.dtype)
            dset_cogy = hdf.create_dataset('Cog_X', shape=cog_x.shape, maxshape=maxshape_run, chunks=cog_x.shape,
                                           dtype=cog_x.dtype)
            dset_cogx = hdf.create_dataset('Cog_Y', shape=cog_x.shape, maxshape=maxshape_run, chunks=cog_x.shape,
                                           dtype=cog_x.dtype)
            maxshape_event = (None,) + zd_deg.shape[1:]
            dset_zd_deg = hdf.create_dataset('Zd_deg', shape=zd_deg.shape, maxshape=maxshape_event, chunks=zd_deg.shape,
                                             dtype=zd_deg.dtype)
            maxshape_az_deg = (None,) + zd_deg.shape[1:]
            dset_az_deg = hdf.create_dataset('Az_deg', shape=az_deg.shape, maxshape=maxshape_az_deg,
                                             chunks=az_deg.shape, dtype=az_deg.dtype)
            maxshape_phi = (None,) + zd_deg.shape[1:]
            dset_phi = hdf.create_dataset('Source_X', shape=source_pos_x.shape, maxshape=maxshape_phi,
                                          chunks=source_pos_x.shape,
                                          dtype=source_pos_x.dtype)
            maxshape_source_pos_y = (None,) + zd_deg.shape[1:]
            dset_source_pos_y = hdf.create_dataset('Source_Y', shape=source_pos_x.shape, maxshape=maxshape_source_pos_y,
                                                   chunks=source_pos_x.shape,
                                                   dtype=source_pos_x.dtype)
            maxshape_zd = (None,) + zd_deg.shape[1:]
            dset_zd = hdf.create_dataset('Source_Zd', shape=zd_deg.shape, maxshape=maxshape_zd,
                                         chunks=sky_source_zd.shape,
                                         dtype=sky_source_zd.dtype)
            maxshape_az = (None,) + zd_deg.shape[1:]
            dset_az = hdf.create_dataset('Source_Az', shape=az_deg.shape, maxshape=maxshape_az,
                                         chunks=sky_source_zd.shape,
                                         dtype=sky_source_zd.dtype)
            maxshape_event1 = (None,) + zd_deg.shape[1:]
            dset_zd_deg1 = hdf.create_dataset('Pointing_Zd', shape=zd_deg1.shape, maxshape=maxshape_event1,
                                              chunks=zd_deg1.shape, dtype=zd_deg1.dtype)
            maxshape_az_deg1 = (None,) + zd_deg.shape[1:]
            dset_az_deg1 = hdf.create_dataset('Pointing_Az', shape=az_deg1.shape, maxshape=maxshape_az_deg1,
                                              chunks=az_deg1.shape, dtype=az_deg1.dtype)

            dset_pic[:] = pic
            dset_run[:] = run
            dset_night[:] = night
            dset_eventnum[:] = event_num
            dset_cogy[:] = cog_y
            dset_cogx[:] = cog_x
            dset_time[:] = energy
            dset_zd_deg[:] = zd_deg
            dset_az_deg[:] = az_deg
            dset_source_pos_y[:] = source_pos_y
            dset_zd[:] = sky_source_zd
            dset_az[:] = sky_source_az
            dset_zd_deg1[:] = zd_deg1
            dset_az_deg1[:] = az_deg1

            for batch in gen:
                pic, energy, zd_deg, az_deg, source_pos_x, source_pos_y, sky_source_zd, sky_source_az, zd_deg1, az_deg1, \
                event_num, night, run, cog_x, cog_y = self.format(batch)

                dset_pic.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_run.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_night.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_eventnum.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_cogy.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_cogx.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_time.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_zd_deg.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_az_deg.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_source_pos_y.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_zd.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_az.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_zd_deg1.resize(row_count + source_pos_y.shape[0], axis=0)
                dset_az_deg1.resize(row_count + source_pos_y.shape[0], axis=0)

                dset_pic[row_count:] = pic
                dset_run[row_count:] = run
                dset_night[row_count:] = night
                dset_eventnum[row_count:] = event_num
                dset_cogy[row_count:] = cog_y
                dset_cogx[row_count:] = cog_x
                dset_time[row_count:] = energy
                dset_zd_deg[row_count:] = zd_deg
                dset_az_deg[row_count:] = az_deg
                dset_source_pos_y[row_count:] = source_pos_y
                dset_zd[row_count:] = sky_source_zd
                dset_az[row_count:] = sky_source_az
                dset_zd_deg1[row_count:] = zd_deg1
                dset_az_deg1[row_count:] = az_deg1

                row_count += run.shape[0]
