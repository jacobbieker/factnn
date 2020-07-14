import numpy as np
import h5py
import photon_stream as ps
from fact.io import read_h5py
import datetime
import os
import pickle

from factnn.data.preprocess.base_preprocessor import BasePreprocessor


class ObservationPreprocessor(BasePreprocessor):
    def init(self):
        self.dl2_file = read_h5py(
            self.dl2_file,
            key="events",
            columns=[
                "event_num",
                "run_id",
                "night",
                "source_position_az",
                "source_position_zd",
                "source_position_x",
                "source_position_y",
                "cog_x",
                "cog_y",
                "timestamp",
                "pointing_position_az",
                "pointing_position_zd",
            ],
        )

    def event_processor(
        self, directory, clean_images=False, only_core=True, clump_size=20
    ):
        for index, file in enumerate(self.paths):
            file_name = file.split("/")[-1].split(".phs")[0]
            try:
                sim_reader = ps.EventListReader(file)
                counter = 0
                for event in sim_reader:
                    df_event = self.dl2_file.loc[
                        (self.dl2_file["event_num"] == event.observation_info.event)
                        & (self.dl2_file["night"] == event.observation_info.night)
                        & (self.dl2_file["run_id"] == event.observation_info.run)
                    ]
                    if not df_event.empty:
                        counter += 1

                        if os.path.isfile(
                            os.path.join(
                                directory,
                                "clump" + str(clump_size),
                                str(file_name) + "_" + str(counter),
                            )
                        ) and os.path.isfile(
                            os.path.join(
                                directory,
                                "core" + str(clump_size),
                                str(file_name) + "_" + str(counter),
                            )
                        ):
                            print("True: " + str(file_name) + "_" + str(counter))
                            continue

                        if clean_images:
                            # Do it for no clumps, all clump, and only core into different subfolders
                            all_photons, clump_photons, core_photons = self.clean_image(
                                event, only_core=only_core, min_samples=clump_size
                            )
                            if core_photons is None:
                                print("No Clumps, skip")
                                continue
                            else:
                                for key, photon_set in {
                                    "no_clean": all_photons,
                                    "clump": clump_photons,
                                    "core": core_photons,
                                }.items():
                                    event.photon_stream.raw = photon_set
                                    # In the event chosen from the file
                                    # Each event is the same as each line below
                                    source_pos_x = df_event["source_position_x"].values[
                                        0
                                    ]
                                    source_pos_y = df_event["source_position_y"].values[
                                        0
                                    ]
                                    timestamp = (
                                        df_event["timestamp"]
                                        .values[0]
                                        .astype(datetime.datetime)
                                    )
                                    event_photons = event.photon_stream.list_of_lists
                                    zd_deg = event.zd
                                    az_deg = event.az
                                    cog_x = df_event["cog_x"].values[0]
                                    cog_y = df_event["cog_y"].values[0]
                                    sky_source_az = df_event[
                                        "source_position_az"
                                    ].values[0]
                                    sky_source_zd = df_event[
                                        "source_position_zd"
                                    ].values[0]
                                    zd_deg1 = df_event["pointing_position_zd"].values[0]
                                    az_deg1 = df_event["pointing_position_az"].values[0]
                                    event_num = event.observation_info.event
                                    night = event.observation_info.night
                                    run = event.observation_info.run
                                    data_dict = [
                                        [
                                            event_photons,
                                            timestamp,
                                            zd_deg,
                                            az_deg,
                                            cog_x,
                                            cog_y,
                                            sky_source_az,
                                            sky_source_zd,
                                            zd_deg1,
                                            az_deg1,
                                            source_pos_x,
                                            source_pos_y,
                                            event_num,
                                            night,
                                            run,
                                        ],
                                        {
                                            "Image": 0,
                                            "Timestamp": 1,
                                            "Zd_Deg": 2,
                                            "Az_Deg": 3,
                                            "COG_X": 4,
                                            "COG_Y": 5,
                                            "Source_Position_Az": 6,
                                            "Source_Position_Zd": 7,
                                            "Pointing_Position_Zd": 8,
                                            "Pointing_Position_Az": 9,
                                            "Source_Position_X": 10,
                                            "Source_Position_Y": 11,
                                            "Event_Number": 12,
                                            "Night": 13,
                                            "Run": 14,
                                        },
                                    ]
                                    if key != "no_clean":
                                        with open(
                                            os.path.join(
                                                directory,
                                                key + str(clump_size),
                                                str(file_name) + "_" + str(counter),
                                            ),
                                            "wb",
                                        ) as event_file:
                                            pickle.dump(data_dict, event_file)
                                    else:
                                        if not os.path.isfile(
                                            os.path.join(
                                                directory,
                                                key,
                                                str(file_name) + "_" + str(counter),
                                            )
                                        ):
                                            with open(
                                                os.path.join(
                                                    directory,
                                                    key,
                                                    str(file_name) + "_" + str(counter),
                                                ),
                                                "wb",
                                            ) as event_file:
                                                pickle.dump(data_dict, event_file)
                        else:
                            # In the event chosen from the file
                            # Each event is the same as each line below
                            # Each event is the same as each line below
                            source_pos_x = df_event["source_position_x"].values[0]
                            source_pos_y = df_event["source_position_y"].values[0]
                            timestamp = (
                                df_event["timestamp"]
                                .values[0]
                                .astype(datetime.datetime)
                            )
                            event_photons = event.photon_stream.list_of_lists
                            zd_deg = event.zd
                            az_deg = event.az
                            cog_x = df_event["cog_x"].values[0]
                            cog_y = df_event["cog_y"].values[0]
                            sky_source_az = df_event["source_position_az"].values[0]
                            sky_source_zd = df_event["source_position_zd"].values[0]
                            zd_deg1 = df_event["pointing_position_zd"].values[0]
                            az_deg1 = df_event["pointing_position_az"].values[0]
                            event_num = event.observation_info.event
                            night = event.observation_info.night
                            run = event.observation_info.run
                            data_dict = [
                                [
                                    event_photons,
                                    timestamp,
                                    zd_deg,
                                    az_deg,
                                    cog_x,
                                    cog_y,
                                    sky_source_az,
                                    sky_source_zd,
                                    zd_deg1,
                                    az_deg1,
                                    source_pos_x,
                                    source_pos_y,
                                    event_num,
                                    night,
                                    run,
                                ],
                                {
                                    "Image": 0,
                                    "Timestamp": 1,
                                    "Zd_Deg": 2,
                                    "Az_Deg": 3,
                                    "COG_X": 4,
                                    "COG_Y": 5,
                                    "Source_Position_Az": 6,
                                    "Source_Position_Zd": 7,
                                    "Pointing_Position_Zd": 8,
                                    "Pointing_Position_Az": 9,
                                    "Source_Position_X": 10,
                                    "Source_Position_Y": 11,
                                    "Event_Number": 12,
                                    "Night": 13,
                                    "Run": 14,
                                },
                            ]
                            with open(
                                os.path.join(
                                    directory, str(file_name) + "_" + str(counter)
                                ),
                                "wb",
                            ) as event_file:
                                pickle.dump(data_dict, event_file)
            except Exception as e:
                print(str(e))
                pass

    def batch_processor(self, clean_images=False):
        self.init()
        for index, file in enumerate(self.paths):
            print(file)
            try:
                sim_reader = ps.EventListReader(file)
                data = []
                for event in sim_reader:
                    df_event = self.dl2_file.loc[
                        (self.dl2_file["event_num"] == event.observation_info.event)
                        & (self.dl2_file["night"] == event.observation_info.night)
                        & (self.dl2_file["run_id"] == event.observation_info.run)
                    ]
                    if not df_event.empty:
                        if clean_images:
                            event = self.clean_image(event)
                        # In the event chosen from the file
                        # Each event is the same as each line below
                        source_pos_x = df_event["source_position_x"].values[0]
                        source_pos_y = df_event["source_position_y"].values[0]
                        energy = (
                            df_event["timestamp"].values[0].astype(datetime.datetime)
                        )
                        event_photons = event.photon_stream.list_of_lists
                        zd_deg = event.zd
                        az_deg = event.az
                        cog_x = df_event["cog_x"].values[0]
                        cog_y = df_event["cog_y"].values[0]
                        sky_source_az = df_event["source_position_az"].values[0]
                        sky_source_zd = df_event["source_position_zd"].values[0]
                        zd_deg1 = df_event["pointing_position_zd"].values[0]
                        az_deg1 = df_event["pointing_position_az"].values[0]
                        event_num = event.observation_info.event
                        night = event.observation_info.night
                        run = event.observation_info.run
                        input_matrix = np.zeros(
                            [self.shape[1], self.shape[2], self.shape[3]]
                        )
                        chid_to_pixel = self.rebinning[0]
                        pixel_index_to_grid = self.rebinning[1]
                        for index in range(1440):
                            for element in chid_to_pixel[index]:
                                coords = pixel_index_to_grid[element[0]]
                                for value in event_photons[index]:
                                    if self.end >= value >= self.start:
                                        input_matrix[coords[0]][coords[1]][
                                            value - self.start
                                        ] += (element[1] * 1)

                        data.append(
                            [
                                np.fliplr(np.rot90(input_matrix, 3)),
                                energy,
                                zd_deg,
                                az_deg,
                                source_pos_x,
                                source_pos_y,
                                sky_source_zd,
                                sky_source_az,
                                zd_deg1,
                                az_deg1,
                                event_num,
                                night,
                                run,
                                cog_x,
                                cog_y,
                            ]
                        )
                yield data

            except Exception as e:
                print(str(e))

    def single_processor(
        self,
        normalize=False,
        collapse_time=False,
        final_slices=5,
        as_channels=False,
        clean_images=False,
    ):
        while True:
            print("New Crab")
            for index, file in enumerate(self.paths):
                try:
                    sim_reader = ps.EventListReader(file)
                    for event in sim_reader:
                        data = []
                        df_event = self.dl2_file.loc[
                            (self.dl2_file["event_num"] == event.observation_info.event)
                            & (self.dl2_file["night"] == event.observation_info.night)
                            & (self.dl2_file["run_id"] == event.observation_info.run)
                        ]
                        if not df_event.empty:
                            if clean_images:
                                event = self.clean_image(event)
                            # In the event chosen from the file
                            # Each event is the same as each line below
                            source_pos_x = df_event["source_position_x"].values[0]
                            source_pos_y = df_event["source_position_y"].values[0]
                            energy = (
                                df_event["timestamp"]
                                .values[0]
                                .astype(datetime.datetime)
                            )
                            event_photons = event.photon_stream.list_of_lists
                            zd_deg = event.zd
                            az_deg = event.az
                            cog_x = df_event["cog_x"].values[0]
                            cog_y = df_event["cog_y"].values[0]
                            sky_source_az = df_event["source_position_az"].values[0]
                            sky_source_zd = df_event["source_position_zd"].values[0]
                            zd_deg1 = df_event["pointing_position_zd"].values[0]
                            az_deg1 = df_event["pointing_position_az"].values[0]
                            event_num = event.observation_info.event
                            night = event.observation_info.night
                            run = event.observation_info.run
                            input_matrix = np.zeros(
                                [self.shape[1], self.shape[2], self.shape[3]]
                            )
                            chid_to_pixel = self.rebinning[0]
                            pixel_index_to_grid = self.rebinning[1]
                            for index in range(1440):
                                for element in chid_to_pixel[index]:
                                    coords = pixel_index_to_grid[element[0]]
                                    for value in event_photons[index]:
                                        if self.end >= value >= self.start:
                                            input_matrix[coords[0]][coords[1]][
                                                value - self.start
                                            ] += (element[1] * 1)

                            data.append(
                                [
                                    np.fliplr(np.rot90(input_matrix, 3)),
                                    energy,
                                    zd_deg,
                                    az_deg,
                                    source_pos_x,
                                    source_pos_y,
                                    sky_source_zd,
                                    sky_source_az,
                                    zd_deg1,
                                    az_deg1,
                                    event_num,
                                    night,
                                    run,
                                    cog_x,
                                    cog_y,
                                ]
                            )
                            # need to do the format thing here, and add auxiliary structure
                            data_format = {
                                "Image": 0,
                                "Timestamp": 1,
                                "Zd_Deg": 2,
                                "Az_Deg": 3,
                                "Source_X": 4,
                                "Source_Y": 5,
                                "Theta": 6,
                            }
                            data = self.format(data)
                            if normalize:
                                data = list(data)
                                data[0] = self.normalize_image(data[0])
                                data = tuple(data)
                            if collapse_time:
                                data = list(data)
                                data[0] = self.collapse_image_time(
                                    data[0], final_slices, as_channels
                                )
                                data = tuple(data)
                            yield data, data_format

                except Exception as e:
                    print(str(e))

    def count_events(self):
        if self.num_events < 0:
            count = 0
            for index, file in enumerate(self.paths):
                try:
                    print("Trying...")
                    crab_reader = ps.EventListReader(file)
                    for event in crab_reader:
                        df_event = self.dl2_file.loc[
                            (self.dl2_file["event_num"] == event.observation_info.event)
                            & (self.dl2_file["night"] == event.observation_info.night)
                            & (self.dl2_file["run_id"] == event.observation_info.run)
                        ]
                        if not df_event.empty:
                            count += 1
                    print(count)
                except Exception as e:
                    print(str(e))
            print(count)
            print("\n")
            self.num_events = count
            return count
        else:
            return self.num_events

    def format(self, batch):
        (
            pic,
            energy,
            zd_deg,
            az_deg,
            source_pos_x,
            source_pos_y,
            sky_source_zd,
            sky_source_az,
            zd_deg1,
            az_deg1,
            event_num,
            night,
            run,
            cog_x,
            cog_y,
        ) = zip(*batch)
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
        return (
            pic,
            energy,
            zd_deg,
            az_deg,
            source_pos_x,
            source_pos_y,
            sky_source_zd,
            sky_source_az,
            zd_deg1,
            az_deg1,
            event_num,
            night,
            run,
            cog_x,
            cog_y,
        )
