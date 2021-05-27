import os.path as osp
import os
import pickle
import numpy as np
from zlib import crc32
import pkg_resources as res
from functools import partial


from multiprocessing import Pool, Manager

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data

from photon_stream.representations import (
    list_of_lists_to_raw_phs,
    raw_phs_to_point_cloud,
)
from photon_stream.geometry import GEOMETRY
import photon_stream as ps


from factnn.utils.augment import euclidean_distance, true_sign


def to_list(x):
    if not isinstance(x, (tuple, list)) or isinstance(x, str):
        x = [x]
    return x


class PhotonStreamDataset(Dataset):
    def __init__(
        self,
        root,
        split="trainval",
        include_proton=True,
        task="separation",
        simulated=True,
        transform=None,
        pre_transform=None,
    ):
        """

        Dataset for generating the events from the PhotonStream files, instead of the preprocessed files

        :param task: Either 'Separation', or 'Energy'
        :param split: Splits to include, either 'train', 'val', 'test', or 'trainval' or 'all' for training, validation, test, training and validation sets, or all data respectively
        :param root: Root directory for the dataset
        :param include_proton: Whether to include proton events or not
        :param simulated: Whether this is on simulated data or real data
        """
        self.processed_filenames = []
        self.task = task.lower()
        self.split = split.lower()
        self.include_proton = include_proton
        self.simulated = simulated
        super(PhotonStreamDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return NotImplementedError

    @property
    def processed_file_names(self):
        return self.processed_filenames

    def download(self):
        pass

    def process(self):

        used_paths = split_data(self.raw_paths)[self.split]

        for raw_path in used_paths:
            if not self.include_proton and "proton" in raw_path:
                continue
            # load the pickled file from the disk
            if osp.exists(osp.join(self.processed_dir, self.split, f"{raw_path}.pt")):
                self.processed_filenames.append(f"{raw_path}.pt")
            else:
                if self.simulated:
                    mc_truth = raw_path.split(".phs")[0] + ".ch.gz"
                    event_reader = ps.SimulationReader(
                        photon_stream_path=raw_path, mmcs_corsika_path=mc_truth
                    )
                else:
                    event_reader = ps.EventListReader(raw_path)
                for event in event_reader:
                    print(raw_path)
                    # Convert List of List to Point Cloud, then truncation is simply cutting in the z direction
                    point_cloud = np.asarray(event.photon_stream.point_cloud)
                    # Read data from `raw_path`.
                    data = Data(
                        pos=point_cloud
                    )  # Just need x,y,z ignore derived features
                    if self.simulated:
                        if "proton" in raw_path:
                            data.event_type = torch.tensor(0, dtype=torch.int8)
                        elif "gamma" in raw_path:
                            data.event_type = torch.tensor(1, dtype=torch.int8)
                        else:
                            print("No Event Type")
                            continue
                        data.energy = torch.tensor(
                            event.simulation_truth.air_shower.energy, dtype=torch.float
                        )
                        data.phi = torch.tensor(
                            event.simulation_truth.air_shower.phi, dtype=torch.float
                        )
                        data.theta = torch.tensor(
                            event.simulation_truth.air_shower.theta, dtype=torch.float
                        )
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    torch.save(
                        data,
                        osp.join(
                            self.processed_dir, self.split, "{}.pt".format(raw_path)
                        ),
                    )
                    self.processed_filenames.append("{}.pt".format(raw_path))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, self.split, self.processed_file_names[idx])
        )
        if self.simulated:
            if self.task == "energy":
                data.y = data.energy
            elif self.task == "phi":
                data.y = data.phi
            elif self.task == "theta":
                data.y = data.theta
            elif self.task == "separation":
                data.y = data.event_type
            else:
                print("Not recognized task type")
                return NotImplementedError
        return data


class EventDataset(Dataset):
    def __init__(
        self,
        root,
        split="trainval",
        include_proton=True,
        task="separation",
        cleanliness="no_clean",
        balanced_classes=False,
        fraction=1.0,
        transform=None,
        pre_transform=None,
    ):
        """
        :param task: Either 'separation', 'energy', 'phi', or 'theta'
        :param split: Splits to include, either 'train', 'val', 'test', or 'trainval' or 'all' for training, validation, test, training and validation sets, or all data respectively
        :param root: Root directory for the dataset
        :param balanced_classes: Whether to keep the number of gamma and proton events the same or not
        :param include_proton: Whether to include proton events or not
        :param fraction: Fraction of dataset to use, if not 1.0, then takes randomly the fraction of the dataset to use
        :param cleanliness: str, which version of the DBSCAN cleaned files to use, and which raw filenames to load, one of 'no_clean', 'clump5',
        'clump10', 'clump15', 'clump20', 'core5', 'core10', 'core15', 'core20'
        """
        self.task = task.lower()
        self.split = split.lower()
        self.include_proton = include_proton
        self.balanced_classes = balanced_classes
        self.cleanliness = cleanliness.strip().lower()
        self.fraction = fraction

        try:
            self.event_dict = pickle.load(
                open(
                    res.resource_filename(
                        "factnn.data.resources", f"{self.cleanliness}_raw_names.p"
                    ),
                    "rb",
                )
            )
        except:
            raise ValueError(
                "cleanliness value is not one of: 'no_clean', 'clump5','clump10', 'clump15', 'clump20', 'core5', 'core10', 'core15', 'core20'"
            )
        self.processed_filenames = (
            []
        )  # Because of multithreading, its faster than using file_exists in base class on actual list
        super(EventDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        if self.include_proton:
            return list(self.event_dict["proton"]) + list(self.event_dict["gamma"])
        else:
            return list(self.event_dict["gamma"])

    @property
    def processed_file_names(self):
        return self.processed_filenames

    def download(self):
        pass

    def process_file(
        self, is_proton, processed_list, raw_path
    ):
        """
        Processes a single file given the path
        :param is_proton: Whether the events are proton events or not
        :param raw_path: raw filename of the event to process
        :param processed_list: Threadsafe list to append filenames to
        :return:
        """
        # load the pickled file from the disk
        if osp.exists(osp.join(self.processed_dir, f"{raw_path}.pt")):
            processed_list.append(f"{raw_path}.pt")
        else:
            with open(osp.join(self.raw_dir, raw_path), "rb") as pickled_event:
                event_data, data_format, features, feature_cluster = pickle.load(
                    pickled_event
                )
                # Convert List of List to Point Cloud, then truncation is simply cutting in the z direction
                event_photons = event_data[data_format["Image"]]
                event_photons = list_of_lists_to_raw_phs(event_photons)
                point_cloud = np.asarray(
                    raw_phs_to_point_cloud(
                        event_photons, cx=GEOMETRY.x_angle, cy=GEOMETRY.y_angle
                    )
                )
                # Read data from `raw_path`.
                data = Data(
                    pos=torch.tensor(point_cloud, dtype=torch.float).squeeze(),
                )  # Just need x,y,z ignore derived features
                if is_proton:
                    data.event_type = torch.tensor([0], dtype=torch.long
                    )
                else:
                    data.event_type = torch.tensor([1], dtype=torch.long
                    )
                data.energy = torch.tensor(
                    [event_data[data_format["Energy"]]],
                    dtype=torch.long,
                )
                data.phi = torch.tensor(
                    [event_data[4]],
                    dtype=torch.long,  # Needed because most the proton events had the wrong data_format
                )
                data.theta = torch.tensor(
                    [event_data[5]],
                    dtype=torch.long,  # Needed because most the proton events had the wrong data_format
                )

                # Now add the features from the feature extraction
                if (
                    features["extraction"] == 1
                ):  # Failed extraction, so has no features to use
                    return
                else:
                    feature_list = []
                    feature_list.append(features["head_tail_ratio"])
                    feature_list.append(features["length"])
                    feature_list.append(features["width"])
                    feature_list.append(features["time_gradient"])
                    feature_list.append(features["number_photons"])
                    feature_list.append(
                        features["length"] * features["width"] * np.pi
                    )
                    feature_list.append(
                        (
                            (features["length"] * features["width"] * np.pi)
                            / np.log(features["number_photons"]) ** 2
                        )
                    )
                    feature_list.append(
                        (
                            features["number_photons"]
                            / (features["length"] * features["width"] * np.pi)
                        )
                    )
                # Now make it the node features
                data.features = torch.tensor(
                    np.asarray(feature_list),
                    dtype=torch.float,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    return

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(
                    data, osp.join(self.processed_dir, "{}.pt".format(raw_path)),
                )
                processed_list.append("{}.pt".format(raw_path))

    def process(self):
        used_paths = split_data(self.raw_file_names)[self.split]
        protons = np.intersect1d(
            used_paths, self.event_dict["proton"], assume_unique=True
        )
        gammas = np.intersect1d(
            used_paths, self.event_dict["gamma"], assume_unique=True
        )
        if self.balanced_classes and self.task == 'separation': # Only matters for separation task, all others only need gamma
            num_events = len(protons) if len(protons) < len(gammas) else len(gammas)
            if 0.0 < self.fraction < 1.0:
                num_events *= self.fraction
            protons = np.random.choice(protons, size=int(num_events), replace=False)
            gammas = np.random.choice(gammas, size=int(num_events), replace=False)
        elif 0.0 < self.fraction < 1.0:
            protons = np.random.choice(
                protons, size=int(self.fraction * len(protons)), replace=False
            )
            gammas = np.random.choice(
                gammas, size=int(self.fraction * len(gammas)), replace=False
            )
        manager = Manager()
        pool = Pool()
        threaded_filenames = manager.list()
        if self.task == 'separation':
            proton_proc = partial(self.process_file, True, threaded_filenames)
            processors = pool.map_async(proton_proc, protons)
            processors.wait()
        gamma_proc = partial(self.process_file, False, threaded_filenames)
        processors = pool.map_async(gamma_proc, gammas)
        processors.wait()
        for element in threaded_filenames:
            self.processed_filenames.append(element)

    def len(self):
        return len(self.processed_filenames)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        if self.task == "energy":
            del data.phi
            del data.theta
            del data.event_type
            del data.features
            data.y = torch.tensor(data.energy, dtype=torch.float)
            del data.energy
        elif self.task == "phi":
            del data.event_type
            del data.theta
            del data.energy
            del data.features
            data.y = data.phi
        elif self.task == "theta":
            del data.phi
            del data.event_type
            del data.energy
            del data.features
            data.y = data.theta
        elif self.task == "separation":
            del data.phi
            del data.theta
            data.y = torch.tensor(data.event_type, dtype=torch.long)
            del data.event_type
            del data.energy
            del data.features
        else:
            print("Not recognized task type")
            return NotImplementedError
        return data


class DiffuseDataset(Dataset):
    def __init__(
        self,
        root,
        split="trainval",
        cleanliness="no_clean",
        transform=None,
        pre_transform=None,
            fraction=1.0,
    ):
        """
        EventFile Dataloader for specifically Disp calculations,
        only using the diffuse gamma sources that have the extra information

        Use EventFileDataset for Energy and Separation tasks

        :param num_points: The number of points to have, either using points multiple times, or subselecting from the total points
        """
        self.processed_filenames = []
        self.split = split.lower()
        self.cleanliness = cleanliness.strip().lower()
        self.fraction = fraction
        try:
            self.event_list = pickle.load(
                open(
                    res.resource_filename(
                        "factnn.data.resources", f"{self.cleanliness}_diffuse_raw_names.p"
                    ),
                    "rb",
                )
            )
        except:
            raise ValueError(
                "cleanliness value is not one of: 'no_clean', 'clump20', 'core20'"
            )
        super(DiffuseDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.event_list

    @property
    def processed_file_names(self):
        return self.processed_filenames

    @property
    def raw_dir(self):
        return osp.join(self.root, 'diffuse_raw')

    def download(self):
        pass

    def process_file(self, processed_list, raw_path):
        """
        Processes a single file given the path
        :param raw_path:
        :return:
        """
        # load the pickled file from the disk
        if osp.exists(osp.join(self.processed_dir, f"diffuse_{raw_path}.pt")):
            processed_list.append(f"diffuse_{raw_path}.pt")
        else:
            # Checks that file is not 0
            with open(osp.join(self.raw_dir, raw_path), "rb") as pickled_event:
                event_data, data_format, features = pickle.load(
                    pickled_event
                )
                # Convert List of List to Point Cloud, then truncation is simply cutting in the z direction
                event_photons = event_data[data_format["Image"]]
                event_photons = list_of_lists_to_raw_phs(event_photons)
                point_cloud = np.asarray(
                    raw_phs_to_point_cloud(
                        event_photons, cx=GEOMETRY.x_angle, cy=GEOMETRY.y_angle
                    )
                )
                # Read data from `raw_path`.
                data = Data(
                    pos=torch.tensor(point_cloud, dtype=torch.float).squeeze(),
                )  # Just need x,y,z ignore derived features
                data.y = torch.tensor(
                    [true_sign(
                        event_data[data_format["Source_X"]],
                        event_data[data_format["Source_Y"]],
                        event_data[data_format["COG_X"]],
                        event_data[data_format["COG_Y"]],
                        event_data[data_format["Delta"]],
                    )
                    * euclidean_distance(
                        event_data[data_format["Source_X"]],
                        event_data[data_format["Source_Y"]],
                        event_data[data_format["COG_X"]],
                        event_data[data_format["COG_Y"]],
                    )],
                    dtype=torch.float,
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    return

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(
                    data,
                    osp.join(self.processed_dir, "diffuse_{}.pt".format(raw_path)),
                )
                processed_list.append("diffuse_{}.pt".format(raw_path))

    def process(self):
        used_paths = split_data(self.raw_file_names)[self.split]
        if 0.0 < self.fraction < 1.0:
            used_paths = np.random.choice(used_paths, size=int(self.fraction*len(used_paths)), replace=False)
        manager = Manager()
        threaded_filenames = manager.list()
        pool = Pool()
        func = partial(self.process_file, threaded_filenames)
        processors = pool.map_async(func, used_paths)
        processors.wait()
        for element in threaded_filenames:
            self.processed_filenames.append(element)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, self.processed_file_names[idx])
        )
        return data


class ClusterDataset(Dataset):
    def __init__(
        self,
        root,
        uncleaned_root,
        clump_root=None,
        split="trainval",
        cleanliness="core20",
        transform=None,
        pre_transform=None,
    ):
        """

        Dataset for working on clustering datapoints, such as a replacement for DBSCAN algorithm.
        :param uncleaned_root: Root of files, with same names, that do not have the DBSCAN cleaned output
        :param split: Splits to include, either 'train', 'val', 'test', or 'trainval' for training, validation, test, or training and validation sets
        :param root: Root directory for the dataset, holding the files with the "cleaned" files
        :param clump_root: Root for files that hold the non-core clump outputs from DBSCAN, optional
        :param cleanliness: name of DBSCAN output eventfiles for the files in root, one of 'no_clean', 'clump5','clump10', 'clump15', 'clump20', 'core5', 'core10', 'core15', 'core20'
        """
        self.split = split.lower()
        self.uncleaned_root = uncleaned_root
        self.clump_root = clump_root
        self.cleanliness = cleanliness
        if self.clump_root is not None:
            self.clumps = True
        else:
            self.clumps = False
        try:
            self.event_dict = pickle.load(  # Only need one
                open(
                    res.resource_filename(
                        "factnn.data.resources", f"{self.cleanliness}_raw_names.p"
                    ),
                    "rb",
                )
            )
        except:
            raise ValueError(
                "cleanliness value is not one of: 'no_clean', 'clump5','clump10', 'clump15', 'clump20', 'core5', 'core10', 'core15', 'core20'"
            )
        self.processed_filenames = [
            f"cluster_{event}.pt" for event in self.raw_file_names
        ]
        super(ClusterDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return list(self.event_dict["proton"]) + list(self.event_dict["gamma"])

    @property
    def processed_file_names(self):
        return self.processed_filenames

    @property
    def processed_dir(self):
        return osp.join(self.root, "cluster")  # To not overlap with the non-clustering

    def download(self):
        pass

    def process_file(self, base_path):
        """
        Process single cluster file
        :param base_path:
        :return:
        """
        raw_path = osp.join(self.raw_dir, base_path)
        # Assumes that the folder structure follows the default convention of 'raw'
        uncleaned_path = osp.join(self.uncleaned_root, "raw", base_path)
        clump_path = osp.join(self.clump_root, "raw", base_path)
        # load the pickled file from the disk
        if osp.exists(
            osp.join(self.processed_dir, f"cluster_{base_path}.pt")
        ):
            self.processed_filenames.append(f"cluster_{base_path}.pt")
        else:
            try:
                with open(raw_path, "rb") as pickled_event:
                    with open(uncleaned_path, "rb") as pickled_original:
                        (
                            event_data,
                            data_format,
                            features,
                            feature_cluster,
                        ) = pickle.load(pickled_event)
                        uncleaned_data, _, _, _ = pickle.load(pickled_original)
                        uncleaned_photons = uncleaned_data[data_format["Image"]]
                        uncleaned_photons = list_of_lists_to_raw_phs(uncleaned_photons)
                        uncleaned_cloud = np.asarray(
                            raw_phs_to_point_cloud(
                                uncleaned_photons,
                                cx=GEOMETRY.x_angle,
                                cy=GEOMETRY.y_angle,
                            )
                        )
                        # Convert List of List to Point Cloud
                        event_photons = event_data[data_format["Image"]]
                        event_photons = list_of_lists_to_raw_phs(event_photons)
                        point_cloud = np.asarray(
                            raw_phs_to_point_cloud(
                                event_photons, cx=GEOMETRY.x_angle, cy=GEOMETRY.y_angle
                            )
                        )
                        out = np.where(
                            (uncleaned_cloud == point_cloud[:, None]).all(-1)
                        )[1]
                        point_values = np.zeros(uncleaned_cloud.shape)
                        point_values[out] = 1
                        if self.clumps:
                            with open(clump_path, "rb") as pickled_clump:
                                clump_data, _, _, _ = pickle.load(pickled_clump)
                                clump_photons = clump_data[data_format["Image"]]
                                clump_photons = list_of_lists_to_raw_phs(clump_photons)
                                clump_cloud = np.asarray(
                                    raw_phs_to_point_cloud(
                                        clump_photons,
                                        cx=GEOMETRY.x_angle,
                                        cy=GEOMETRY.y_angle,
                                    )
                                )
                                out = np.where(
                                    (uncleaned_cloud == clump_cloud[:, None]).all(-1)
                                )[1]
                                clump_values = np.zeros(uncleaned_cloud.shape)
                                clump_values[out] = 1
                                # clump_values = np.isclose(clump_cloud, point_cloud)
                                # Convert to ints so that addition works, gives 0 for outside, 1 clump, 2 core
                                point_values = point_values.astype(
                                    int
                                ) + clump_values.astype(int)
                                point_values = point_values[:, 0]
                        else:
                            point_values = point_values.astype(int)
                            point_values = point_values[:, 0]
                        data = Data(
                            pos=torch.tensor(uncleaned_cloud, dtype=torch.float).squeeze(), y=point_values
                        )  # Just need x,y,z ignore derived features
                        if self.pre_filter is not None and not self.pre_filter(data):
                            return

                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        torch.save(
                            data,
                            osp.join(
                                self.processed_dir,
                                "cluster_{}.pt".format(base_path),
                            ),
                        )
                        self.processed_filenames.append(
                            "cluster_{}.pt".format(base_path)
                        )
            except Exception as e:
                print(f"Failed: {e}")
                return

    def process(self):

        used_paths = split_data(self.raw_file_names)[self.split]
        pool = Pool()
        processors = pool.map_async(self.process_file, used_paths)
        processors.wait()
        print("Done Processing!")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, self.processed_file_names[idx])
        )
        return data

# Function to check test set's identifier.
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xFFFFFFFF < test_ratio * 2 ** 32


# Function to split train/test
def split_train_test_by_id(data, test_ratio):
    in_test_set = np.asarray(
        [test_set_check(crc32(str(x).encode()), test_ratio) for x in data]
    )
    return data[~in_test_set], data[in_test_set]


def split_data(paths, val_split=0.2, test_split=0.2):
    """
    Split up the data and return which images should go to which train, test, val directory
    :param paths: The paths to do the splitting on
    :param test_split: Fraction of the data for the test set. the validation set is rolled into the test set.
    :param val_split: Fraction of data in validation set
    :return: A dict containing which images go to which directory
    """

    print(len(paths))
    train, test = split_train_test_by_id(np.asarray(paths), val_split + test_split)
    val, test = split_train_test_by_id(test, val_split)
    print(train.shape)
    print(val.shape)
    print(test.shape)
    return {
        "train": train,
        "val": val,
        "trainval": np.concatenate((train, val)),
        "test": test,
        "all": np.concatenate((train, val, test)),
    }


#gammas = np.loadtxt("/run/media/jacob/SSD_Backup/iact_events/gamma_diffuse/no_clean/no_clean_diffuse.txt", dtype=str)
#pickle.dump(list(gammas), open(osp.join('/home/jacob/Development/factnn/factnn/data/resources','no_clean_diffuse_raw_names.p'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#gammas = np.loadtxt("/run/media/jacob/SSD_Backup/iact_events/gamma_diffuse/clump20/clump20_diffuse.txt", dtype=str)
#pickle.dump(list(gammas), open(osp.join('/home/jacob/Development/factnn/factnn/data/resources','clump20_diffuse_raw_names.p'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#gammas = np.loadtxt("/run/media/jacob/SSD_Backup/iact_events/gamma_diffuse/core20/core20_diffuse.txt", dtype=str)
#pickle.dump(list(gammas), open(osp.join('/home/jacob/Development/factnn/factnn/data/resources','core20_diffuse_raw_names.p'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
# gammas = np.loadtxt("/run/media/jacob/T7/no_clean/processed/t.txt", dtype=str)
# pickle.dump(list(gammas), open(osp.join('/home/jacob/Development/factnn/factnn/data/resources','no_clean_test_processed.p'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
