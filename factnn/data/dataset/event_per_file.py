import os
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from factnn import ProtonPreprocessor, GammaPreprocessor, GammaDiffusePreprocessor
from multiprocessing import Pool
from functools import partial


def process_diffuse_gamma_files(
    data_dir="",
    output_dir="",
    clump_size=5,
    num_workers=12,
    hdf_file="../gamma.hdf5",
    gamma_dl2="../gamma_simulations_diffuse_facttools_dl2.hdf5",
):
    # Get paths from the directories
    paths = []
    for directory in data_dir:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("phs.jsonl.gz"):
                    paths.append(os.path.join(root, file))

    def process_diffuse_gamma(clump_size, path):
        gamma_configuration = {
            "rebin_size": 5,
            "output_file": hdf_file,
            "shape": [30, 70],
            "paths": [path],
            "dl2_file": gamma_dl2,
        }

        gamma_train_preprocessor = GammaDiffusePreprocessor(config=gamma_configuration)
        gamma_train_preprocessor.event_processor(
            directory=output_dir,
            clean_images=True,
            only_core=True,
            clump_size=clump_size,
        )

    pool = Pool(num_workers)
    func = partial(process_diffuse_gamma, clump_size)
    jobs = pool.map_async(func, paths)

    return jobs


def process_gamma_files(
    data_dir="", output_dir="", clump_size=5, num_workers=12, hdf_file="../gamma.hdf5"
):
    # Get paths from the directories
    paths = []
    for directory in data_dir:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("phs.jsonl.gz"):
                    paths.append(os.path.join(root, file))

    def process_gamma(clump_size, path):
        print("Gamma")
        print("Size: ", clump_size)
        gamma_configuration = {
            "rebin_size": 5,
            "output_file": hdf_file,
            "shape": [30, 70],
            "paths": [path],
        }

        gamma_train_preprocessor = GammaPreprocessor(config=gamma_configuration)
        gamma_train_preprocessor.event_processor(
            output_dir, clean_images=True, only_core=True, clump_size=clump_size
        )

    pool = Pool(num_workers)
    func = partial(process_gamma, clump_size)
    jobs = pool.map_async(func, paths)

    return jobs


def process_proton_files(
    data_dir="", output_dir="", clump_size=5, num_workers=12, hdf_file="../proton.hdf5"
):
    # Get paths from the directories
    paths = []
    for directory in data_dir:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("phs.jsonl.gz"):
                    paths.append(os.path.join(root, file))

    def process_proton(clump_size, path):
        proton_configuration = {
            "rebin_size": 5,
            "output_file": hdf_file,
            "shape": [30, 70],
            "paths": [path],
        }

        proton_train_preprocessor = ProtonPreprocessor(config=proton_configuration)
        proton_train_preprocessor.event_processor(
            output_dir, clean_images=True, only_core=True, clump_size=clump_size
        )

    pool = Pool(num_workers)
    func = partial(process_proton, clump_size)
    jobs = pool.map_async(func, paths)

    return jobs
