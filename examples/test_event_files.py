import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from factnn import ProtonPreprocessor, GammaPreprocessor, GammaDiffusePreprocessor
import os

import multiprocessing
from multiprocessing import Pool


base_dir = "/home/jacob/ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]
proton_dir = [base_dir + "sim/proton/"]
gamma_dl2 = "../gamma_simulations_diffuse_facttools_dl2.hdf5"

output_path = "/home/jacob/Documents/cleaned_event_files_test"

shape = [30,70]
rebin_size = 5

# Get paths from the directories
gamma_paths = []
for directory in gamma_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                gamma_paths.append(os.path.join(root, file))


def f(path):
    gamma_configuration = {
        'rebin_size': 5,
        'output_file': "../gamma.hdf5",
        'shape': shape,
        'paths': [path]
    }

    gamma_train_preprocessor = GammaPreprocessor(config=gamma_configuration)
    gamma_train_preprocessor.event_processor(os.path.join(output_path, "gamma"), clean_images=True)

# Get paths from the directories
crab_paths = []
for directory in proton_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                crab_paths.append(os.path.join(root, file))

def d(path):
    print(len(path))
    proton_configuration = {
        'rebin_size': rebin_size,
        'output_file': "../proton.hdf5",
        'shape': shape,
        'paths': [path],
    }

    proton_train_preprocessor = ProtonPreprocessor(config=proton_configuration)
    proton_train_preprocessor.event_processor(os.path.join(output_path, "proton"), clean_images=True)


# Now do the Kfold Cross validation Part for both sets of paths


if __name__ == '__main__':
    with Pool(5) as p:
        p.map(f, gamma_paths)
    with Pool(5) as p:
        p.map(d, crab_paths)

