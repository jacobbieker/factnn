import os
from os import path

# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from factnn import ProtonPreprocessor, GammaPreprocessor, GammaDiffusePreprocessor
import os

from multiprocessing import Pool
from functools import partial


base_dir = "/run/media/bieker/T7/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/werner", base_dir + "sim/gamma/gustav"]
proton_dir = [base_dir + "sim/proton/"]
gamma_dl2 = "/run/media/bieker/T7/gamma_simulations_diffuse_facttools_dl2 (1).hdf5"
gamma_dl2 = "/run/media/bieker/T7/gamma_simulations_facttools_dl2.hdf5"


# output_path = "/home/jacob/Documents/cleaned_event_files_test"
output_path = "/run/media/bieker/T7/proton/no_clean"

shape = [30, 70]
rebin_size = 5

# Get paths from the directories
gamma_paths = []
for directory in gamma_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                gamma_paths.append(os.path.join(root, file))
print(len(gamma_paths))


def gf(clump_size, path):
    print("Gamma_Diffuse")
    print("Size: ", clump_size)
    gamma_configuration = {
        "rebin_size": clump_size,
        "output_file": "../gamma.hdf5",
        "shape": shape,
        "paths": [path],
        "dl2_file": gamma_dl2,
    }

    gamma_train_preprocessor = GammaDiffusePreprocessor(config=gamma_configuration)
    gamma_train_preprocessor.event_processor(
        directory=os.path.join(output_path),
        clean_images=False,
        only_core=True,
        clump_size=clump_size,
    )


# Now do the Kfold Cross validation Part for both sets of paths


if __name__ == "__main__":

    clump_size = 5

    # proton_pool = Pool(8)
    gamma_pool = Pool()
    # dunc = partial(d, clump_size)
    # gunc = partial(gf, clump_size)
    # g = proton_pool.map_async(dunc, crab_paths)
    func = partial(gf, clump_size)
    r = gamma_pool.map(func, gamma_paths)

    # g.wait()
    # print("\n\n\n\n\n\n\n----------------------------------Done Proton------------------------------------------------\n\n\n\n\n\n\n\n")
    r.wait()
