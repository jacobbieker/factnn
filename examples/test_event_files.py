from factnn import ProtonPreprocessor, GammaPreprocessor, GammaDiffusePreprocessor
import os

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]
proton_dir = [base_dir + "sim/proton/"]
gamma_dl2 = "../gamma_simulations_diffuse_facttools_dl2.hdf5"

output_path = "/home/jacob/Development/FACT-NN-Analysis/event_files"

shape = [30,70]
rebin_size = 5

# Get paths from the directories
crab_paths = []
for directory in proton_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                crab_paths.append(os.path.join(root, file))

proton_configuration = {
    'rebin_size': rebin_size,
    'output_file': "../proton.hdf5",
    'shape': shape,
    'paths': crab_paths,
    'as_channels': True
}


proton_train_preprocessor = ProtonPreprocessor(config=proton_configuration)
proton_train_preprocessor.event_processor(os.path.join(output_path, "proton"))

# Get paths from the directories
gamma_paths = []
for directory in gamma_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                gamma_paths.append(os.path.join(root, file))

# Now do the Kfold Cross validation Part for both sets of paths

gamma_configuration = {
    'rebin_size': rebin_size,
    'output_file': "../gamma.hdf5",
    'shape': shape,
    'paths': gamma_paths
}



gamma_train_preprocessor = GammaPreprocessor(config=gamma_configuration)
gamma_train_preprocessor.event_processor(os.path.join(output_path, "gamma"))

gamma_configuration = {
    'rebin_size': rebin_size,
    'output_file': "../gamma.hdf5",
    'shape': shape,
    'dl2_file': gamma_dl2,
    'paths': gamma_paths
}


gamma_train_preprocessor = GammaDiffusePreprocessor(config=gamma_configuration)
gamma_train_preprocessor.event_processor(os.path.join(output_path, "diffuse_gamma"))
