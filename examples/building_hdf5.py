from factnn import GammaPreprocessor, GammaDiffusePreprocessor, ProtonPreprocessor
import os.path
"""
This is just to show how to make all the HDF5 files, its simply the same as in the energy, separation, and source_detection
files but without the need to import tensorflow or anything
"""

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]
proton_dir = [base_dir + "sim/proton/"]
gamma_dl2 = "../gamma_simulations_diffuse_facttools_dl2.hdf5"

shape = [35,60]
rebin_size = 10

gamma_diffuse_configuration = {
    'directories': gamma_dir,
    'rebin_size': rebin_size,
    'dl2_file': gamma_dl2,
    'output_file': "../gamma_diffuse.hdf5",
    'shape': shape

}

gamma_configuration = {
    'directories': gamma_dir,
    'rebin_size': rebin_size,
    'output_file': "../gamma.hdf5",
    'shape': shape

}

proton_configuration = {
    'directories': proton_dir,
    'rebin_size': rebin_size,
    'output_file': "../proton.hdf5",
    'shape': shape

}

gamma_diffuse_preprocessor = GammaDiffusePreprocessor(config=gamma_diffuse_configuration)
proton_preprocessor = ProtonPreprocessor(config=proton_configuration)
gamma_preprocessor = GammaPreprocessor(config=gamma_configuration)

if not os.path.isfile(gamma_diffuse_configuration["output_file"]):
    gamma_diffuse_preprocessor.create_dataset()
if not os.path.isfile(proton_configuration["output_file"]):
    proton_preprocessor.create_dataset()
if not os.path.isfile(gamma_configuration["output_file"]):
    gamma_preprocessor.create_dataset()