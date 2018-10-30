from factnn import GammaDiffusePreprocessor, DispGenerator, DispModel, SignGenerator, SignModel
import os.path
import numpy as np

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]
gamma_dl2 = "../gamma_simulations_diffuse_facttools_dl2.hdf5"

shape = [30,70]
rebin_size = 3

# Get paths from the directories
gamma_paths = []
for directory in gamma_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                gamma_paths.append(os.path.join(root, file))


gamma_diffuse_configuration = {
    'directories': gamma_dir,
    'rebin_size': rebin_size,
    'dl2_file': gamma_dl2,
    'output_file': "../gamma_diffuse.hdf5",
    'shape': shape

}

gamma_diffuse_preprocessor = GammaDiffusePreprocessor(config=gamma_diffuse_configuration)

if not os.path.isfile(gamma_diffuse_configuration["output_file"]):
    gamma_diffuse_preprocessor.create_dataset()

# Since the Gamma Diffuse simulations seem to be located in order, this should allow for better testing of small groups
# So that not need to use whole thing, but also get a random uniform sample from it anyways
used_positions = list(np.random.randint(0, 550000, size=200000))

source_generator_configuration = {
    'seed': 1337,
    'batch_size': 32,
    'input': '../gamma_diffuse.hdf5',
    'start_slice': 0,
    'number_slices': 25,
    'train_fraction': 0.6,
    'validate_fraction': 0.2,
    'mode': 'train',
    'samples': used_positions,
    'chunked': False,
    'augment': True,
}


disp_train = DispGenerator(config=source_generator_configuration)
disp_validate = DispGenerator(config=source_generator_configuration)
disp_test = DispGenerator(config=source_generator_configuration)

disp_train.mode = "train"
disp_validate.mode = "validate"
disp_test.mode = "test"

sign_train = SignGenerator(config=source_generator_configuration)
sign_validate = SignGenerator(config=source_generator_configuration)
sign_test = SignGenerator(config=source_generator_configuration)

sign_train.mode = "train"
sign_validate.mode = "validate"
sign_test.mode = "test"


source_model_configuration = {
    'conv_dropout': 0.2,
    'lstm_dropout': 0.3,
    'fc_dropout': 0.5,
    'num_conv3d': 1,
    'kernel_conv3d': 2,
    'strides_conv3d': 1,
    'num_lstm': 3,
    'kernel_lstm': 2,
    'strides_lstm': 1,
    'num_fc': 2,
    'pooling': True,
    'neurons': [32, 32, 32, 64, 32, 48],
    'shape': [25, 38, 38, 1],
    'start_slice': 0,
    'number_slices': 25,
    'activation': 'relu',
    'patience': 200,
}

sign_model_configuration = {
    'conv_dropout': 0.2,
    'lstm_dropout': 0.3,
    'fc_dropout': 0.5,
    'num_conv3d': 2,
    'kernel_conv3d': 2,
    'strides_conv3d': 1,
    'num_lstm': 2,
    'kernel_lstm': 2,
    'strides_lstm': 2,
    'num_fc': 2,
    'pooling': True,
    'neurons': [32, 16, 8, 16, 32, 48],
    'shape': [25, 38, 38, 1],
    'start_slice': 0,
    'number_slices': 25,
    'activation': 'relu',
    'name': 'testSign',
}

disp_model = DispModel(config=source_model_configuration)
sign_model = SignModel(config=sign_model_configuration)

"""

Now run the models with the generators!

"""

disp_model.train_generator = disp_train
disp_model.validate_generator = disp_validate
disp_model.test_generator = disp_test

disp_model.train()

sign_model.train_generator = sign_train
sign_model.validate_generator = sign_validate
sign_model.test_generator = sign_test

sign_model.train()
