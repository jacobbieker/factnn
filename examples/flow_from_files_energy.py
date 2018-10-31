from factnn import GammaPreprocessor, EnergyGenerator, EnergyModel
import os.path
from factnn.utils import kfold

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]

shape = [0,70]
rebin_size = 10

# Get paths from the directories
gamma_paths = []
for directory in gamma_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                gamma_paths.append(os.path.join(root, file))

# Now do the Kfold Cross validation Part for both sets of paths
gamma_indexes = kfold.split_data(gamma_paths, kfolds=5)

gamma_configuration = {
    'rebin_size': rebin_size,
    'output_file': "../gamma.hdf5",
    'shape': shape,
    'paths': gamma_indexes[0][0]
}



gamma_train_preprocessor = GammaPreprocessor(config=gamma_configuration)

gamma_configuration['paths'] = gamma_indexes[1][0]

gamma_validate_preprocessor = GammaPreprocessor(config=gamma_configuration)

energy_generator_configuration = {
    'seed': 1337,
    'batch_size': 32,
    'start_slice': 0,
    'number_slices': shape[1]-shape[0],
    'mode': 'train',
    'chunked': False,
    'augment': True,
    'from_directory': True,
    'input_shape': [-1, gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
}

energy_validate = EnergyGenerator(config=energy_generator_configuration)
energy_train = EnergyGenerator(config=energy_generator_configuration)
energy_test = EnergyGenerator(config=energy_generator_configuration)
energy_validate.mode = "validate"
energy_train.mode = "train"
energy_test.mode = "test"

energy_train.train_preprocessor = gamma_train_preprocessor
energy_validate.validate_preprocessor = gamma_validate_preprocessor

energy_model_configuration = {
    'conv_dropout': 0.1,
    'lstm_dropout': 0.2,
    'fc_dropout': 0.3,
    'num_conv3d': 3,
    'kernel_conv3d': 3,
    'strides_conv3d': 1,
    'num_lstm': 0,
    'kernel_lstm': 2,
    'strides_lstm': 1,
    'num_fc': 2,
    'pooling': True,
    'neurons': [8, 16, 32, 64, 128, 256],
    'shape': [gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
    'start_slice': 0,
    'number_slices': 25,
    'activation': 'relu',
    'patience': 10,
}

energy_model = EnergyModel(config=energy_model_configuration)
print(energy_model)

"""

Now run the models with the generators!

"""

energy_model.train_generator = energy_train
energy_model.validate_generator = energy_validate
energy_model.test_generator = energy_test

energy_model.train(train_generator=energy_train, validate_generator=energy_validate, num_events=gamma_train_preprocessor.count_events(), val_num=gamma_validate_preprocessor.count_events())


