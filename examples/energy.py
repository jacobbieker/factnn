from factnn import GammaPreprocessor, EnergyGenerator, EnergyModel
import os.path
from factnn.utils import kfold
import numpy as np

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]

shape = [35,60]
rebin_size = 10

gamma_configuration = {
    'directories': gamma_dir,
    'rebin_size': rebin_size,
    'output_file': "../gamma.hdf5",
    'shape': shape

}

gamma_preprocessor = GammaPreprocessor(config=gamma_configuration)

if not os.path.isfile(gamma_configuration["output_file"]):
    gamma_preprocessor.create_dataset()

used_positions = list(np.random.choice(800000, replace=False, size=400000))
indexes = kfold.split_data(used_positions, kfolds=5)
print(len(indexes[0][0]))
print(len(indexes[1][0]))
print(len(indexes[2][0]))

energy_generator_configuration = {
    'seed': 1337,
    'batch_size': 32,
    'input': '../gamma.hdf5',
    'start_slice': 0,
    'number_slices': 25,
    'train_data': indexes[0][0],
    'validate_data': indexes[1][0],
    'test_data': indexes[2][0],
    'mode': 'train',
    'chunked': False,
    'augment': True,
}

energy_validate = EnergyGenerator(config=energy_generator_configuration)
energy_train = EnergyGenerator(config=energy_generator_configuration)
energy_test = EnergyGenerator(config=energy_generator_configuration)
energy_validate.mode = "validate"
energy_train.mode = "train"
energy_test.mode = "test"

energy_model_configuration = {
    'conv_dropout': 0.1,
    'lstm_dropout': 0.2,
    'fc_dropout': 0.3,
    'num_conv3d': 4,
    'kernel_conv3d': 2,
    'strides_conv3d': 1,
    'num_lstm': 0,
    'kernel_lstm': 2,
    'strides_lstm': 1,
    'num_fc': 2,
    'pooling': True,
    'neurons': [32, 64, 128, 64, 64, 128],
    'shape': [25, 38, 38, 1],
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

energy_model.train(train_generator=energy_train, validate_generator=energy_validate)
energy_model.apply()


