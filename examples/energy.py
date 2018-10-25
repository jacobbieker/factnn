from factnn import GammaPreprocessor, EnergyGenerator, EnergyModel
import os.path

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

energy_generator_configuration = {
    'seed': 1337,
    'batch_size': 64,
    'input': '../gamma.hdf5',
    'start_slice': 0,
    'number_slices': 25,
    'train_fraction': 0.6,
    'validate_fraction': 0.2,
    'mode': 'train',
    'samples': 800000,
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
    'fc_dropout': 0.5,
    'num_conv3d': 3,
    'kernel_conv3d': 2,
    'strides_conv3d': 1,
    'num_lstm': 1,
    'kernel_lstm': 2,
    'strides_lstm': 1,
    'num_fc': 3,
    'pooling': True,
    'neurons': [32, 16, 8, 16, 32, 48, 64],
    'shape': [25, 38, 38, 1],
    'start_slice': 0,
    'number_slices': 25,
    'activation': 'relu',
    'name': 'testEnergy',
}

energy_model = EnergyModel(config=energy_model_configuration)

"""

Now run the models with the generators!

"""

energy_model.train_generator = energy_train
energy_model.validate_generator = energy_validate
energy_model.test_generator = energy_test

energy_model.train()
energy_model.apply()


