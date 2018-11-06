from factnn import GammaPreprocessor, ProtonPreprocessor, SeparationGenerator, SeparationModel
import os.path
from factnn.utils import kfold

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]
proton_dir = [base_dir + "sim/proton/"]

shape = [30,70]
rebin_size = 5

gamma_configuration = {
    'directories': gamma_dir,
    'rebin_size': rebin_size,
    'output_file': "../gamma.hdf5",
    'shape': shape

}

proton_configuration = {
    'directories': proton_dir,
    'rebin_size': rebin_size,
    'output_file': "../proton1.hdf5",
    'shape': shape

}

proton_preprocessor = ProtonPreprocessor(config=proton_configuration)
gamma_preprocessor = GammaPreprocessor(config=gamma_configuration)

if not os.path.isfile(proton_configuration["output_file"]):
    proton_preprocessor.create_dataset()
if not os.path.isfile(gamma_configuration["output_file"]):
    gamma_preprocessor.create_dataset()


indexes = kfold.split_data(range(0, 19258), kfolds=5)
print(len(indexes[0][0]))
print(len(indexes[1][0]))
print(len(indexes[2][0]))

separation_generator_configuration = {
    'seed': 1337,
    'batch_size': 16,
    'input': '../gamma.hdf5',
    'second_input': '../proton1.hdf5',
    'start_slice': 0,
    'number_slices': shape[1]-shape[0],
    'train_data': indexes[0][0],
    'validate_data': indexes[1][0],
    'test_data': indexes[2][0],
    'mode': 'train',
    'samples': 19258,
    'chunked': False,
    'augment': True,
}

separation_validate = SeparationGenerator(config=separation_generator_configuration)
separation_train = SeparationGenerator(config=separation_generator_configuration)
separation_test = SeparationGenerator(config=separation_generator_configuration)

separation_validate.mode = "validate"
separation_train.mode = "train"
separation_test.mode = "test"

separation_model_configuration = {
    'conv_dropout': 0.1,
    'lstm_dropout': 0.2,
    'fc_dropout': 0.4,
    'num_conv3d': 3,
    'kernel_conv3d': 3,
    'strides_conv3d': 1,
    'num_lstm': 0,
    'kernel_lstm': 2,
    'strides_lstm': 1,
    'num_fc': 2,
    'pooling': True,
    'neurons': [32, 32, 32, 32, 64],
    'shape': [40, 75, 75, 1],
    'start_slice': 0,
    'number_slices': 25,
    'activation': 'relu',
    'name': 'testSep',
}

separation_model = SeparationModel(config=separation_model_configuration)

print(separation_model)

"""

Now run the models with the generators!

"""

separation_model.train_generator = separation_train
separation_model.validate_generator = separation_validate
separation_model.train_generator = separation_test

separation_model.train(train_generator=separation_train, validate_generator=separation_validate)

