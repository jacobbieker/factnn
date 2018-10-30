from factnn import GammaDiffusePreprocessor, DispGenerator, DispModel, SignGenerator, SignModel
import os.path
from factnn.data import kfold

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]
gamma_dl2 = "../gamma_simulations_diffuse_facttools_dl2.hdf5"

shape = [30,70]
rebin_size = 2

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
    'dl2_file': gamma_dl2,
    'paths': gamma_indexes[0][0]
}


gamma_train_preprocessor = GammaDiffusePreprocessor(config=gamma_configuration)

gamma_configuration['paths'] = gamma_indexes[1][0]

gamma_validate_preprocessor = GammaDiffusePreprocessor(config=gamma_configuration)


source_generator_configuration = {
    'seed': 1337,
    'batch_size': 4,
    'start_slice': 0,
    'number_slices': 38,
    'mode': 'train',
    'chunked': False,
    'augment': True,
    'from_directory': True,
    'input_shape': [-1, 38, gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
}


disp_train = DispGenerator(config=source_generator_configuration)
disp_validate = DispGenerator(config=source_generator_configuration)
disp_test = DispGenerator(config=source_generator_configuration)

disp_train.train_preprocessor = gamma_train_preprocessor
disp_train.validate_preprocessor = gamma_validate_preprocessor
disp_validate.train_preprocessor = gamma_train_preprocessor
disp_validate.validate_preprocessor = gamma_validate_preprocessor

disp_train.mode = "train"
disp_validate.mode = "validate"
disp_test.mode = "test"

source_model_configuration = {
    'conv_dropout': 0.2,
    'lstm_dropout': 0.3,
    'fc_dropout': 0.5,
    'num_conv3d': 2,
    'kernel_conv3d': 2,
    'strides_conv3d': 1,
    'num_lstm': 1,
    'kernel_lstm': 2,
    'strides_lstm': 1,
    'num_fc': 2,
    'pooling': True,
    'neurons': [16, 16, 32, 24, 36],
    'shape': [38, gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
    'start_slice': 0,
    'number_slices': 38,
    'activation': 'relu',
}

disp_model = DispModel(config=source_model_configuration)

"""

Now run the models with the generators!

"""

disp_model.train_generator = disp_train
disp_model.validate_generator = disp_validate
disp_model.test_generator = disp_test

disp_model.train(train_generator=disp_train, validate_generator=disp_validate, num_events=int(550000*0.8*0.8), val_num=int(550000*0.8*0.2))
