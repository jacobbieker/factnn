import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from factnn import GammaPreprocessor, ProtonPreprocessor, SeparationModel
from factnn.generator.generator.separation_generators import SeparationGenerator
import os.path
from factnn.utils import kfold
from keras.models import load_model

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]
proton_dir = [base_dir + "sim/proton/"]

shape = [30,70]
rebin_size = 5

# Get paths from the directories
gamma_paths = []
for directory in gamma_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                gamma_paths.append(os.path.join(root, file))


# Get paths from the directories
proton_paths = []
for directory in proton_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                proton_paths.append(os.path.join(root, file))


# Now do the Kfold Cross validation Part for both sets of paths
gamma_indexes = kfold.split_data(gamma_paths, kfolds=5)
proton_indexes = kfold.split_data(proton_paths, kfolds=5)


gamma_configuration = {
    'rebin_size': rebin_size,
    'output_file': "../gamma.hdf5",
    'shape': shape,
    'paths': gamma_indexes[0][0]
}

proton_configuration = {
    'rebin_size': rebin_size,
    'output_file': "../proton.hdf5",
    'shape': shape,
    'paths': proton_indexes[0][0]
}


proton_train_preprocessor = ProtonPreprocessor(config=proton_configuration)
gamma_train_preprocessor = GammaPreprocessor(config=gamma_configuration)

gamma_configuration['paths'] = gamma_indexes[1][0]
proton_configuration['paths'] = proton_indexes[1][0]

proton_validate_preprocessor = ProtonPreprocessor(config=proton_configuration)
gamma_validate_preprocessor = GammaPreprocessor(config=gamma_configuration)


separation_generator_configuration = {
    'seed': 1337,
    'batch_size': 16,
    'start_slice': 0,
    'number_slices': shape[1] - shape[0],
    'mode': 'train',
    'chunked': False,
    'augment': True,
    'from_directory': True,
    'input_shape': [-1, gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
}

separation_validate = SeparationGenerator(config=separation_generator_configuration)
separation_train = SeparationGenerator(config=separation_generator_configuration)

separation_validate.mode = "validate"
separation_train.mode = "train"

separation_train.proton_train_preprocessor = proton_train_preprocessor
separation_train.proton_validate_preprocessor = proton_validate_preprocessor
separation_train.train_preprocessor = gamma_train_preprocessor
separation_train.validate_preprocessor = gamma_validate_preprocessor

separation_validate.proton_train_preprocessor = proton_train_preprocessor
separation_validate.proton_validate_preprocessor = proton_validate_preprocessor
separation_validate.train_preprocessor = gamma_train_preprocessor
separation_validate.validate_preprocessor = gamma_validate_preprocessor

separation_model_configuration = {
    'conv_dropout': 0.4,
    'lstm_dropout': 0.5,
    'fc_dropout': 0.5,
    'num_conv3d': 0,
    'kernel_conv3d': 3,
    'strides_conv3d': 1,
    'num_lstm': 3,
    'kernel_lstm': 5,
    'strides_lstm': 2,
    'num_fc': 2,
    'pooling': True,
    'neurons': [16, 16, 16, 16, 16],
    'shape': [gamma_train_preprocessor.shape[3], gamma_train_preprocessor.shape[2], gamma_train_preprocessor.shape[1], 1],
    'start_slice': 0,
    'number_slices': shape[1] - shape[0],
    'activation': 'relu',
}

separation_model = SeparationModel(config=separation_model_configuration)

print(separation_model)

# Save the base model to use for the kfold validation
separation_model.save("Base_Separation.hdf5")
separation_model.model.save_weights("Base_Separation_weights.hdf5")
"""

Now run the models with the generators!

"""

separation_model.train_generator = separation_train
separation_model.validate_generator = separation_validate

from examples.open_crab_sample_constants import NUM_EVENTS_GAMMA, NUM_EVENTS_PROTON

for fold in range(5):
    print(fold)
    # Now change preprocessors
    gamma_configuration['paths'] = gamma_indexes[0][fold]
    proton_configuration['paths'] = proton_indexes[0][fold]
    proton_train_preprocessor = ProtonPreprocessor(config=proton_configuration)
    gamma_train_preprocessor = GammaPreprocessor(config=gamma_configuration)

    gamma_configuration['paths'] = gamma_indexes[1][fold]
    proton_configuration['paths'] = proton_indexes[1][fold]

    proton_validate_preprocessor = ProtonPreprocessor(config=proton_configuration)
    gamma_validate_preprocessor = GammaPreprocessor(config=gamma_configuration)

    separation_validate = SeparationGenerator(config=separation_generator_configuration)
    separation_train = SeparationGenerator(config=separation_generator_configuration)

    separation_validate.mode = "validate"
    separation_train.mode = "train"

    separation_train.proton_train_preprocessor = proton_train_preprocessor
    separation_train.proton_validate_preprocessor = proton_validate_preprocessor
    separation_train.train_preprocessor = gamma_train_preprocessor
    separation_train.validate_preprocessor = gamma_validate_preprocessor

    separation_validate.proton_train_preprocessor = proton_train_preprocessor
    separation_validate.proton_validate_preprocessor = proton_validate_preprocessor
    separation_validate.train_preprocessor = gamma_train_preprocessor
    separation_validate.validate_preprocessor = gamma_validate_preprocessor
    separation_model.model.load_weights("Base_Separation_weights.hdf5")
    separation_model.train(train_generator=separation_train, validate_generator=separation_validate, val_num=int(NUM_EVENTS_PROTON*0.8*0.2), num_events=int(NUM_EVENTS_PROTON*0.8*0.8))
    separation_model.save("fold_" + str(fold) + "_separation.hdf5")