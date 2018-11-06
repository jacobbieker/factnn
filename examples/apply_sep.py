from factnn import SeparationGenerator, SeparationModel, ObservationPreprocessor, GammaPreprocessor, ProtonPreprocessor
import os.path
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from examples.open_crab_sample_constants import NUM_EVENTS_CRAB

base_dir = "../ihp-pc41.ethz.ch/public/phs/"
obs_dir = [base_dir + "public/"]
gamma_dir = [base_dir + "sim/gamma/"]
proton_dir = [base_dir + "sim/proton/"]

crab_dl2 = "../open_crab_sample_facttools_dl2.hdf5"

shape = [30,70]
rebin_size = 5

# Get paths from the directories
crab_paths = []
for directory in obs_dir:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("phs.jsonl.gz"):
                crab_paths.append(os.path.join(root, file))

gamma_configuration = {
    'rebin_size': rebin_size,
    'shape': shape,
    'paths': crab_paths,
    'as_channels': True,
    'dl2_file': crab_dl2
}

gamma_train_preprocessor = ObservationPreprocessor(config=gamma_configuration)

separation_model = load_model("Outside_sep_prelu.hdf5")

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
    'as_channels': True,
}
gamma_train_preprocessor.num_events = NUM_EVENTS_CRAB
separation_validate = SeparationGenerator(config=separation_generator_configuration)
separation_validate.mode = "test"
separation_validate.test_preprocessor = gamma_train_preprocessor
separation_validate.proton_test_preprocessor = gamma_train_preprocessor

crab_configuration = {
    'rebin_size': rebin_size,
    'shape': shape,
    'paths': crab_paths,
    'as_channels': True
}

num_events = NUM_EVENTS_CRAB
steps = int(np.floor(num_events/16))
starting_step = 0
truth = []
predictions = []
import pickle
if os.path.isfile("crab_predictions.p"):
    with open("crab_predictions.p", "rb") as savedfile:
        predictions = pickle.load(savedfile)
        starting_step = int(len(predictions))
        for i in range(0, starting_step):
            next(separation_validate)
            print(i)

for i in range(starting_step, steps):
    print("Step: " + str(i) + "/" + str(steps))
    # Get each batch and test it
    test_images, test_labels = next(separation_validate)
    test_predictions = separation_model.predict_on_batch(test_images)
    predictions.append(test_predictions)
    truth.append(test_labels)
    if i % 10 == 0 and i != 0:
        # Save predictions every ten steps
        with open("crab_predictions.p", "wb") as savefile:
            pickle.dump(predictions, savefile)

predictions = np.asarray(predictions).reshape(-1, )
truth = np.asarray(truth).reshape(-1, )

from factnn import plotting

plot = plotting.plot_roc(truth, predictions)
plt.show()
