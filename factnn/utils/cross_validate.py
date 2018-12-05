import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
import keras
from ..data.preprocess.eventfile_preprocessor import EventFilePreprocessor
from ..generator.keras.eventfile_generator import EventFileGenerator
import os


def split_data(indicies, kfolds, seeds=None):
    """
    Splits the data into the indicies for kfold validation
    :param indicies: Int or Array, if int, assumed to be the size of the dataset
    :param kfolds:  Number of folds to do
    :return: Array (3, kfolds, variable) that holds the indicies for the training, validation, and testing
    """

    if type(indicies) is int:
        indicies = np.arange(0, indicies)

    if seeds is None:
        seeds = []
        for fold in range(kfolds):
            seeds.append(np.random.randint(0, 2 ** 32 - 1))

    # Now split into kfolds,
    list_of_training = []
    list_of_validate = []
    list_of_testing = []
    validate_fraction = 0.2
    indicies = shuffle(indicies)
    indicies = np.asarray(indicies)
    # Now get KFOLD splits
    kf = KFold(n_splits=kfolds, shuffle=True)

    for train_index, test_index in kf.split(indicies):
        # Need to split train_index into validation data
        train_data, validate_data = train_test_split(indicies[train_index], train_size=(1.0 - validate_fraction),
                                                     test_size=validate_fraction)
        list_of_training.append(train_data)
        list_of_validate.append(validate_data)
        list_of_testing.append(indicies[test_index])

    # Now convert to a numpy array
    return list_of_training, list_of_validate, list_of_testing


def data(start_slice, end_slice, final_slices, rebin_size, gamma_train, proton_train=None, model_type="Separation",
         batch_size=8, as_channels=True, normalize=False, kfold_index=0):
    shape = [start_slice, end_slice]

    gamma_configuration = {
        'rebin_size': rebin_size,
        'shape': shape,
        'paths': gamma_train[0][kfold_index],
        'as_channels': as_channels
    }
    gamma_train_preprocessor = EventFilePreprocessor(config=gamma_configuration)
    gamma_configuration['paths'] = gamma_train[1][kfold_index]
    gamma_validate_preprocessor = EventFilePreprocessor(config=gamma_configuration)
    gamma_configuration['paths'] = gamma_train[1][kfold_index]
    gamma_test_preprocessor = EventFilePreprocessor(config=gamma_configuration)

    if model_type == "Separation":
        proton_configuration = {
            'rebin_size': rebin_size,
            'shape': shape,
            'paths': proton_train[0][kfold_index],
            'as_channels': as_channels
        }
        proton_train_preprocessor = EventFilePreprocessor(config=proton_configuration)
        proton_configuration['paths'] = proton_train[1][kfold_index]
        proton_validate_preprocessor = EventFilePreprocessor(config=proton_configuration)
        proton_configuration['paths'] = proton_train[1][kfold_index]
        proton_test_preprocessor = EventFilePreprocessor(config=proton_configuration)

        train = EventFileGenerator(paths=gamma_train[0][kfold_index], batch_size=batch_size,
                                   preprocessor=gamma_train_preprocessor,
                                   proton_paths=proton_train[0][kfold_index],
                                   proton_preprocessor=proton_train_preprocessor,
                                   as_channels=as_channels,
                                   final_slices=final_slices,
                                   slices=(start_slice, end_slice),
                                   augment=True,
                                   normalize=normalize,
                                   training_type=model_type)
        validate = EventFileGenerator(paths=gamma_train[1][kfold_index], batch_size=batch_size,
                                      proton_paths=proton_train[1][kfold_index],
                                      proton_preprocessor=proton_validate_preprocessor,
                                      preprocessor=gamma_validate_preprocessor,
                                      as_channels=as_channels,
                                      final_slices=final_slices,
                                      slices=(start_slice, end_slice),
                                      augment=False,
                                      normalize=normalize,
                                      training_type=model_type)
        test = EventFileGenerator(paths=gamma_train[2][kfold_index], batch_size=batch_size,
                                  proton_paths=proton_train[2][kfold_index],
                                  proton_preprocessor=proton_test_preprocessor,
                                  preprocessor=gamma_test_preprocessor,
                                  as_channels=as_channels,
                                  final_slices=final_slices,
                                  slices=(start_slice, end_slice),
                                  augment=False,
                                  normalize=normalize,
                                  training_type=model_type)
    else:
        train = EventFileGenerator(paths=gamma_train[0][kfold_index], batch_size=batch_size,
                                   preprocessor=gamma_train_preprocessor,
                                   as_channels=as_channels,
                                   final_slices=final_slices,
                                   slices=(start_slice, end_slice),
                                   augment=True,
                                   normalize=normalize,
                                   training_type=model_type)
        validate = EventFileGenerator(paths=gamma_train[1][kfold_index], batch_size=batch_size,
                                      preprocessor=gamma_validate_preprocessor,
                                      as_channels=as_channels,
                                      final_slices=final_slices,
                                      slices=(start_slice, end_slice),
                                      augment=False,
                                      normalize=normalize,
                                      training_type=model_type)
        test = EventFileGenerator(paths=gamma_train[2][kfold_index], batch_size=batch_size,
                                  preprocessor=gamma_test_preprocessor,
                                  as_channels=as_channels,
                                  final_slices=final_slices,
                                  slices=(start_slice, end_slice),
                                  augment=False,
                                  normalize=normalize,
                                  training_type=model_type)

    if as_channels:
        final_shape = (gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], final_slices)
    else:
        final_shape = (final_slices, gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], 1)
    return train, validate, test, final_shape


def fit_model(model, train_gen, val_gen, workers=10, verbose=1):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0002,
                                               patience=5,
                                               verbose=0, mode='auto',
                                               restore_best_weights=False)
    nan_stop = keras.callbacks.TerminateOnNaN()

    model.fit_generator(
        generator=train_gen,
        epochs=500,
        verbose=verbose,
        validation_data=val_gen,
        callbacks=[early_stop, nan_stop],
        use_multiprocessing=True,
        workers=workers,
        max_queue_size=50,
    )
    return model


def model_evaluate(model, test_gen, workers=10, verbose=0):
    evaluation = model.evaluate_generator(
        generator=test_gen,
        verbose=verbose,
        use_multiprocessing=True,
        workers=workers,
        max_queue_size=50,
    )
    return evaluation


def cross_validate(model, directory, proton_directory="", indicies=(30, 129, 3), rebin=50,
                   as_channels=False, kfolds=5, model_type="Separation", normalize=False, batch_size=32,
                   workers=10, verbose=1, plot=False):
    """

    :param model: Keras Model
    :param directory: Directory of Gamma events in EventFile format
    :param proton_directory: Directory of Proton events, in EventFile format
    :param indicies: In (start, end, final slices) order
    :param rebin: Rebin size
    :param as_channels: Whether to do it as channels or as time slices
    :param kfolds: Number of folds for kfold validation
    :param model_type: Type of Model, one of "Separation", "Disp", "Energy", "Sign", "COG"
    :param normalize: Whether to normalize the data cube or not
    :param batch_size: Batch size
    :param workers: Number of worker threads for the fitting and evaluation
    :param verbose: How verbose the fitting and evaluation should be
    :param plot: Whether to plot the output or not
    :return:
    """

    paths = []
    for source_dir in directory:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                paths.append(os.path.join(root, file))
    gamma_paths = split_data(paths, kfolds=kfolds)

    if model_type == "Separation":
        proton_paths = []
        for source_dir in proton_directory:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    proton_paths.append(os.path.join(root, file))
        proton_paths = split_data(proton_paths, kfolds=kfolds)

        train_gen, val_gen, test_gen, shape = data(start_slice=indicies[0], end_slice=indicies[1],
                                                   final_slices=indicies[2],
                                                   rebin_size=rebin, gamma_train=gamma_paths,
                                                   proton_train=proton_paths, batch_size=batch_size,
                                                   normalize=normalize,
                                                   model_type=model_type, as_channels=as_channels)
    else:
        train_gen, val_gen, test_gen, shape = data(start_slice=indicies[0], end_slice=indicies[1],
                                                   final_slices=indicies[2],
                                                   rebin_size=rebin, gamma_train=gamma_paths,
                                                   batch_size=batch_size, normalize=normalize,
                                                   model_type=model_type, as_channels=as_channels)

    model = fit_model(model, train_gen, val_gen, workers=workers, verbose=verbose)
    evaluation = model_evaluate(model, test_gen, workers=workers, verbose=verbose)

    return model, evaluation
