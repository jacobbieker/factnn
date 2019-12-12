import os

import numpy as np
import tensorflow.keras
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

from ..data.preprocess.eventfile_preprocessor import EventFilePreprocessor
from ..generator.keras.eventfile_generator import EventFileGenerator


def split_data(indicies, kfolds, seed=None):
    """
    Splits the data into the indicies for kfold validation
    :param indicies: Int or Array, if int, assumed to be the size of the dataset
    :param kfolds:  Number of folds to do
    :return: Array (3, kfolds, variable) that holds the indicies for the training, validation, and testing
    """

    if type(indicies) is int:
        indicies = np.arange(0, indicies)

    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)

    # Now split into kfolds,
    list_of_training = []
    list_of_validate = []
    list_of_testing = []
    validate_fraction = 0.2
    indicies = np.asarray(indicies)
    # Now get KFOLD splits
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

    for train_index, test_index in kf.split(indicies):
        # Need to split train_index into validation data
        train_data, validate_data = train_test_split(indicies[train_index],
                                                     test_size=validate_fraction, random_state=seed, shuffle=True)
        list_of_training.append(train_data)
        list_of_validate.append(validate_data)
        list_of_testing.append(indicies[test_index])

    # Now convert to a numpy array
    return list_of_training, list_of_validate, list_of_testing


def data(start_slice, end_slice, final_slices, rebin_size, gamma_train, proton_train=None, model_type="Separation",
         batch_size=8, as_channels=True, normalize=False, kfold_index=0, truncate=True, dynamic_resize=True, equal_slices=False,
         return_collapsed=False, return_features=False):
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
                                   training_type=model_type,
                                   truncate=truncate,
                                   dynamic_resize=dynamic_resize,
                                   equal_slices=equal_slices,
                                   return_collapsed=return_collapsed,
                                   return_features=return_features)
        validate = EventFileGenerator(paths=gamma_train[1][kfold_index], batch_size=batch_size,
                                      proton_paths=proton_train[1][kfold_index],
                                      proton_preprocessor=proton_validate_preprocessor,
                                      preprocessor=gamma_validate_preprocessor,
                                      as_channels=as_channels,
                                      final_slices=final_slices,
                                      slices=(start_slice, end_slice),
                                      augment=False,
                                      normalize=normalize,
                                      training_type=model_type,
                                      truncate=truncate,
                                      dynamic_resize=dynamic_resize,
                                      equal_slices=equal_slices,
                                      return_collapsed=return_collapsed,
                                      return_features=return_features)
        test = EventFileGenerator(paths=gamma_train[2][kfold_index], batch_size=batch_size,
                                  proton_paths=proton_train[2][kfold_index],
                                  proton_preprocessor=proton_test_preprocessor,
                                  preprocessor=gamma_test_preprocessor,
                                  as_channels=as_channels,
                                  final_slices=final_slices,
                                  slices=(start_slice, end_slice),
                                  augment=False,
                                  normalize=normalize,
                                  training_type=model_type,
                                  truncate=truncate,
                                  dynamic_resize=dynamic_resize,
                                  equal_slices=equal_slices,
                                  return_collapsed=return_collapsed,
                                  return_features=return_features)
    else:
        train = EventFileGenerator(paths=gamma_train[0][kfold_index], batch_size=batch_size,
                                   preprocessor=gamma_train_preprocessor,
                                   as_channels=as_channels,
                                   final_slices=final_slices,
                                   slices=(start_slice, end_slice),
                                   augment=True,
                                   normalize=normalize,
                                   training_type=model_type,
                                   truncate=truncate,
                                   dynamic_resize=dynamic_resize,
                                   equal_slices=equal_slices,
                                   return_collapsed=return_collapsed,
                                   return_features=return_features)
        validate = EventFileGenerator(paths=gamma_train[1][kfold_index], batch_size=batch_size,
                                      preprocessor=gamma_validate_preprocessor,
                                      as_channels=as_channels,
                                      final_slices=final_slices,
                                      slices=(start_slice, end_slice),
                                      augment=False,
                                      normalize=normalize,
                                      training_type=model_type,
                                      truncate=truncate,
                                      dynamic_resize=dynamic_resize,
                                      equal_slices=equal_slices,
                                      return_collapsed=return_collapsed,
                                      return_features=return_features)
        test = EventFileGenerator(paths=gamma_train[2][kfold_index], batch_size=batch_size,
                                  preprocessor=gamma_test_preprocessor,
                                  as_channels=as_channels,
                                  final_slices=final_slices,
                                  slices=(start_slice, end_slice),
                                  augment=False,
                                  normalize=normalize,
                                  training_type=model_type,
                                  truncate=truncate,
                                  dynamic_resize=dynamic_resize,
                                  equal_slices=equal_slices,
                                  return_collapsed=return_collapsed,
                                  return_features=return_features)

    if as_channels:
        final_shape = (gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], final_slices)
    else:
        final_shape = (final_slices, gamma_train_preprocessor.shape[1], gamma_train_preprocessor.shape[2], 1)
    return train, validate, test, final_shape


def fit_model(model, train_gen, val_gen, workers=10, verbose=1):
    early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0002,
                                                          patience=5,
                                                          verbose=0, mode='auto',
                                                          restore_best_weights=True)
    nan_stop = tensorflow.keras.callbacks.TerminateOnNaN()

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


def get_data_generators(directory, proton_directory="", indicies=(30,129,3),
                      rebin=50, as_channels=False, model_type="Separation",
                      normalize=False, batch_size=32, truncate=True,
                      dynamic_resize=True, equal_slices=False, seed=1337, max_elements=None,
                      return_collapsed=False, return_features=False, kfolds=5):
    """
    This is to obtain just the generators, for when cross-validation not needed
    Keeps 20% of the files out by default
    :param directory:
    :param proton_directory:
    :param indicies:
    :param rebin:
    :param as_channels:
    :param model_type:
    :param normalize:
    :param truncate:
    :param dynamic_resize:
    :param equal_slices:
    :param seed:
    :param max_elements:
    :param return_collapsed:
    :param return_features:
    :return: Returns images and labels, to be split with the Keras validation split
    """

    paths = []
    for source_dir in directory:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                paths.append(os.path.join(root, file))
    if max_elements is not None:
        paths = shuffle(paths)
        paths = paths[0:max_elements]
    gamma_paths = split_data(paths, kfolds=kfolds, seed=seed)

    if model_type == "Separation":
        proton_paths = []
        for source_dir in proton_directory:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    proton_paths.append(os.path.join(root, file))
        if max_elements is not None:
            proton_paths = shuffle(proton_paths)
            proton_paths = proton_paths[0:max_elements]
        proton_paths = split_data(proton_paths, kfolds=kfolds, seed=seed)

    if model_type == "Separation":
        train_gen, val_gen, test_gen, shape = data(start_slice=indicies[0], end_slice=indicies[1],
                                                   final_slices=indicies[2],
                                                   rebin_size=rebin, gamma_train=gamma_paths,
                                                   proton_train=proton_paths, batch_size=batch_size,
                                                   normalize=normalize,
                                                   model_type=model_type, as_channels=as_channels,
                                                   truncate=truncate,
                                                   dynamic_resize=dynamic_resize,
                                                   equal_slices=equal_slices,
                                                   return_collapsed=return_collapsed,
                                                   return_features=return_features)
    else:
        train_gen, val_gen, test_gen, shape = data(start_slice=indicies[0], end_slice=indicies[1],
                                                   final_slices=indicies[2],
                                                   rebin_size=rebin, gamma_train=gamma_paths,
                                                   batch_size=batch_size, normalize=normalize,
                                                   model_type=model_type, as_channels=as_channels,
                                                   truncate=truncate,
                                                   dynamic_resize=dynamic_resize,
                                                   equal_slices=equal_slices,
                                                   return_collapsed=return_collapsed,
                                                   return_features=return_features)

    return train_gen, val_gen, test_gen, shape


def get_chunk_of_data(directory, proton_directory="", indicies=(30,129,3),
                      rebin=50, as_channels=False, model_type="Separation",
                      normalize=False, chunk_size=1000, truncate=True,
                      dynamic_resize=True, equal_slices=False, seed=1337, max_elements=None,
                      return_collapsed=False, return_features=False):
    """
    This is to obtain a single chunk of data, for situations where generators should not be used, only need a single block of data

    :param directory:
    :param proton_directory:
    :param indicies:
    :param rebin:
    :param as_channels:
    :param model_type:
    :param normalize:
    :param chunk_size:
    :param truncate:
    :param dynamic_resize:
    :param equal_slices:
    :param seed:
    :param max_elements:
    :param return_collapsed:
    :param return_features:
    :return: Returns images and labels, to be split with the Keras validation split
    """

    paths = []
    for source_dir in directory:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                paths.append(os.path.join(root, file))
    if max_elements is not None:
        paths = shuffle(paths)
        paths = paths[0:max_elements]
    gamma_paths = split_data(paths, kfolds=2, seed=seed)

    if model_type == "Separation":
        proton_paths = []
        for source_dir in proton_directory:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    proton_paths.append(os.path.join(root, file))
        if max_elements is not None:
            proton_paths = shuffle(proton_paths)
            proton_paths = proton_paths[0:max_elements]
        proton_paths = split_data(proton_paths, kfolds=2, seed=seed)

    if model_type == "Separation":
        train_gen, val_gen, test_gen, shape = data(start_slice=indicies[0], end_slice=indicies[1],
                                                   final_slices=indicies[2],
                                                   rebin_size=rebin, gamma_train=gamma_paths,
                                                   proton_train=proton_paths, batch_size=1,
                                                   normalize=normalize,
                                                   model_type=model_type, as_channels=as_channels,
                                                   truncate=truncate,
                                                   dynamic_resize=dynamic_resize,
                                                   equal_slices=equal_slices,
                                                   return_collapsed=return_collapsed,
                                                   return_features=return_features)
    else:
        train_gen, val_gen, test_gen, shape = data(start_slice=indicies[0], end_slice=indicies[1],
                                                   final_slices=indicies[2],
                                                   rebin_size=rebin, gamma_train=gamma_paths,
                                                   batch_size=1, normalize=normalize,
                                                   model_type=model_type, as_channels=as_channels,
                                                   truncate=truncate,
                                                   dynamic_resize=dynamic_resize,
                                                   equal_slices=equal_slices,
                                                   return_collapsed=return_collapsed,
                                                   return_features=return_features)

    x, y = train_gen.__getitem__(0)
    for i in range(1, chunk_size):
        x_temp, y_temp = train_gen.__getitem__(i)
        x = np.append(x, x_temp, axis=0)
        y = np.append(y, y_temp, axis=0)

    return x, y

def cross_validate(model, directory, proton_directory="", indicies=(30, 129, 3), rebin=50,
                   as_channels=False, kfolds=5, model_type="Separation", normalize=False, batch_size=32,
                   workers=10, verbose=1, truncate=True, dynamic_resize=True, equal_slices=False, seed=1337, max_elements=None,
                   return_collapsed=False, return_features=False, plot=False):
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
    if max_elements is not None:
        paths = shuffle(paths)
        paths = paths[0:max_elements]
    gamma_paths = split_data(paths, kfolds=kfolds, seed=seed)

    if model_type == "Separation":
        proton_paths = []
        for source_dir in proton_directory:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    proton_paths.append(os.path.join(root, file))
        if max_elements is not None:
            proton_paths = shuffle(proton_paths)
            proton_paths = proton_paths[0:max_elements]
        proton_paths = split_data(proton_paths, kfolds=kfolds, seed=seed)

    evaluations = []
    # Save default weights for reuse
    model.save_weights("cv_default.h5")
    for i in range(kfolds):
        model.load_weights("cv_default.h5")
        if model_type == "Separation":
            train_gen, val_gen, test_gen, shape = data(start_slice=indicies[0], end_slice=indicies[1],
                                                       final_slices=indicies[2],
                                                       rebin_size=rebin, gamma_train=gamma_paths,
                                                       proton_train=proton_paths, batch_size=batch_size,
                                                       normalize=normalize,
                                                       model_type=model_type, as_channels=as_channels,
                                                       truncate=truncate,
                                                       dynamic_resize=dynamic_resize,
                                                       equal_slices=equal_slices,
                                                       return_collapsed=return_collapsed,
                                                       return_features=return_features)
        else:
            train_gen, val_gen, test_gen, shape = data(start_slice=indicies[0], end_slice=indicies[1],
                                                       final_slices=indicies[2],
                                                       rebin_size=rebin, gamma_train=gamma_paths,
                                                       batch_size=batch_size, normalize=normalize,
                                                       model_type=model_type, as_channels=as_channels,
                                                       truncate=truncate,
                                                       dynamic_resize=dynamic_resize,
                                                       equal_slices=equal_slices,
                                                       return_collapsed=return_collapsed,
                                                       return_features=return_features)

        model = fit_model(model, train_gen, val_gen, workers=workers, verbose=verbose)
        evaluation = model_evaluate(model, test_gen, workers=workers, verbose=verbose)
        print("Evaluation: " + str(evaluation))
        evaluations.append(evaluation)

    return model, evaluations
