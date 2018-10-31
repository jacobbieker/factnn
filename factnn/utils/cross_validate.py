import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle


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
    kf = KFold(n_splits=kfolds)

    for train_index, test_index in kf.split(indicies):
        # Need to split train_index into validation data
        train_data, validate_data = train_test_split(indicies[train_index], train_size=(1.0 - validate_fraction),
                                                     test_size=validate_fraction)
        list_of_training.append(train_data)
        list_of_validate.append(validate_data)
        list_of_testing.append(indicies[test_index])

    # Now convert to a numpy array
    return list_of_training, list_of_validate, list_of_testing


def cross_validate(model, data, generator, proton_data=None, proton_generator=None, kfolds=5, preprocessor=None, proton_preprocessor=None,
                   plot=False):
    """
    Performs a k-fold cross validation on a given model and data
    :param proton_preprocessor: Preprocessor for the proton data, if used, default is None
    :param preprocessor: Preprocessor for gamma data, if used, usually for streaming files, default None
    :param model: Keras Model instance to train
    :param generator: Generator for Keras fit_generator
    :param proton_generator: Generator for Keras fit_generator if using proton data, default None
    :param kfolds: Number of folds to do for the k-fold validation
    :param plot: Whether to plot the output
    :return:
    """




    return NotImplementedError
