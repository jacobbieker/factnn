import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
"""
This holds the functions needed for k-fold validation
"""

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
            seeds.append(np.random.randint(0,2**32-1))

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
        train_data, validate_data = train_test_split(indicies[train_index], train_size=(1.0-validate_fraction), test_size=validate_fraction)
        list_of_training.append(train_data)
        list_of_validate.append(validate_data)
        list_of_testing.append(indicies[test_index])

    # Now convert to a numpy array
    return list_of_training, list_of_validate, list_of_testing


def perform_kfold(indicies):
    """
    Try to have it do the K-Fold validation on its own with the generator inputs and models configs
    Have to do a lot more work here for it. Could be easier just to have it with a reset model in the models base
    :param indicies:
    :return:
    """
    return NotImplementedError