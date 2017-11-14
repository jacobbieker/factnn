'''
A module containing metrics defined for use with Keras at train-time - takes
tf.Tensor objects.

Taken from William's RISE project
'''
from __future__ import absolute_import, division
from keras import backend as K

# From https://github.com/fchollet/keras/issues/5400
# Adapted to actually make it work
def precision(y_true, y_pred):
    """
    Computes batch-wise precision
    For use on tf.Tensor objects
    Arguments:
        y_true - a tf.Tensor object containing the true labels
        y_pred - a tf.Tensor object containing the predicted labels
    Returns:
        precision - a float, the batch-wise precision
    """
    true_positives = K.sum(K.round(y_true[:, 1] * y_pred[:, 1]))
    predicted_positives = K.sum(K.round(y_pred[:, 1]))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# From https://github.com/fchollet/keras/issues/5400
# Adapted to make it work
def recall(y_true, y_pred):
    """
    Computes batch-wise recall
    For use on tf.Tensor objects
    Arguments:
        y_true - a tf.Tensor object containing the true labels
        y_pred - a tf.Tensor object containing the predicted labels
    Returns:
        recall - a float, the batch-wise recall
    """
    true_positives = K.sum(K.round(y_true[:, 1] * y_pred[:, 1]))
    possible_positives = K.sum(K.round(y_true[:, 1]))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# From https://github.com/fchollet/keras/issues/5400
def f1(y_true, y_pred):
    """
    Computes batch-wise F1 score
    For use on tf.Tensor objects
    Arguments:
        y_true - a tf.Tensor object containing the true labels
        y_pred - a tf.Tensor object containing the predicted labels
    Returns:
        f1 - a float, the batch-wise F1 score
    """
    prc = precision(y_true, y_pred)
    rcl = recall(y_true, y_pred)
    return 2*((prc*rcl)/(prc+rcl))

def class_balance(y_true, y_pred):
    """
    Computes the balance between the classes in the batch
    For use on tf.Tensor objects
    Arguments:
        y_true - a tf.Tensor object containing the true labels
        y_pred - unused but required for a metric
    Returns:
        balance - a float, the fraction of double-pulse waveforms in the batch
    """
    return K.mean(y_true[:, 1])