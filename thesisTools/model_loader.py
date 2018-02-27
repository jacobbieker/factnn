'''
A module that loads a model from the model store into memory

Credit to William Martin, https://github.com/wjam1995/tu-dortmund-ice-cube
'''
from keras.models import load_model as _load_model

# Constants

def load_model(filepath):
    '''
    Loads and compiles the specified Keras model.
    NB Model must have been pre-compiled with an optimiser
    Arguments:
        filepath - the filepath of the model
    Returns:
        model - a keras.models.Sequential object
    '''
    return _load_model(filepath)

def load_uncompiled_model(filepath):
    '''
    Loads the specified Keras model without compiling it.
    NB Model can be either compiled or uncompiled.
    Arguments:
        filepath - the filepath of the model
    Returns:
        model - a keras.models.Sequential object
    '''
    return _load_model(filepath, compile=False)