'''
A module containing the class ModelSaver, which will periodically save a Keras
model and a summary of its parameters to the model store.

Credit to William Martin, https://github.com/wjam1995/tu-dortmund-ice-cube
'''
from __future__ import print_function

import csv
import time

from keras.callbacks import ModelCheckpoint

# Constants
MODEL_DIR = '/run/media/jbieker/Seagate/A_test'    # Path to model directory
MODEL_SUMMARY = 'model_summary.csv'           # CSV summary of models

class ModelSaver(ModelCheckpoint):
    """
    A Keras callback that saves the model to the model store at regular
    intervals.
    The function saves the model when instantiated, at specified intervals
    during training, and at the end of training.
    Inherits from keras.callbacks.ModelCheckpoint
    """

    def _save_model(self):
        '''
        Function that saves the model to the filestore

        No arguments
        No returns
        '''
        self.model.save(self.filepath, overwrite=True)
        if self.verbose:
            print("Model saved to " + self.filepath)

    def _write_summary(self, model_name, params, comment):
        '''
        Function that writes the parameters of the model to the summary file
        Arguments:
            model_name - the filename of the model
            params - a dictionary of the parameters used during training, of
                     the following format:
                lr - the learning rate of the Adam optimiser
                conv_dr - the dropout rate after the convolutional layers
                fc_dr - the dropout rate after the fully-connected layers
                no_epochs - the number of epochs to run for
                steps_per_epoch - the number of batches in each epoch
                dp_prob - the proportion of double pulse waveforms shown at
                          train time
                batch_norm - if True, use batch norm after each layer
                regularise - if True, uses L2 regularisation on the weights for
                             each layer
                decay - the amount of decay in class balance during training
            comment - a string that is written to the csv file (for, e.g.,
                      observations about the model)
        No returns
        '''
        # Update CSV file
        with open(MODEL_DIR + MODEL_SUMMARY, 'a+') as csvfile:
            # Read file to extract headers
            reader = csv.DictReader(csvfile)
            fields = reader.fieldnames

            # Add model_name to parameter dictionary
            # Replace all commas, forward slashes with underscores
            params['model_name'] = \
                model_name.replace(",", "_").replace("/", "_")

            # Add comment to parameter dictionary
            # Replace all commas with semicolons
            params['comments'] = comment.replace(",", ";")

            # Write param dictionary to file
            writer = csv.DictWriter(csvfile, fields)
            writer.writerow(params)

        if self.verbose:
            print("Model added to summary")

    def __init__(self, model, nn_str, params,
                 comment="", period=100, verbose=False):
        '''
        Class constructor for ModelSaver
        Instantiates a model saver object, saves the model initially, and
        writes the parameters of the model to the model summary file
        Arguments:
            model - the Keras model to be saved
            nn_str - the string specifying what type of model was trained
                     (e.g. "cnn", "snn", etc.)
                     NB used for the filename - no commas or forward-slashes!
            params - the parameter dictionary for the model, containing:
                lr - the learning rate of the Adam optimiser
                conv_dr - the dropout rate after the convolutional layers
                fc_dr - the dropout rate after the fully-connected layers
                no_epochs - the number of epochs to run for
                steps_per_epoch - the number of batches in each epoch
                dp_prob - the proportion of double pulse waveforms shown at
                          train time
                batch_norm - if True, use batch norm after each layer
                regularise - if True, uses L2 regularisation on the weights for
                             each layer
                decay - the amount of decay in class balance during training
            comment - string to put in comments field in CSV (default: "")
                      NB commas will be replaced with semicolons!
            period - number of epochs between checkpoint saves (default: 100)
            verbose - if True, prints when files are saved (default: False)
        Returns:
            model_saver - a ModelSaver object
        '''
        # Create savepath
        datetime = time.strftime("%Y%m%d_%H%M%S_")
        model_name = datetime + nn_str + 'Keras.h5'
        filepath = MODEL_DIR + model_name

        # Initialise the parent class ModelSaver
        super(ModelSaver, self).__init__(
            filepath=filepath,
            verbose=int(verbose),
            period=period
        )

        # Set, save model
        self.set_model(model)
        self._save_model()

        # Update CSV file listing models
        self._write_summary(model_name, params, comment)

    def on_train_end(self, logs=None):
        '''
        Saves model on the end of training (called when training ends)
        Inherited from keras.callbacks.ModelCheckpoint
        Arguments:
            logs - unused but required for inheritance reasons
        No returns
        '''
        self._save_model()