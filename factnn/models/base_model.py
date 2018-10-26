import h5py
import keras
import numpy as np


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items()) == 0:
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()


class BaseModel(object):

    def __init__(self, config):
        '''
        Base model for other models to use
        :param config: Dictionary of values for the given model
        '''

        if 'conv_dropout' in config:
            self.conv_dropout = config['conv_dropout']
        else:
            self.conv_dropout = 0.0

        if 'lstm_dropout' in config:
            self.lstm_dropout = config['lstm_dropout']
        else:
            self.lstm_dropout = 0.0

        if 'fc_dropout' in config:
            self.fc_dropout = config['fc_dropout']
        else:
            self.fc_dropout = 0.0

        if 'num_conv3d' in config:
            self.num_conv3d = config['num_conv3d']
        else:
            self.num_conv3d = 0

        if 'kernel_conv3d' in config:
            self.kernel_conv3d = config['kernel_conv3d']
        else:
            self.kernel_conv3d = 1

        if 'strides_conv3d' in config:
            self.strides_conv3d = config['strides_conv3d']
        else:
            self.strides_conv3d = 1

        if 'num_lstm' in config:
            self.num_lstm = config['num_lstm']
        else:
            self.num_lstm = 0

        if 'kernel_lstm' in config:
            self.kernel_lstm = config['kernel_lstm']
        else:
            self.kernel_lstm = 1

        if 'strides_lstm' in config:
            self.strides_lstm = config['strides_lstm']
        else:
            self.strides_lstm = 1

        if 'num_fc' in config:
            self.num_fc = config['num_fc']
        else:
            self.num_fc = 0

        if 'seed' in config:
            self.seed = config['seed']

        if 'pooling' in config:
            self.pooling = config['pooling']

        self.neurons = config['neurons']
        self.shape = config['shape']
        self.start_slice = config['start_slice']
        self.number_slices = config['number_slices']
        if 'batch_normalization' in config:
            self.batch_normalization = config['batch_normalization']
        else:
            self.batch_normalization = False
        self.activation = config['activation']
        self.model_type = None
        self.auc = None
        self.r2 = None
        self.model = None
        self.train_generator = None
        self.validate_generator = None
        self.test_generator = None

        if 'epochs' in config:
            self.epochs = config['epochs']
        else:
            self.epochs = 500

        if 'patience' in config:
            self.patience = config['patience']
        else:
            self.patience = 10

        if 'learning_rate' in config:
            self.learning_rate = config['learning_rate']

        if 'verbose' in config:
            self.verbose = config['verbose']

        if 'augment' in config:
            self.augment = config['augment']

        if 'name' in config:
            self.name = config['name']
        else:
            self.name = None

        self.init()
        self.create()

    def init(self):
        '''
        Model specific inits are here, such as r2 or auc default values
        :return:
        '''
        return NotImplemented

    def create(self):
        '''
        Creates the Keras model
        :return:
        '''
        return NotImplemented

    def apply(self):
        '''
        Apply model to given data set
        :return:
        '''
        if isinstance(self.test_generator.items, int):
            num_events = int(self.test_generator.items * self.test_generator.test_fraction)
        else:
            num_events = int(len(self.test_generator.items) * self.train_generator.test_fraction)
        steps = int(np.floor(num_events / self.test_generator.batch_size))
        truth = []
        predictions = []
        for i in range(steps):
            # Get each batch and test it
            test_images, test_labels = next(self.test_generator)
            test_predictions = self.model.predict_on_batch(test_images)
            predictions.append(test_predictions)
            truth.append(test_labels)

        predictions = np.asarray(predictions).reshape(-1, )
        truth = np.asarray(truth).reshape(-1, )

        return (predictions, truth)

    def train(self):
        '''
        Train model
        :return:
        '''
        model_checkpoint = keras.callbacks.ModelCheckpoint(self.name,
                                                           monitor='val_loss',
                                                           verbose=0,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='auto', period=1)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                   patience=self.patience,
                                                   verbose=0, mode='auto')
        if isinstance(self.train_generator.items, int):
            num_events = int(self.train_generator.items * self.train_generator.train_fraction)
            val_num = int(self.train_generator.items * self.train_generator.validate_fraction)
        else:
            num_events = int(len(self.train_generator.items) * self.train_generator.train_fraction)
            val_num = int(len(self.train_generator.items) * self.train_generator.validate_fraction)

        self.model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=int(np.floor(num_events / self.train_generator.batch_size)),
            epochs=self.epochs,
            verbose=1,
            validation_data=self.validate_generator,
            callbacks=[early_stop, model_checkpoint],
            validation_steps=int(np.floor(val_num / self.validate_generator.batch_size))
        )

    def save(self):
        """
        Save the model, separate from the Keras training model save
        :return:
        """
        return NotImplemented

    def __str__(self):
        self.model.summary()
        return ""

    def __repr__(self):
        return NotImplemented
