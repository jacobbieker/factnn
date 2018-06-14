import h5py

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

        if len(f.items())==0:
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

        if config['conv_dropout']:
            self.conv_dropout = config['conv_dropout']
        else:
            self.conv_dropout = 0.0

        if config['lstm_dropout']:
            self.lstm_dropout = config['lstm_dropout']
        else:
            self.lstm_dropout = 0.0

        if config['fc_dropout']:
            self.fc_dropout = config['fc_dropout']
        else:
            self.fc_dropout = 0.0

        if config['num_conv3d']:
            self.num_conv3d = config['num_conv3d']
        else:
            self.num_conv3d = 0

        if config['kernel_conv3d']:
            self.kernel_conv3d = config['kernel_conv3d']
        else:
            self.kernel_conv3d = 1

        if config['strides_conv3d']:
            self.strides_conv3d = config['strides_conv3d']
        else:
            self.strides_conv3d = 1

        if config['num_lstm']:
            self.num_lstm = config['num_lstm']
        else:
            self.num_lstm = 0

        if config['kernel_lstm']:
            self.kernel_lstm = config['kernel_lstm']
        else:
            self.kernel_lstm = 1

        if config['strides_lstm']:
            self.strides_lstm = config['strides_lstm']
        else:
            self.strides_lstm = 1

        if config['num_fc']:
            self.num_fc = config['num_fc']
        else:
            self.num_fc = 0

        if config['seed']:
            self.seed = config['seed']

        if config['pooling']:
            self.pooling = config['pooling']

        self.neurons = config['neurons']
        self.shape = config['shape']
        self.end_slice = config['end_slice']
        self.number_slices = config['number_slices']
        self.batch_normalization = config['batch_normalization']
        self.activation = config['activation']
        self.model_type = None
        self.auc = None
        self.r2 = None
        self.model = None

        if config['learning_rate']:
            self.learning_rate = config['learning_rate']

        if config['verbose']:
            self.verbose = config['verbose']

        if config['augment']:
            self.augment = config['augment']

        if config['name']:
            self.name = config['name']
        else:
            self.name = None

        self.init()


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
        return NotImplemented

    def train(self):
        '''
        Train model
        :return:
        '''
        return NotImplemented

    def save(self):
        """
        Save the model, separate from the Keras training model save
        :return:
        """
        return NotImplemented

    def __str__(self):
        return NotImplemented

    def __repr__(self):
        return NotImplemented

