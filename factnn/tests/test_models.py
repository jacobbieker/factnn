import unittest
import tensorflow as tf
import keras.backend as K
from factnn.models.separation_models import SeparationModel
from factnn.models.energy_models import EnergyModel
from factnn.models.source_models import DispModel


class TestSeparationModels(unittest.TestCase):
    def setUp(self):
        self.configuration = {
            'conv_dropout': 0.3,
            'lstm_dropout': 0.4,
            'fc_dropout': 0.5,
            'num_conv3d': 2,
            'kernel_conv3d': 2,
            'strides_conv3d': 1,
            'num_lstm': 2,
            'kernel_lstm': 2,
            'strides_lstm': 2,
            'num_fc': 2,
            'pooling': True,
            'neurons': [32, 16, 8, 16, 32, 48],
            'shape': [100, 75, 75, 1],
            'end_slice': 35,
            'number_slices': 24,
            'activation': 'relu',
            'name': 'testModel',
        }

    def tearDown(self):
        K.clear_session()
        tf.reset_default_graph()

    def test_config(self):
        model = SeparationModel(config=self.configuration)
        self.assertEqual(model.number_slices, 24)
        self.assertEqual(model.num_lstm, 2)
        self.assertEqual(model.num_fc, 2)
        self.assertEqual(model.num_conv3d, 2)
        self.assertEqual(model.epochs, 500)
        self.assertEqual(model.patience, 10)
        self.assertEqual(model.name, 'testModel')
        self.assertEqual(model.activation, 'relu')
        self.assertEqual(model.model_type, 'Separation')

    def test_create_model(self):
        model = SeparationModel(config=self.configuration)
        test_model = model.model
        layers = test_model.layers

        self.assertEqual(len(layers), 7)
        self.assertEqual(test_model.input_shape, [100, 75, 75, 1])


class TestEnergyModels(unittest.TestCase):
    def setUp(self):
        self.configuration = {
            'conv_dropout': 0.3,
            'lstm_dropout': 0.4,
            'fc_dropout': 0.5,
            'num_conv3d': 2,
            'kernel_conv3d': 2,
            'strides_conv3d': 1,
            'num_lstm': 2,
            'kernel_lstm': 2,
            'strides_lstm': 2,
            'num_fc': 2,
            'pooling': True,
            'neurons': [32, 16, 8, 16, 32, 48],
            'shape': [100, 75, 75, 1],
            'end_slice': 35,
            'number_slices': 24,
            'activation': 'relu',
            'name': 'testModel',
        }

    def tearDown(self):
        K.clear_session()
        tf.reset_default_graph()

    def test_config(self):
        model = EnergyModel(config=self.configuration)
        self.assertEqual(model.number_slices, 24)
        self.assertEqual(model.num_lstm, 2)
        self.assertEqual(model.num_fc, 2)
        self.assertEqual(model.num_conv3d, 2)
        self.assertEqual(model.epochs, 500)
        self.assertEqual(model.patience, 10)
        self.assertEqual(model.name, 'testModel')
        self.assertEqual(model.activation, 'relu')
        self.assertEqual(model.model_type, 'Energy')

    def test_create_model(self):
        model = SeparationModel(config=self.configuration)
        test_model = model.model
        layers = test_model.layers

        self.assertEqual(len(layers), 7)
        self.assertEqual(test_model.input_shape, [100, 75, 75, 1])


class TestDispModels(unittest.TestCase):
    def setUp(self):
        self.configuration = {
            'conv_dropout': 0.3,
            'lstm_dropout': 0.4,
            'fc_dropout': 0.5,
            'num_conv3d': 2,
            'kernel_conv3d': 2,
            'strides_conv3d': 1,
            'num_lstm': 2,
            'kernel_lstm': 2,
            'strides_lstm': 2,
            'num_fc': 2,
            'pooling': True,
            'neurons': [32, 16, 8, 16, 32, 48],
            'shape': [100, 75, 75, 1],
            'end_slice': 35,
            'number_slices': 24,
            'activation': 'relu',
            'name': 'testModel',
        }

    def tearDown(self):
        K.clear_session()
        tf.reset_default_graph()

    def test_config(self):
        model = DispModel(config=self.configuration)
        self.assertEqual(model.number_slices, 24)
        self.assertEqual(model.num_lstm, 2)
        self.assertEqual(model.num_fc, 2)
        self.assertEqual(model.num_conv3d, 2)
        self.assertEqual(model.epochs, 500)
        self.assertEqual(model.patience, 10)
        self.assertEqual(model.name, 'testModel')
        self.assertEqual(model.activation, 'relu')
        self.assertEqual(model.model_type, 'Disp')

    def test_create_model(self):
        model = SeparationModel(config=self.configuration)
        test_model = model.model
        layers = test_model.layers

        self.assertEqual(len(layers), 7)
        self.assertEqual(test_model.input_shape, [100, 75, 75, 1])
