import unittest
from factnn.generator.generator.energy_generators import EnergyGenerator
from factnn.generator.generator.separation_generators import SeparationGenerator
from factnn.generator.generator.source_generators import DispGenerator, SignGenerator


class TestEnergyGenerator(unittest.TestCase):

    def setUp(self):
        self.configuration = {
            'seed': 1337,
            'batch_size': 4,
            'input': None,
            'labels': None,
            'end_slice': 40,
            'number_slices': 35,
            'train_fraction': 0.6,
            'validate_fraction': 0.2,
            'mode': 'train',
            'samples': 10,

        }

    def test_config(self):
        self.configuration['chunked'] = True
        self.configuration['augment'] = False

        generator = EnergyGenerator(config=self.configuration)

        self.assertEqual(generator.type_gen, 'Energy')
        self.assertEqual(generator.augment, False)
        self.assertEqual(generator.chunked, True)
        self.assertEqual(generator.end_slice, 40)
        self.assertEqual(generator.mode, 'train')
        self.assertEqual(generator.items, 10)
        self.assertEqual(generator.number_slices, 35)
        self.assertEqual(generator.batch_size, 4)
        self.assertEqual(generator.train_fraction, 0.6)
        self.assertEqual(generator.validate_fraction, 0.2)

    def test_chunked(self):
        self.configuration['chunked'] = True
        self.configuration['augment'] = False
        return NotImplemented

    def test_random(self):
        self.configuration['chunked'] = False
        self.configuration['augment'] = False
        return NotImplemented

    def test_augment(self):
        self.configuration['augment'] = True
        return NotImplemented


class TestSeparationGenerator(unittest.TestCase):

    def setUp(self):
        self.configuration = {
            'seed': 1337,
            'batch_size': 4,
            'input': None,
            'second_input': None,
            'end_slice': 40,
            'number_slices': 35,
            'train_fraction': 0.6,
            'validate_fraction': 0.2,
            'mode': 'train',
            'samples': 10,

        }

    def test_config(self):
        self.configuration['chunked'] = True
        self.configuration['augment'] = False

        generator = SeparationGenerator(config=self.configuration)

        self.assertEqual(generator.type_gen, 'Separation')
        self.assertEqual(generator.augment, False)
        self.assertEqual(generator.chunked, True)
        self.assertEqual(generator.end_slice, 40)
        self.assertEqual(generator.mode, 'train')
        self.assertEqual(generator.items, 10)
        self.assertEqual(generator.number_slices, 35)
        self.assertEqual(generator.batch_size, 4)
        self.assertEqual(generator.train_fraction, 0.6)
        self.assertEqual(generator.validate_fraction, 0.2)

    def test_chunked(self):
        self.configuration['chunked'] = True
        self.configuration['augment'] = False
        return NotImplemented

    def test_random(self):
        self.configuration['chunked'] = False
        self.configuration['augment'] = False
        return NotImplemented

    def test_augment(self):
        self.configuration['augment'] = True
        return NotImplemented


class TestDispGenerator(unittest.TestCase):

    def setUp(self):
        self.configuration = {
            'seed': 1337,
            'batch_size': 4,
            'input': None,
            'end_slice': 40,
            'number_slices': 35,
            'train_fraction': 0.6,
            'validate_fraction': 0.2,
            'mode': 'train',
            'samples': 10,

        }

    def test_config(self):
        self.configuration['chunked'] = True
        self.configuration['augment'] = False

        generator = DispGenerator(config=self.configuration)

        self.assertEqual(generator.type_gen, 'Disp')
        self.assertEqual(generator.augment, False)
        self.assertEqual(generator.chunked, True)
        self.assertEqual(generator.end_slice, 40)
        self.assertEqual(generator.mode, 'train')
        self.assertEqual(generator.items, 10)
        self.assertEqual(generator.number_slices, 35)
        self.assertEqual(generator.batch_size, 4)
        self.assertEqual(generator.train_fraction, 0.6)
        self.assertEqual(generator.validate_fraction, 0.2)

    def test_chunked(self):
        self.configuration['chunked'] = True
        self.configuration['augment'] = False
        return NotImplemented

    def test_random(self):
        self.configuration['chunked'] = False
        self.configuration['augment'] = False
        return NotImplemented

    def test_augment(self):
        self.configuration['augment'] = True
        return NotImplemented


class TestSignGenerator(unittest.TestCase):

    def setUp(self):
        self.configuration = {
            'seed': 1337,
            'batch_size': 4,
            'input': None,
            'end_slice': 40,
            'number_slices': 35,
            'train_fraction': 0.6,
            'validate_fraction': 0.2,
            'mode': 'train',
            'samples': 10,

        }

    def test_config(self):
        self.configuration['chunked'] = True
        self.configuration['augment'] = False

        generator = SignGenerator(config=self.configuration)

        self.assertEqual(generator.type_gen, 'Sign')
        self.assertEqual(generator.augment, False)
        self.assertEqual(generator.chunked, True)
        self.assertEqual(generator.end_slice, 40)
        self.assertEqual(generator.mode, 'train')
        self.assertEqual(generator.items, 10)
        self.assertEqual(generator.number_slices, 35)
        self.assertEqual(generator.batch_size, 4)
        self.assertEqual(generator.train_fraction, 0.6)
        self.assertEqual(generator.validate_fraction, 0.2)

    def test_chunked(self):
        self.configuration['chunked'] = True
        self.configuration['augment'] = False
        return NotImplemented

    def test_random(self):
        self.configuration['chunked'] = False
        self.configuration['augment'] = False
        return NotImplemented

    def test_augment(self):
        self.configuration['augment'] = True
        return NotImplemented


if __name__ == '__main__':
    unittest.main()
