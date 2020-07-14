import unittest

from factnn.data.preprocess.simulation_preprocessors import GammaPreprocessor


class TestProtonPreprocessor(unittest.TestCase):
    def setUp(self):
        self.configuration = {
            "directories": None,
            "paths": None,
            "dl2_file": None,
            "output_file": None,
        }
        return NotImplemented

    def test_config(self):
        return NotImplemented

    def test_rebinning(self):
        return NotImplemented

    def test_create_dataset(self):
        return NotImplemented


class TestGammaPreprocessor(unittest.TestCase):
    def setUp(self):
        self.configuration = {
            "directories": None,
            "paths": None,
            "rebin_size": 5,
            "dl2_file": None,
            "output_file": None,
        }
        return NotImplemented

    def test_config(self):
        generator = GammaPreprocessor(config=self.configuration)

        self.assertEqual(generator.output_file, None)
        self.assertEqual(generator.dl2_file, None)
        return NotImplemented

    def test_rebinning(self):
        return NotImplemented

    def test_create_dataset(self):
        return NotImplemented


class TestObservationPreprocessor(unittest.TestCase):
    def setUp(self):
        self.configuration = {
            "directories": None,
            "paths": None,
            "rebin_size": 5,
            "dl2_file": None,
            "output_file": None,
        }
        return NotImplemented

    def test_config(self):
        return NotImplemented

    def test_rebinning(self):
        return NotImplemented

    def test_create_dataset(self):
        return NotImplemented


if __name__ == "__main__":
    unittest.main()
