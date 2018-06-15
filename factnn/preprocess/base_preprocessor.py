
class BasePreprocessor(object):

    def __init__(self, config):
        self.directories = None

    def batch_processor(self):
        return NotImplemented

    def single_processor(self):
        return NotImplemented

    def reformat(self):
        return NotImplemented

    def format(self):
        return NotImplemented

    def create_dataset(self):
        return NotImplemented
