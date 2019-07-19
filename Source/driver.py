import numpy as np
import pandas


class Driver:
    def __init__(self, path):
        self.path = path
        self._data = np.array(pandas.read_csv(filepath_or_buffer=path, sep=' ', header=None)).flatten()
        print('Dataset loaded completely.')

    def get_data(self):
        return self._data

    def generate_test_data(self, sample):
        return np.concatenate((self._data[1:], [sample]))
