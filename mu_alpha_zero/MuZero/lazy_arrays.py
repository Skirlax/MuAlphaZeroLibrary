import gzip
import os
import uuid

import numpy as np


class LazyArray:
    def __init__(self, array: np.ndarray, directory_path: str):
        self.directory_path = directory_path
        self.path = f"{directory_path}/array_{uuid.uuid4()}.npy.gz"
        self.persistent = False
        self.save_array(array)

    def load_array(self):
        f = gzip.GzipFile(self.path, 'r')
        arr = np.load(f)
        f.close()
        return arr

    def save_array(self, array: np.ndarray):
        file = gzip.GzipFile(self.path, 'w')
        np.save(file, array)
        file.close()

    def __array__(self):
        return self.load_array()

    def remove_array(self):
        os.remove(self.path)

    def make_persistent(self):
        self.persistent = True


