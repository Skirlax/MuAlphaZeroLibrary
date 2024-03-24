import uuid
import atexit
import numpy as np
import os


class LazyArray:
    def __init__(self, array: np.ndarray, directory_path: str):
        self.directory_path = directory_path
        self.path = f"{directory_path}/array_{uuid.uuid4()}.npy"
        np.save(self.path, array)

    def load_array(self):
        return np.load(self.path)

    def __array__(self):
        return self.load_array()

    def remove_array(self):
        os.remove(self.path)
