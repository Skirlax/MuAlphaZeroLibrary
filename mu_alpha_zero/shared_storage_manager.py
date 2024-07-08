from mu_alpha_zero.mem_buffer import MemBuffer

import multiprocessing
from multiprocessing.managers import BaseManager


class SharedStorage:
    def __init__(self, mem_buffer: MemBuffer):
        self.mem_buffer = mem_buffer
        self.experimental_network_params: dict = None
        self.stable_network_params: dict = None

    def get_experimental_network_params(self):
        return self.experimental_network_params.clone()

    def set_experimental_network_params(self, network_params: dict):
        self.experimental_network_params = network_params.clone()

    def get_stable_network_params(self):
        return self.stable_network_params.clone()

    def set_stable_network_params(self, network_params: dict):
        self.stable_network_params = network_params.clone()

    def get_mem_buffer(self):
        return self.mem_buffer


class SharedStorageManager(BaseManager):
    pass


SharedStorageManager.register("SharedStorage", SharedStorage)
