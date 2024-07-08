import copy
from mu_alpha_zero.mem_buffer import MemBuffer

from multiprocessing_on_dill.managers import BaseManager


class SharedStorage:
    def __init__(self, mem_buffer: MemBuffer):
        self.mem_buffer = mem_buffer
        self.experimental_network_params: dict = None
        self.stable_network_params: dict = None

    def get_experimental_network_params(self):
        return copy.deepcopy(self.experimental_network_params)

    def set_experimental_network_params(self, network_params: dict):
        self.experimental_network_params = copy.deepcopy(network_params)

    def get_stable_network_params(self):
        return copy.deepcopy(self.stable_network_params)

    def set_stable_network_params(self, network_params: dict):
        self.stable_network_params = copy.deepcopy(network_params)

    def get_mem_buffer(self):
        return self.mem_buffer

    def add_list(self, list_:list):
        self.mem_buffer.add_list(list_)


class SharedStorageManager(BaseManager):
    pass


SharedStorageManager.register("SharedStorage", SharedStorage)
SharedStorageManager.register("MemBuffer", MemBuffer)
