import copy
import multiprocess
# from pathos.helpers import mp as multiprocess
from multiprocess.managers import BaseManager
from mu_alpha_zero.mem_buffer import MemBuffer


class SharedStorage:
    def __init__(self, mem_buffer: MemBuffer):
        self.mem_buffer = mem_buffer
        self.experimental_network_params: dict or None = None
        self.stable_network_params: dict or None = None
        self.lock = multiprocess.context._default_context.Lock()

    def get_experimental_network_params(self):
        return copy.deepcopy(self.experimental_network_params)

    def set_experimental_network_params(self, network_params: dict or None):
        self.experimental_network_params = copy.deepcopy(network_params)

    def get_stable_network_params(self):
        return copy.deepcopy(self.stable_network_params)

    def set_stable_network_params(self, network_params: dict):
        self.stable_network_params = copy.deepcopy(network_params)

    # def get_mem_buffer(self):
    #     return self.mem_buffer

    def __getattr__(self, item):
        with self.lock:
            if hasattr(self, item):
                return getattr(self, item)
            if hasattr(self.mem_buffer, item):
                return getattr(self.mem_buffer, item)
        raise AttributeError(f"SharedStorage and MemBuffer have no attribute {item}")


class SharedStorageManager(BaseManager):
    pass


SharedStorageManager.register("SharedStorage", SharedStorage)
SharedStorageManager.register("MemBuffer", MemBuffer)
