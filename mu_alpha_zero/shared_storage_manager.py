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
        self.optimizer_state_dict = None
        self.was_pitted = True

    def get_experimental_network_params(self):
        return copy.deepcopy(self.experimental_network_params)

    def set_experimental_network_params(self, network_params: dict or None):
        self.experimental_network_params = copy.deepcopy(network_params)

    def get_stable_network_params(self):
        return copy.deepcopy(self.stable_network_params)

    def set_stable_network_params(self, network_params: dict):
        self.stable_network_params = copy.deepcopy(network_params)

    def eval_length(self):
        with self.lock:
            return self.mem_buffer.eval_length()

    def train_length(self):
        with self.lock:
            return self.mem_buffer.train_length()

    def add_list(self, *args, **kwargs):
        with self.lock:
            return self.mem_buffer.add_list(*args, **kwargs)

    def get_dir_path(self):
        with self.lock:
            return self.mem_buffer.get_dir_path()

    def get_buffer(self):
        with self.lock:
            return self.mem_buffer.get_buffer()

    def set_optimizer(self, optimizer):
        self.optimizer_state_dict = copy.deepcopy(optimizer)

    def get_optimizer(self):
        return copy.deepcopy(self.optimizer_state_dict)

    def set_was_pitted(self, was_pitted: bool):
        self.was_pitted = was_pitted

    def get_was_pitted(self):
        return self.was_pitted

    def batch_with_priorities(self, *args, **kwargs):
        with self.lock:
            return self.mem_buffer.batch_with_priorities(*args, **kwargs)

    def reset_priorities(self):
        with self.lock:
            return self.mem_buffer.reset_priorities()

    def batch(self,*args,**kwargs):
        with self.lock:
            return self.mem_buffer.batch(*args, **kwargs)


class SharedStorageManager(BaseManager):
    pass


SharedStorageManager.register("SharedStorage", SharedStorage)
SharedStorageManager.register("MemBuffer", MemBuffer)
