import math

from mu_alpha_zero.MuZero.Network.networks import MuZeroNet
from mu_alpha_zero.General.utils import find_project_root
import torch as th

class JavaNetworks:
    def save(self, networks_wrapper: MuZeroNet) -> str:
        path = f"{find_project_root()}/mz_net.pth"
        networks_wrapper.to_pickle(path)
        return path

    def trace_all(self, networks_wrapper: MuZeroNet, path: str):
        dynamics_trace = th.jit.trace()
