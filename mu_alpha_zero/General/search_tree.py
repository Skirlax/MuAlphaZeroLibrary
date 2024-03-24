from abc import ABC, abstractmethod

import numpy as np
import torch as th

from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.network import GeneralNetwork


class SearchTree(ABC):

    @abstractmethod
    def play_one_game(self, network, device: th.device,dir_path: str or None = None) -> tuple[list, int, int, int]:
        """
        Performs one game of the algorithm
        """
        pass

    @abstractmethod
    def search(self, network, state: np.ndarray, current_player: int or None, device: th.device,
               tau: float or None = None):
        """
        Performs MCTS search for given number of simulations.
        """

        pass

    @abstractmethod
    def make_fresh_instance(self):
        """
        Return new instance of this class.
        """
        pass

    @abstractmethod
    def step_root(self, action: int or None):
        """
        Steps the root node to the given action.
        """
        pass

    @abstractmethod
    def self_play(self, net: GeneralNetwork, device: th.device, num_games: int, memory: GeneralMemoryBuffer) -> tuple[
        int, int, int]:
        """
        Performs self play for given number of games.
        """
        pass
