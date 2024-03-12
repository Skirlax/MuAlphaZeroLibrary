import os
import sys

import numpy as np

sys.path.append(os.path.abspath("."))
from typing import Type

import torch as th
from mu_alpha_zero.AlphaZero.Arena.players import NetPlayer
from mu_alpha_zero.AlphaZero.Network.trainer import Trainer
from mu_alpha_zero.General.utils import find_project_root, net_not_none
from mu_alpha_zero.General.mz_game import MuZeroGame
from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.network import GeneralNetwork
from mu_alpha_zero.MuZero.MZ_Arena.arena import MzArena
from mu_alpha_zero.MuZero.MZ_MCTS.mz_search_tree import MuZeroSearchTree
from mu_alpha_zero.mem_buffer import PickleMemBuffer
from mu_alpha_zero.config import MuZeroConfig


class MuZero:
    """
    Class for managing the training and creation of a MuZero model.

    Attributes:
        game_manager (MuZeroGame): The game manager instance.
        net (GeneralNetwork): The neural network used by MuZero for prediction.
        trainer (Trainer): The trainer object responsible for training the model.
        device (torch.device): The device (CPU or CUDA) used for computations.

    Methods:
        __init__(self, game_manager: MuZeroGame)
            Initializes a new instance of the MuZero class.

        create_new(self, args: dict, network_class: Type[GeneralNetwork], headless: bool = True,
                   checkpointer_verbose: bool = False)
            Creates a new MuZero model using the specified arguments.

        train(self)
            Trains the MuZero model.
    """

    def __init__(self, game_manager: MuZeroGame):
        self.game_manager = game_manager
        self.net = None
        self.trainer = None
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.tree = None
        self.muzero_config = None

    def create_new(self, muzero_config: MuZeroConfig, network_class: Type[GeneralNetwork], memory: GeneralMemoryBuffer,
                   headless: bool = True,
                   checkpointer_verbose: bool = False):
        muzero_config.net_action_size = int(self.game_manager.get_num_actions())
        if not os.path.isabs(muzero_config.pickle_dir):
            muzero_config.pickle_dir = find_project_root() + "/" + muzero_config.pickle_dir
        self.muzero_config = muzero_config
        network = network_class.make_from_config(muzero_config).to(self.device)
        self.tree = MuZeroSearchTree(self.game_manager.make_fresh_instance(), muzero_config)

        net_player = NetPlayer(self.game_manager.make_fresh_instance(),
                               **{"network": network, "monte_carlo_tree_search": self.tree})

        arena = MzArena(self.game_manager.make_fresh_instance(), self.muzero_config, self.device)
        java_manager = None
        self.trainer = Trainer.create(self.muzero_config, self.game_manager.make_fresh_instance(), network, self.tree,
                                      net_player,
                                      headless=headless,
                                      checkpointer_verbose=checkpointer_verbose, arena_override=arena,
                                      memory_override=memory, java_manager=java_manager)
        self.net = self.trainer.get_network()

    def from_checkpoint(self, network_class: Type[GeneralNetwork], memory: GeneralMemoryBuffer, path: str,
                        checkpoint_dir: str,
                        headless: bool = True,
                        checkpointer_verbose: bool = False):
        self.trainer = Trainer.from_checkpoint(network_class, MuZeroSearchTree, NetPlayer, path,
                                               checkpoint_dir,
                                               self.game_manager,
                                               headless=headless,
                                               checkpointer_verbose=checkpointer_verbose,
                                               mem=memory)
        self.net = self.trainer.get_network()
        self.tree = self.trainer.get_tree()
        self.args = self.trainer.get_args()

    def train(self):
        net_not_none(self.net)
        self.trainer.train()

    def predict(self, x: np.ndarray, tau: float = 0) -> int:
        net_not_none(self.net)
        assert x.shape == self.muzero_config.target_resolution + (
            3,), "Input shape must match target resolution with 3 channels. Got: " + str(x.shape)
        self.net.eval()
        pi, (v, _) = self.tree.search(self.net, x, None, self.device, tau=tau)
        move = self.game_manager.select_move(pi)
        return move
