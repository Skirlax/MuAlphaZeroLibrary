import sys
from typing import Type

import numpy as np
import torch as th

from mu_alpha_zero.AlphaZero.Arena.arena import Arena
from mu_alpha_zero.AlphaZero.Arena.players import NetPlayer
from mu_alpha_zero.AlphaZero.MCTS.az_search_tree import McSearchTree
from mu_alpha_zero.AlphaZero.Network.trainer import Trainer
from mu_alpha_zero.General.az_game import AlphaZeroGame
from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.network import GeneralNetwork
from mu_alpha_zero.General.utils import net_not_none, find_project_root
from mu_alpha_zero.config import AlphaZeroConfig


class AlphaZero:
    def __init__(self, game_instance: AlphaZeroGame):
        self.trainer = None
        self.net = None
        self.game = game_instance
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.alpha_zero_config: AlphaZeroConfig = None
        self.tree: McSearchTree = None

    def create_new(self, alpha_zero_config: AlphaZeroConfig, network_class: Type[GeneralNetwork],
                   memory: GeneralMemoryBuffer,
                   headless: bool = True,
                   checkpointer_verbose: bool = False):
        network = network_class.make_from_config(alpha_zero_config).to(self.device)
        tree = McSearchTree(self.game.make_fresh_instance(), alpha_zero_config)
        self.tree = tree
        net_player = NetPlayer(self.game.make_fresh_instance(), **{"network": network, "monte_carlo_tree_search": tree})
        self.alpha_zero_config = alpha_zero_config
        self.trainer = Trainer.create(alpha_zero_config, self.game, network, tree, net_player, headless=headless,
                                      checkpointer_verbose=checkpointer_verbose, memory_override=memory)
        self.net = self.trainer.get_network()

    def load_checkpoint(self, network_class: Type[GeneralNetwork], path: str, checkpoint_dir: str,
                        headless: bool = True,
                        checkpointer_verbose: bool = False):
        self.trainer = Trainer.from_checkpoint(network_class, McSearchTree, NetPlayer, path, checkpoint_dir, self.game,
                                               headless=headless,
                                               checkpointer_verbose=checkpointer_verbose)
        self.net = self.trainer.get_network()
        self.tree = self.trainer.get_tree()
        self.args = self.trainer.get_args()

    def train(self):
        net_not_none(self.net)
        self.trainer.train()

    def predict(self, x: np.ndarray, tau: float = 0) -> int:
        net_not_none(self.net)
        assert x.shape == (self.args["board_size"], self.args["board_size"], self.args["num_net_in_channels"]), \
            "Input shape is not correct. Expected (board_size, board_size, num_net_in_channels)." \
            "Got: " + str(x.shape)
        pi, _ = self.tree.search(self.net, x, 1, self.device, tau=tau)
        return self.game.select_move(pi)

    def play(self, p1_name: str, p2_name: str, num_games: int, alpha_zero_config: AlphaZeroConfig, starts: int = 1,
             switch_players: bool = True):
        net_not_none(self.net)
        self.net.to(self.device)
        self.net.eval()
        manager = self.game.make_fresh_instance()
        tree = McSearchTree(manager, alpha_zero_config)
        kwargs = {"network": self.net, "monte_carlo_tree_search": tree, "evaluate_fn": manager.eval_board,
                  "depth": alpha_zero_config.minimax_depth, "player": -1}
        path_prefix = find_project_root().replace("\\", "/").split("/")[-1]
        p1 = sys.modules[f"{path_prefix}.AlphaZero.Arena.players"].__dict__[p1_name](manager, **kwargs)
        p2 = sys.modules[f"{path_prefix}.AlphaZero.Arena.players"].__dict__[p2_name](manager, **kwargs)
        arena_manager = self.game.make_fresh_instance()
        arena_manager.set_headless(False)
        arena = Arena(arena_manager, alpha_zero_config, self.device)
        p1_w, p2_w, ds = arena.pit(p1, p2, num_games, alpha_zero_config.num_simulations, one_player=not switch_players,
                                   start_player=starts, add_to_kwargs=kwargs)
        print(f"Results: Player 1 wins: {p1_w}, Player 2 wins: {p2_w}, Draws: {ds}")

