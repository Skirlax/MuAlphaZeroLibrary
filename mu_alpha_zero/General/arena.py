import time
from abc import abstractmethod, ABC
from typing import Type

import wandb

from mu_alpha_zero.AlphaZero.Arena.players import Player, NetPlayer, RandomPlayer
from mu_alpha_zero.shared_storage_manager import SharedStorage


class GeneralArena(ABC):
    @abstractmethod
    def pit(self, player1: Type[Player], player2: Type[Player],
            num_games_to_play: int, num_mc_simulations: int, one_player: bool = False,
            start_player: int = 1):
        pass

    def continuous_pit(self, player1: NetPlayer, player2: NetPlayer, player_2_2: RandomPlayer, num_games_to_play: int,
                       num_mc_simulations: int,
                       shared_storage: SharedStorage,
                       one_player: bool = False, start_player: int = 1):
        wandb.init(project="MZ",name="Arena Pit")
        conf = self.muzero_config if hasattr(self, "muzero_config") else self.alpha_zero_config
        player1.network.eval()
        player2.network.eval()
        for iter_ in range(conf.num_worker_iters):
            tested_params = shared_storage.get_experimental_network_params()
            if tested_params is None:
                time.sleep(5)
                continue
            player1.network.load_state_dict(tested_params)
            player2.network.load_state_dict(shared_storage.get_stable_network_params())
            results_p1, results_p2, _ = self.pit(player1, player2, num_games_to_play, num_mc_simulations,
                                                 one_player=one_player, start_player=start_player)
            wandb.log({"wins_p1_vs_p2": results_p1, "wins_p2_vs_p1": results_p2})
            not_zero = lambda x: x if x != 0 else 1
            if results_p1 / not_zero(results_p1 + results_p2) >= conf.update_threshold:
                shared_storage.set_stable_network_params(tested_params)

            results_p1, results_p2, _ = self.pit(player1, player_2_2, num_games_to_play, num_mc_simulations,
                                                 one_player=one_player, start_player=start_player)
            wandb.log({"wins_p1_vs_random": results_p1, "wins_random_vs_p1": results_p2})
            shared_storage.set_experimental_network_params(None)
