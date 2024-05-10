import random
from typing import Type

import torch as th

from mu_alpha_zero.AlphaZero.Arena.players import Player
from mu_alpha_zero.General.arena import GeneralArena
from mu_alpha_zero.General.mz_game import MuZeroGame
from mu_alpha_zero.Hooks.hook_manager import HookManager
from mu_alpha_zero.Hooks.hook_point import HookAt
from mu_alpha_zero.MuZero.utils import resize_obs, scale_state, scale_action
from mu_alpha_zero.config import MuZeroConfig


class MzArena(GeneralArena):
    def __init__(self, game_manager: MuZeroGame, muzero_config: MuZeroConfig, device: th.device,
                 hook_manager: HookManager or None = None):
        self.game_manager = game_manager
        self.muzero_config = muzero_config
        self.hook_manager = hook_manager if hook_manager is not None else HookManager()
        self.device = device

    def pit(self, player1: Type[Player], player2: Type[Player], num_games_to_play: int, num_mc_simulations: int,
            one_player: bool = False, start_player: int = 1):
        tau = self.muzero_config.arena_tau
        rewards = {1: [], -1: []}
        if one_player:
            num_games_per_player = num_games_to_play
        else:
            num_games_per_player = num_games_to_play // 2
        noop_num = random.randint(0, 30)
        players = {"1": player1, "-1": player2}
        for player in [1, -1]:
            kwargs = {"num_simulations": num_mc_simulations, "current_player": player, "device": self.device,
                      "tau": tau, "unravel": False}
            for game in range(num_games_per_player):
                self.game_manager.reset()

                state, _, _ = self.game_manager.frame_skip_step(self.game_manager.get_noop(), player,
                                                                frame_skip=noop_num)
                state = resize_obs(state, self.muzero_config.target_resolution, self.muzero_config.resize_images)
                state = scale_state(state,self.muzero_config.scale_state)
                for step in range(self.muzero_config.num_steps):
                    self.game_manager.render()
                    move = players[str(player)].choose_move(state, **kwargs)

                    state, reward, done = self.game_manager.frame_skip_step(move, player)
                    state = resize_obs(state, self.muzero_config.target_resolution, self.muzero_config.resize_images)
                    state = scale_state(state,self.muzero_config.scale_state)
                    try:
                        players[str(player)].monte_carlo_tree_search.buffer.add_frame(state, scale_action(move,
                                                                                                          self.game_manager.get_num_actions()),
                                                                                      player)
                    except AttributeError:
                        # Player is probably not net player and doesn't have monte_carlo_tree_search.
                        pass
                    rewards[player].append(reward)
                    if done:
                        break

        self.hook_manager.process_hook_executes(self, self.pit.__name__, __file__, HookAt.TAIL, (rewards, kwargs))
        return sum(rewards[1]), sum(rewards[-1]), 0

    def run_on_training_end(self):
        self.hook_manager.process_hook_executes(self, self.run_on_training_end.__name__, __file__, HookAt.ALL)
