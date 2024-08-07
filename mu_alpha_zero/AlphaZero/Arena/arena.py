import time

import wandb

from mu_alpha_zero.AlphaZero.Arena.players import Player
from mu_alpha_zero.Game.tictactoe_game import TicTacToeGameManager as GameManager
from mu_alpha_zero.General.arena import GeneralArena
from mu_alpha_zero.Hooks.hook_manager import HookManager
from mu_alpha_zero.Hooks.hook_point import HookAt
from mu_alpha_zero.MuZero.utils import scale_action, resize_obs
from mu_alpha_zero.config import AlphaZeroConfig


class Arena(GeneralArena):
    def __init__(self, game_manager: GameManager, alpha_zero_config: AlphaZeroConfig, device,
                 hook_manager: HookManager or None = None, state_managed: bool = False):
        self.game_manager = game_manager
        self.state_managed = state_managed
        self.device = device
        self.hook_manager = hook_manager if hook_manager is not None else HookManager()
        self.alpha_zero_config = alpha_zero_config

    def pit(self, player1: Player, player2: Player, num_games_to_play: int, num_mc_simulations: int,
            one_player: bool = False, start_player: int = 1, add_to_kwargs: dict or None = None, debug: bool = False) -> \
            tuple[int, int, int]:
        """
        Pit two players against each other for a given number of games and gather the results.
        :param start_player: Which player should start the game.
        :param one_player: If True always only the first player will start the game.
        :param player1:
        :param player2:
        :param num_games_to_play:
        :param num_mc_simulations:
        :return: number of wins for player1, number of wins for player2, number of draws
        """

        results = {"wins_p1": 0, "wins_p2": 0, "draws": 0}
        tau = self.alpha_zero_config.arena_tau
        if one_player:
            num_games_per_player = num_games_to_play
        else:
            num_games_per_player = num_games_to_play // 2

        if self.state_managed:
            if player1.name == "NetPlayer":
                player1.monte_carlo_tree_search.game_manager = self.game_manager
            if player2.name == "NetPlayer":
                player2.monte_carlo_tree_search.game_manager = self.game_manager

            player1.set_game_manager(self.game_manager)
            player2.set_game_manager(self.game_manager)

        for game in range(num_games_to_play):
            if game < num_games_per_player:
                current_player = start_player
            else:
                current_player = -start_player

            kwargs = {"num_simulations": num_mc_simulations, "current_player": current_player, "device": self.device,
                      "tau": tau, "unravel": self.alpha_zero_config.unravel}
            if add_to_kwargs is not None:
                kwargs.update(add_to_kwargs)
            if not self.alpha_zero_config.requires_player_to_reset:
                state = self.game_manager.reset()
            else:
                state = self.game_manager.reset(player=current_player)
            if self.alpha_zero_config.arena_running_muzero:
                try:
                    player1.monte_carlo_tree_search.buffer.init_buffer(self.game_manager.get_state_for_player(state, 1 if current_player == 1 else 2),
                                                                       current_player)
                    player2.monte_carlo_tree_search.buffer.init_buffer(
                        self.game_manager.get_state_for_player(state, -1 if current_player == -1 else -2), -current_player)
                except AttributeError:
                    pass

            if player1.name == "NetworkPlayer":
                player1.monte_carlo_tree_search.step_root(None)
            if player2.name == "NetworkPlayer":
                player2.monte_carlo_tree_search.step_root(None)
            # time.sleep(0.01)
            while True:
                self.game_manager.render()
                if current_player == 1:
                    move = player1.choose_move(state, **kwargs)
                else:
                    move = player2.choose_move(state, **kwargs)
                self.hook_manager.process_hook_executes(self, self.pit.__name__, __file__, HookAt.MIDDLE,
                                                        args=(move, kwargs, current_player))
                if not self.state_managed:
                    state = self.game_manager.get_next_state(state, move, current_player)
                    status = self.game_manager.game_result(current_player, state)
                else:
                    state = self.game_manager.get_next_state(move, current_player)[0]
                    status = self.game_manager.game_result(current_player)
                self.game_manager.render()
                if self.alpha_zero_config.arena_running_muzero:
                    move = scale_action(move, self.game_manager.get_num_actions()) if isinstance(move, int) else \
                        scale_action(move[0] * state.shape[0] + move[1], self.game_manager.get_num_actions())
                    try:
                        if current_player == -1:
                            state_fp = self.game_manager.get_state_for_player(state, -2)
                            player2.monte_carlo_tree_search.buffer.add_frame(state_fp, move, current_player)
                            player1.monte_carlo_tree_search.buffer.add_frame(state, move, -current_player)
                        else:
                            state_fp = self.game_manager.get_state_for_player(state, 2)
                            player1.monte_carlo_tree_search.buffer.add_frame(state_fp, move, current_player)
                            player2.monte_carlo_tree_search.buffer.add_frame(state, move, -current_player)
                    except AttributeError:
                        # Player is probably not a net player and doesn't have monte_carlo_tree_search.
                        pass

                if status is not None:
                    if status == 1:
                        if current_player == 1:
                            results["wins_p1"] += 1
                        else:
                            results["wins_p2"] += 1

                    elif status == -1:
                        if current_player == 1:
                            results["wins_p2"] += 1
                        else:
                            results["wins_p1"] += 1
                    else:
                        results["draws"] += 1

                    # if debug:
                    #     self.wait_keypress()
                    if (player1.name == "HumanPlayer" or player2.name == "HumanPlayer") and debug:
                        time.sleep(0.2)

                    wandb.log({"wins_p1": results["wins_p1"], "wins_p2": results["wins_p2"], "draws": results["draws"]})
                    break

                current_player *= -1
                kwargs["current_player"] = current_player
        return results["wins_p1"], results["wins_p2"], results["draws"]

    def wait_keypress(self):
        inpt = input("Press any key to continue...")
        return inpt
