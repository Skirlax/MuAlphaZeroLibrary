import numpy as np
import torch as th
from mu_alpha_zero.General.mz_game import MuZeroGame

from connect4_game import Connect4


class GameEnv(MuZeroGame):
    def __init__(self, rows: int, columns: int, num_to_win: int, headless: bool):
        super(GameEnv, self).__init__()
        self.rows = rows
        self.columns = columns
        self.num_to_win = num_to_win
        self.headless = headless
        self.num_actions = columns
        self.game = Connect4(rows, columns, num_to_win, headless)

    def get_next_state(self, action: int, player: int or None) -> (
            np.ndarray or th.Tensor, int, bool):
        state, reward, done = self.game.step(action, player)
        state = state.reshape(self.rows, self.columns, 1)
        player_array = np.full((self.rows, self.columns, 1), -player)
        state = np.concatenate((state, player_array), axis=2)
        return state, reward, done

    def reset(self, player: int = 1) -> np.ndarray or th.Tensor:
        self.game = Connect4(self.rows, self.columns, self.num_to_win, self.headless)
        state = self.game.board.reshape(self.rows, self.columns, 1)
        player_array = np.full((self.rows, self.columns, 1), player)
        state = np.concatenate((state, player_array), axis=2)
        return state

    def get_noop(self) -> int:
        return -1

    def get_num_actions(self) -> int:
        return self.num_actions

    def game_result(self, player: int or None) -> bool or None:
        return self.game.game_result(player)

    def make_fresh_instance(self):
        return GameEnv(self.rows, self.columns, self.num_to_win, self.headless)

    def render(self):
        self.game.render()

    def frame_skip_step(self, action: int, player: int or None, frame_skip: int = 4) -> (
            np.ndarray or th.Tensor, int, bool):
        return self.get_next_state(action, player)

    def get_invalid_actions(self, state: np.ndarray, player: int):
        first_row = state[0, :, 0]
        mask = np.where(first_row == 0, 1, 0)
        return mask

    def get_random_valid_action(self, state: np.ndarray, **kwargs):
        mask = self.get_invalid_actions(state, kwargs.get("current_player"))
        return np.random.choice(np.where(mask == 1)[0])

    def get_state_for_passive_player(self, state: np.ndarray, player: int):
        return state

    def get_human_input(self, board: np.ndarray):
        return self.game.get_human_input(board)[1]
