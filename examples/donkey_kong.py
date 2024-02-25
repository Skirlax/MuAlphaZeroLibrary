import numpy as np
import torch as th
from gymnasium import make
from mu_alpha_zero import MuZeroGame


class DonkeyKongGame(MuZeroGame):
    def __init__(self):
        self.donkey_kong_env = make("ALE/DonkeyKong-v5", obs_type="rgb")
        self.is_done = False

    def get_next_state(self, action: int, player: int or None) -> (
            np.ndarray or th.Tensor, int, bool):
        obs, rew, done, _, _ = self.donkey_kong_env.step(action)
        return obs, rew, done

    def reset(self) -> np.ndarray or th.Tensor:
        obs, _ = self.donkey_kong_env.reset()
        return obs

    def get_noop(self) -> int:
        return 0

    def get_num_actions(self) -> int:
        return self.donkey_kong_env.action_space.n

    def game_result(self, player: int or None) -> bool or None:
        return self.is_done

    def make_fresh_instance(self):
        return DonkeyKongGame()

    def render(self):
        pass

    def frame_skip_step(self, action: int, player: int or None, frame_skip: int = 4) -> (
            np.ndarray or th.Tensor, int, bool):
        obs, rew, done = self.get_next_state(action, player)
        for i in range(frame_skip - 1):
            obs, rew, done = self.get_next_state(action, player)
        return obs, rew, done