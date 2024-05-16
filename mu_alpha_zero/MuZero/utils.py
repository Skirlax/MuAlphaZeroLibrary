import json
import math
from typing import Callable

import numpy as np
import optuna
import torch as th
from PIL import Image

from mu_alpha_zero.config import MuZeroConfig


def add_actions_to_obs(observations: th.Tensor, actions: th.Tensor, dim=0):
    return th.cat((observations, actions), dim=dim)


def match_action_with_obs(observations: th.Tensor, action: int):
    action = th.full((1, observations.shape[1], observations.shape[2]), action, dtype=th.float32,
                     device=observations.device)
    return add_actions_to_obs(observations, action)


def match_action_with_obs_batch(observation_batch: th.Tensor, action_batch: list[int]):
    tensors = [th.full((1, 1, observation_batch.shape[2], observation_batch.shape[3]), action, dtype=th.float32,
                       device=observation_batch.device) for action in action_batch]
    actions = th.cat(tensors, dim=0)
    return add_actions_to_obs(observation_batch, actions, dim=1)


def resize_obs(observations: np.ndarray, size: tuple[int, int], resize: bool) -> np.ndarray:
    if not resize:
        return observations
    obs = Image.fromarray(observations)
    obs = obs.resize(size)
    return np.array(obs)


def scale_state(state: np.ndarray, scale: bool):
    if not scale:
        return state
    # scales the given state to be between 0 and 1
    return state / 255


def scale_action(action: int, num_actions: int):
    return action / (num_actions - 1)


def scale_hidden_state(hidden_state: th.Tensor):
    return (hidden_state - th.min(hidden_state)) / (th.max(hidden_state) - th.min(hidden_state))


def scale_reward_value(value: th.Tensor, e: float = 0.001):
    if isinstance(value, float) or isinstance(value, np.float32):
        scaled_v = np.sign(value) * (np.sqrt(np.abs(value) + 1) - 1 + value * e)
        return float(scaled_v)
    return th.sign(value) * (th.sqrt(th.abs(value) + 1) - 1 + value * e)


def scale_reward(reward: float):
    return math.log(reward + 1, 5)


def mz_optuna_parameter_search(n_trials: int, storage: str or None, study_name: str, game,
                               muzero_config: MuZeroConfig, in_memory: bool = False, direction: str = "maximize",
                               arena_override= None, memory_override=None):
    def objective(trial: optuna.Trial):
        muzero_config.num_simulations = trial.suggest_int("num_mc_simulations", 60, 800)
        muzero_config.lr = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
        muzero_config.tau = trial.suggest_float("temp", 0.5, 2)
        muzero_config.arena_tau = trial.suggest_float("arena_temp", 0, 2)
        muzero_config.c = trial.suggest_float("cpuct", 0.7, 2)
        muzero_config.l2 = trial.suggest_float("l2_norm", 1e-6, 6e-2)
        muzero_config.frame_buffer_size = trial.suggest_int("frame_buffer_size", 10, 40)
        muzero_config.alpha = trial.suggest_float("alpha", 0.4, 2)
        muzero_config.beta = trial.suggest_float("beta", 0.1, 1)
        muzero_config.balance_term = trial.suggest_categorical("loss_scale", [1, 0.5])
        muzero_config.num_blocks = trial.suggest_int("num_blocks", 5, 32)
        muzero_config.K = trial.suggest_int("k", 2, 25)
        muzero_config.epochs = trial.suggest_int("epochs", 200, 500)

        muzero_config.self_play_games = 300
        muzero_config.num_iters = 1

        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        trial.net_action_size = int(game.get_num_actions())
        network = MuZeroNet.make_from_config(muzero_config, game).to(device)
        tree = MuZeroSearchTree(game.make_fresh_instance(), muzero_config)
        net_player = NetPlayer(game.make_fresh_instance(), **{"network": network, "monte_carlo_tree_search": tree})
        if arena_override is None:
            arena = MzArena(game.make_fresh_instance(), muzero_config, device)
        else:
            arena = arena_override
        if memory_override is None:
            mem = MemBuffer(muzero_config.max_buffer_size, disk=True, full_disk=False, dir_path=muzero_config.pickle_dir)
        else:
            mem = memory_override
        trainer = Trainer.create(muzero_config, game.make_fresh_instance(), network, tree, net_player, headless=True,
                                 arena_override=arena, checkpointer_verbose=False, memory_override=mem)
        trainer.train()
        mean = trainer.get_arena_win_frequencies_mean()
        trial.report(mean, muzero_config.num_iters)
        print(f"Trial {trial.number} finished with win freq {mean}.")
        del trainer
        del network
        del tree
        del net_player
        return mean

    from mu_alpha_zero.MuZero.MZ_Arena.arena import MzArena
    from mu_alpha_zero.MuZero.MZ_MCTS.mz_search_tree import MuZeroSearchTree
    from mu_alpha_zero.MuZero.Network.networks import MuZeroNet
    from mu_alpha_zero.AlphaZero.Network.trainer import Trainer
    from mu_alpha_zero.AlphaZero.Arena.players import NetPlayer
    from mu_alpha_zero.mem_buffer import MemBuffer

    muzero_config.show_tqdm = False
    if in_memory:
        study = optuna.create_study(study_name=study_name, direction=direction)
    else:
        if storage is None:
            raise ValueError("Storage can't be None if in_memory is False.")
        study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials)
    with open(f"{muzero_config.checkpoint_dir}/study_params.json", "w") as file:
        json.dump(study.best_params, file)


def mask_invalid_actions(invalid_actions: np.ndarray, pi: np.ndarray):
    pi = pi.reshape(-1) * invalid_actions.reshape(-1)
    return pi / pi.sum()


def mask_invalid_actions_batch(get_invalid_actions: Callable, pis: th.Tensor, players: list[int]):
    invalid_actions_ts = th.empty(pis.shape)
    for i, player in enumerate(players):
        invaid_actions = get_invalid_actions(pis[i], player)
        invalid_actions_ts[i] = invaid_actions
    return invalid_actions_ts
