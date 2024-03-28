import json
import math

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


def resize_obs(observations: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    obs = Image.fromarray(observations)
    obs = obs.resize(size)
    return np.array(obs)


def scale_state(state: np.ndarray):
    # scales the given state to be between 0 and 1
    return state / 255


def scale_action(action: int, num_actions: int):
    return action / (num_actions - 1)


def scale_reward_value(value: th.Tensor, e: float = 0.001):
    if isinstance(value, float) or isinstance(value, np.float32):
        scaled_v = np.sign(value) * (np.sqrt(np.abs(value) + 1) - 1 + value * e)
        return np.array([scaled_v])
    return th.sign(value) * (th.sqrt(th.abs(value) + 1) - 1 + value * e)


def scale_reward(reward: float):
    return math.log(reward + 1, 5)


def mz_optuna_parameter_search(n_trials: int, init_net_path: str, storage: str or None, study_name: str, game,
                               muzero_config: MuZeroConfig, in_memory: bool = False, direction: str = "maximize"):
    def objective(trial):
        # num_mc_simulations = trial.suggest_int("num_mc_simulations", 100, 1200)
        # num_self_play_games = trial.suggest_int("num_self_play_games", 100, 500)
        # num_epochs = trial.suggest_int("num_epochs", 100, 3000)
        lr = trial.suggest_float("lr", 1e-8, 1e-2, log=True)
        tau = trial.suggest_float("tau", 0.5, 1.5)
        # arena_tau = trial.suggest_float("arena_tau", 0.01, 0.5)
        c = trial.suggest_float("c", 0.5, 5)
        c2 = trial.suggest_categorical("c2", [19652, 10_000, 0.01, 10, 0.1])
        K = trial.suggest_int("K", 1, 10)

        muzero_config.num_simulations = 200
        muzero_config.self_play_games = 20
        muzero_config.epochs = 3600
        muzero_config.lr = lr
        muzero_config.tau = tau
        muzero_config.c = c
        muzero_config.c2 = c2
        # muzero_config.arena_tau = arena_tau
        muzero_config.K = K
        muzero_config.num_iters = 5

        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        trial.net_action_size = int(game.get_num_actions())
        network = MuZeroNet.make_from_config(muzero_config).to(device)
        network.load_state_dict(th.load(init_net_path))
        tree = MuZeroSearchTree(game.make_fresh_instance(), muzero_config)
        net_player = NetPlayer(game.make_fresh_instance(), **{"network": network, "monte_carlo_tree_search": tree})
        arena = MzArena(game.make_fresh_instance(), muzero_config, device)
        mem = MemBuffer(muzero_config.max_buffer_size, disk=True, full_disk=False, dir_path=muzero_config.pickle_dir)
        trainer = Trainer.create(muzero_config, game.make_fresh_instance(), network, tree, net_player, headless=True,
                                 arena_override=arena, checkpointer_verbose=False, memory_override=mem)
        mean = trainer.self_play_get_r_mean()
        trial.report(mean, trial.number)
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
