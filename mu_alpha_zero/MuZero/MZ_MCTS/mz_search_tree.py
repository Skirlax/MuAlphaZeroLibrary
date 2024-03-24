import copy
import gc
from multiprocessing import Pool

import numpy as np
import torch as th

from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.mz_game import MuZeroGame
from mu_alpha_zero.General.search_tree import SearchTree
from mu_alpha_zero.MuZero.MZ_MCTS.mz_node import MzAlphaZeroNode
from mu_alpha_zero.MuZero.Network.networks import MuZeroNet
from mu_alpha_zero.MuZero.lazy_arrays import LazyArray
from mu_alpha_zero.MuZero.utils import match_action_with_obs, resize_obs, scale_action, scale_reward, scale_state
from mu_alpha_zero.config import MuZeroConfig
from mu_alpha_zero.mem_buffer import MuZeroFrameBuffer


class MuZeroSearchTree(SearchTree):

    def __init__(self, game_manager: MuZeroGame, muzero_config: MuZeroConfig):
        self.game_manager = game_manager
        self.muzero_config = muzero_config
        self.buffer = MuZeroFrameBuffer(self.muzero_config.frame_buffer_size, self.game_manager.get_noop(),
                                        self.muzero_config.net_action_size)
        self.min_max_q = [float("inf"), -float("inf")]

    def play_one_game(self, network_wrapper: MuZeroNet, device: th.device, dir_path: str or None = None) -> list:
        num_steps = self.muzero_config.num_steps
        frame_skip = self.muzero_config.frame_skip
        state = self.game_manager.reset()
        state = resize_obs(state, self.muzero_config.target_resolution)
        state = scale_state(state)
        self.buffer.init_buffer(state)
        data = []
        for step in range(num_steps):
            pi, (v, latent) = self.search(network_wrapper, state, None, device)
            move = self.game_manager.select_move(pi)
            _, pred_v = network_wrapper.prediction_forward(latent.unsqueeze(0), predict=True)
            state, rew, done = self.game_manager.frame_skip_step(move, None, frame_skip=frame_skip)
            rew = scale_reward(rew)
            state = resize_obs(state, self.muzero_config.target_resolution)
            state = scale_state(state)
            if done:
                break
            move = scale_action(move, self.game_manager.get_num_actions())
            frame = self.buffer.concat_frames().detach().cpu().numpy()
            data.append(
                (pi, v, (rew, move, float(pred_v[0])), frame if dir_path is None else LazyArray(frame, dir_path)))
            self.buffer.add_frame(state, move)

        gc.collect()  # To clear any memory leaks, might not be necessary.
        return data

    def search(self, network_wrapper, state: np.ndarray, current_player: int or None, device: th.device,
               tau: float or None = None):
        if len(self.buffer) == 0:
            self.buffer.init_buffer(state)
        num_simulations = self.muzero_config.num_simulations
        if tau is None:
            tau = self.muzero_config.tau

        root_node = MzAlphaZeroNode()
        state_ = network_wrapper.representation_forward(
            self.buffer.concat_frames().permute(2, 0, 1).unsqueeze(0)).squeeze(0)
        pi, v = network_wrapper.prediction_forward(state_.unsqueeze(0), predict=True)
        pi = pi.flatten().tolist()
        root_node.expand_node(state_, pi, 0)
        for simulation in range(num_simulations):
            current_node = root_node
            path = [current_node]
            action = None
            while current_node.was_visited():
                current_node, action = current_node.get_best_child(c=self.muzero_config.c, c2=self.muzero_config.c2)
                path.append(current_node)

            action = scale_action(action, self.game_manager.get_num_actions())

            current_node_state_with_action = match_action_with_obs(current_node.parent().state, action)
            next_state, reward = network_wrapper.dynamics_forward(current_node_state_with_action.unsqueeze(0),
                                                                  predict=True)
            reward = reward[0][0]
            v = self.game_manager.game_result(current_node.current_player)
            if v is None or not v:
                pi, v = network_wrapper.prediction_forward(next_state.unsqueeze(0), predict=True)
                pi = pi.flatten().tolist()
                v = v.flatten().tolist()[0]
                current_node.expand_node(next_state, pi, reward)
            self.backprop(v, path)

        action_probs = root_node.get_self_action_probabilities(tau=tau)
        root_val_latent = (root_node.get_self_value(), root_node.get_latent())
        root_node = None
        return action_probs, root_val_latent

    def backprop(self, v, path):
        G = 0
        gamma = self.muzero_config.gamma
        for node in reversed(path):
            G = v + node.reward if G == 0 else G
            node.total_value += G
            node.update_q(G)
            self.update_min_max_q(node.q)
            node.scale_q(self.min_max_q[0], self.min_max_q[1])
            node.times_visited += 1
            G = node.reward + gamma * G

    def self_play(self, net: MuZeroNet, device: th.device, num_games: int, memory: GeneralMemoryBuffer) -> tuple[
        int, int, int]:
        for game in range(num_games):
            game_results = self.play_one_game(net, device)
            memory.add_list(game_results)

        return None, None, None

    def make_fresh_instance(self):
        return MuZeroSearchTree(self.game_manager.make_fresh_instance(), copy.deepcopy(self.muzero_config))

    def step_root(self, action: int or None):
        # I am never reusing the tree in MuZero.
        pass

    def update_min_max_q(self, q):
        self.min_max_q[0] = min(self.min_max_q[0], q)
        self.min_max_q[1] = max(self.min_max_q[1], q)

    @staticmethod
    def parallel_self_play(nets: list, trees: list, memory: GeneralMemoryBuffer, device: th.device, num_games: int,
                           num_jobs: int):
        with Pool(num_jobs) as p:
            if memory.is_disk and memory.full_disk:
                results = p.starmap(p_self_play, [
                    (nets[i], trees[i], copy.deepcopy(device), num_games // num_jobs, copy.deepcopy(memory),None) for i in
                    range(len(nets))])
            else:
                results = p.starmap(p_self_play,
                                    [(nets[i], trees[i], copy.deepcopy(device), num_games // num_jobs, None,memory.dir_path) for i in
                                     range(len(nets))])
        for result in results:
            memory.add_list(result)

        return None, None, None


def p_self_play(net, tree, dev, num_g, mem, dir_path: str or None = None):
    data = []
    for _ in range(num_g):
        game_results = tree.play_one_game(net, dev, dir_path=dir_path)
        if mem is not None:
            mem.add_list(game_results)
        else:
            data.extend(game_results)
    return data
