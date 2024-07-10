import copy
import gc
import time

from multiprocess import set_start_method

set_start_method("spawn", force=True)
from multiprocess.pool import Pool

import numpy as np
import torch as th
from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.mz_game import MuZeroGame
from mu_alpha_zero.General.search_tree import SearchTree
from mu_alpha_zero.Hooks.hook_manager import HookManager
from mu_alpha_zero.Hooks.hook_point import HookAt
from mu_alpha_zero.MuZero.MZ_MCTS.mz_node import MzAlphaZeroNode
from mu_alpha_zero.MuZero.Network.networks import MuZeroNet
from mu_alpha_zero.MuZero.lazy_arrays import LazyArray
from mu_alpha_zero.MuZero.utils import match_action_with_obs, resize_obs, scale_action, scale_state, \
    scale_hidden_state, mask_invalid_actions
from mu_alpha_zero.config import MuZeroConfig
from mu_alpha_zero.mem_buffer import MuZeroFrameBuffer
from mu_alpha_zero.shared_storage_manager import SharedStorage


class MuZeroSearchTree(SearchTree):

    def __init__(self, game_manager: MuZeroGame, muzero_config: MuZeroConfig, hook_manager: HookManager or None = None):
        self.game_manager = game_manager
        self.muzero_config = muzero_config
        self.hook_manager = hook_manager if hook_manager is not None else HookManager()
        self.buffer = self.init_frame_buffer()
        self.min_max_q = [float("inf"), -float("inf")]

    def init_frame_buffer(self):
        if self.muzero_config.enable_frame_buffer:
            return MuZeroFrameBuffer(self.muzero_config.frame_buffer_size, self.game_manager.get_noop(),
                                     self.muzero_config.net_action_size)
        return MuZeroFrameBuffer(1, self.game_manager.get_noop(), self.muzero_config.net_action_size)

    def play_one_game(self, network_wrapper: MuZeroNet, device: th.device, dir_path: str or None = None,
                      calculate_avg_num_children: bool = False) -> list:
        num_steps = self.muzero_config.num_steps
        frame_skip = self.muzero_config.frame_skip
        state = self.game_manager.reset()
        state = resize_obs(state, self.muzero_config.target_resolution, self.muzero_config.resize_images)
        state = scale_state(state, self.muzero_config.scale_state)
        player = 1
        self.buffer.init_buffer(state, player)
        if self.muzero_config.multiple_players:
            self.buffer.init_buffer(self.game_manager.get_state_for_passive_player(state, -player), -player)
        data = []

        for step in range(num_steps):
            pi, (v, latent) = self.search(network_wrapper, state, player, device, calculate_avg_num_children=(
                    calculate_avg_num_children and step == 0))
            move = self.game_manager.select_move(pi, tau=self.muzero_config.tau)
            _, pred_v = network_wrapper.prediction_forward(latent.unsqueeze(0), predict=True)
            state, rew, done = self.game_manager.frame_skip_step(move, player, frame_skip=frame_skip)
            state = resize_obs(state, self.muzero_config.target_resolution, self.muzero_config.resize_images)
            state = scale_state(state, self.muzero_config.scale_state)
            move = scale_action(move, self.game_manager.get_num_actions())
            frame = self.buffer.concat_frames(player).detach().cpu().numpy()
            data.append(
                (pi, v, (rew, move, float(pred_v[0]), player),
                 frame if dir_path is None else LazyArray(frame, dir_path)))
            if done:
                break
            if self.muzero_config.multiple_players:
                player = -player
            self.buffer.add_frame(state, move, player)
            self.buffer.add_frame(self.game_manager.get_state_for_passive_player(state, -player), move, -player)

        gc.collect()  # To clear any memory leaks, might not be necessary.
        return data

    def search(self, network_wrapper, state: np.ndarray, current_player: int or None, device: th.device,
               tau: float or None = None, calculate_avg_num_children: bool = False):
        if self.buffer.__len__(current_player) == 0:
            self.buffer.init_buffer(state, current_player)
        num_simulations = self.muzero_config.num_simulations
        if tau is None:
            tau = self.muzero_config.tau

        root_node = MzAlphaZeroNode(current_player=current_player)
        state_ = network_wrapper.representation_forward(
            self.buffer.concat_frames(current_player).permute(2, 0, 1).unsqueeze(0)).squeeze(0)
        state_ = scale_hidden_state(state_)
        pi, v = network_wrapper.prediction_forward(state_.unsqueeze(0), predict=True)
        pi = pi + np.random.dirichlet([self.muzero_config.dirichlet_alpha] * self.muzero_config.net_action_size)
        pi = mask_invalid_actions(self.game_manager.get_invalid_actions(state, current_player), pi)
        pi = pi.flatten().tolist()
        root_node.expand_node(state_, pi, 0)
        for simulation in range(num_simulations):
            current_node = root_node
            path = [current_node]
            action = None
            while current_node.was_visited():
                current_node, action = current_node.get_best_child(self.min_max_q[0], self.min_max_q[1],
                                                                   c=self.muzero_config.c, c2=self.muzero_config.c2)
                path.append(current_node)

            action = scale_action(action, self.game_manager.get_num_actions())

            current_node_state_with_action = match_action_with_obs(current_node.parent().state, action)
            next_state, reward = network_wrapper.dynamics_forward(current_node_state_with_action.unsqueeze(0),
                                                                  predict=True)
            next_state = scale_hidden_state(next_state)
            reward = reward[0][0]
            pi, v = network_wrapper.prediction_forward(next_state.unsqueeze(0), predict=True)
            pi = pi.flatten().tolist()
            v = v.flatten().tolist()[0]
            current_node.expand_node(next_state, pi, reward)
            self.backprop(v, path)

        action_probs = root_node.get_self_action_probabilities()
        root_val_latent = (root_node.get_self_value(), root_node.get_latent())
        self.hook_manager.process_hook_executes(self, self.search.__name__, __file__, HookAt.TAIL,
                                                args=(action_probs, root_val_latent, root_node))
        # if calculate_avg_num_children:
        #     num_nodes = self.get_num_nodes(root_node)
        #     wandb.log({"Number of non-leaf nodes": num_nodes})
        root_node = None
        return action_probs, root_val_latent

    def backprop(self, v, path):
        G = v
        G_node = -v
        for node in reversed(path):
            gamma = 1 if G == v else self.muzero_config.gamma
            if self.muzero_config.multiple_players:
                G = node.reward + gamma * (
                    -G)  # G should be from the perspective of the parent as the parent is selecting from the children based on what's good for them.
                G_node = node.reward + gamma * (-G_node)
                node.total_value += G_node
            else:
                G = node.reward + gamma * G
                node.total_value += G
            node.update_q(G)
            self.update_min_max_q(node.q)
            node.times_visited += 1

    def self_play(self, net: MuZeroNet, device: th.device, num_games: int, memory: GeneralMemoryBuffer) -> tuple[
        int, int, int]:
        for game in range(num_games):
            game_results = self.play_one_game(net, device, calculate_avg_num_children=game == num_games - 1)
            memory.add_list(game_results)

        return None, None, None

    def make_fresh_instance(self):
        return MuZeroSearchTree(self.game_manager.make_fresh_instance(), copy.deepcopy(self.muzero_config),
                                hook_manager=copy.deepcopy(self.hook_manager))

    def step_root(self, action: int or None):
        # I am never reusing the tree in MuZero.
        pass

    def update_min_max_q(self, q):
        self.min_max_q[0] = min(self.min_max_q[0], q)
        self.min_max_q[1] = max(self.min_max_q[1], q)

    def get_num_nodes(self, root_node: MzAlphaZeroNode):
        num_nodes = 0
        if len(root_node.children) > 0:
            num_nodes = 1
        for child in root_node.children.values():
            nn = self.get_num_nodes(child)
            num_nodes += nn

        return num_nodes

    @staticmethod
    def parallel_self_play(nets: list, trees: list, memory: GeneralMemoryBuffer, device: th.device, num_games: int,
                           num_jobs: int):
        with Pool(num_jobs) as p:
            if memory.is_disk and memory.full_disk:
                results = p.starmap(p_self_play, [
                    (nets[i], trees[i], copy.deepcopy(device), num_games // num_jobs, copy.deepcopy(memory), None) for i
                    in range(len(nets))])
            else:
                results = p.starmap(p_self_play, [
                    (nets[i], trees[i], copy.deepcopy(device), num_games // num_jobs, None, memory.dir_path) for i in
                    range(len(nets))])
        for result in results:
            memory.add_list(result)

        return None, None, None

    @staticmethod
    def start_continuous_self_play(nets: list, trees: list, shared_storage: SharedStorage,
                                   device: th.device, num_games: int, num_jobs: int, num_worker_iters: int):
        pool = Pool(num_jobs)
        for i in range(num_jobs):
            pool.apply_async(c_p_self_play, args=(
                nets[i], trees[i], copy.deepcopy(device), num_games // num_jobs, shared_storage, num_worker_iters,
                shared_storage.get_mem_buffer().get_dir_path())
            )

        return pool

    @staticmethod
    def reanalyze(net, tree, device, shared_storage: SharedStorage, config: MuZeroConfig):
        def get_first_n(n: int, mem: GeneralMemoryBuffer):
            buffer = mem.get_buffer()
            data = []
            for i in range(n):
                data.append((buffer[i], i))
            return data

        memory = shared_storage.get_mem_buffer()
        net = net.to(device)
        net.eval()
        while len(memory.get_buffer()) < config.batch_size * 6:
            time.sleep(5)
        for iter_ in range(config.num_worker_iters):
            data = get_first_n(config.batch_size * 2, memory)
            for pi, v, (rew, move, pred_v, player), frame in data:
                if isinstance(frame, LazyArray):
                    frame = frame.load_array()
                state = th.tensor(frame, device=device).float()

    def run_on_training_end(self):
        self.hook_manager.process_hook_executes(self, self.run_on_training_end.__name__, __file__, HookAt.ALL)


def p_self_play(net, tree, dev, num_g, mem, dir_path: str or None = None):
    data = []
    for game in range(num_g):
        game_results = tree.play_one_game(net, dev, dir_path=dir_path, calculate_avg_num_children=game == num_g - 1)
        if mem is not None:
            mem.add_list(game_results)
        else:
            data.extend(game_results)
    return data


def c_p_self_play(net, tree, device, num_g, shared_storage: SharedStorage, num_worker_iters: int,
                  dir_path: str or None = None):
    net = net.to(device)
    net.eval()
    print("Starting continuous self play.")
    for iter_ in range(num_worker_iters):
        for game in range(num_g):
            net.load_state_dict(shared_storage.get_stable_network_params())
            game_results = tree.play_one_game(net, device, dir_path=dir_path,
                                              calculate_avg_num_children=game == num_g - 1)
            shared_storage.get_mem_buffer().add_list(game_results)
