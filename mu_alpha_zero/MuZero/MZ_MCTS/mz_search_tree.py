import copy
import gc
from torch.multiprocessing import Pool

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
    scale_hidden_state, mask_invalid_actions, scale_reward_value
from mu_alpha_zero.config import MuZeroConfig
from mu_alpha_zero.mem_buffer import MuZeroFrameBuffer
import wandb


class MuZeroSearchTree(SearchTree):

    def __init__(self, game_manager: MuZeroGame, muzero_config: MuZeroConfig, hook_manager: HookManager or None = None):
        self.game_manager = game_manager
        self.muzero_config = muzero_config
        self.hook_manager = hook_manager if hook_manager is not None else HookManager()
        self.buffer = MuZeroFrameBuffer(self.muzero_config.frame_buffer_size, self.game_manager.get_noop(),
                                        self.muzero_config.net_action_size)
        self.min_max_q = [float("inf"), -float("inf")]

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
            self.buffer.init_buffer(self.game_manager.get_state_for_player(state, -2), -player)
        data = []

        for step in range(num_steps):
            pi, (v, latent) = self.search(network_wrapper, state, player, device, calculate_avg_num_children=(
                    calculate_avg_num_children and step == 0))
            move = self.game_manager.select_move(pi, tau=self.muzero_config.tau)
            _, pred_v = network_wrapper.prediction_forward(latent.unsqueeze(0), predict=True)
            state, rew, done = self.game_manager.frame_skip_step(move, player, frame_skip=frame_skip)
            if self.muzero_config.multiple_players:
                # We want to signify that player can see this state but their not the one acting based on it
                # (this is important mainly if part of the observation is player specific).
                # Without this the player buffer would contain only state at time step t and then t + 2, t + 4, ...
                sign = np.sign(player)
                player_observing_flag = sign * (sign * player + 1)
                state = self.game_manager.get_state_for_player(state, player_observing_flag)
                scaled_move = scale_action(move, self.game_manager.get_num_actions())
                self.buffer.add_frame(state, scaled_move, player)
            rew = scale_reward_value(float(rew))
            state = resize_obs(state, self.muzero_config.target_resolution, self.muzero_config.resize_images)
            state = scale_state(state, self.muzero_config.scale_state)
            if done:
                break
            move = scale_action(move, self.game_manager.get_num_actions())
            frame = self.buffer.concat_frames(player).detach().cpu().numpy()
            data.append(
                (pi, v, (rew, move, float(pred_v[0]), player),
                 frame if dir_path is None else LazyArray(frame, dir_path)))
            if self.muzero_config.multiple_players:
                player = -player
            self.buffer.add_frame(state, move, player)

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
        for node in reversed(path):
            gamma = 1 if G == v else self.muzero_config.gamma
            if self.muzero_config.multiple_players:
                G = node.reward + gamma * (-G)
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
