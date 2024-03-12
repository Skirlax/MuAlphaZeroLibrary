import copy
from multiprocessing import Pool

import torch as th

from mu_alpha_zero.AlphaZero.MCTS.az_node import AlphaZeroNode
from mu_alpha_zero.AlphaZero.utils import augment_experience_with_symmetries, mask_invalid_actions
from mu_alpha_zero.Game.tictactoe_game import TicTacToeGameManager
from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.network import GeneralNetwork
from mu_alpha_zero.General.search_tree import SearchTree
from mu_alpha_zero.config import AlphaZeroConfig


class McSearchTree(SearchTree):
    def __init__(self, game_manager: TicTacToeGameManager, alpha_zero_config: AlphaZeroConfig):
        self.game_manager = game_manager
        self.alpha_zero_config = alpha_zero_config
        self.root_node = None

    def play_one_game(self, network: GeneralNetwork, device: th.device) -> tuple[list, int, int, int]:
        """
        Plays a single game using the Monte Carlo Tree Search algorithm.

        Args:
            network: The neural network used for searching and evaluating moves.
            device: The device (e.g., CPU or GPU) on which the network is located.

        Returns:
            A tuple containing the game history, and the number of wins, losses, and draws.
            The game history is a list of tuples, where each tuple contains:
            - The game state multiplied by the current player
            - The policy vector (probability distribution over moves)
            - The game result (1 for a win, -1 for a loss, 0 for a draw)
            - The current player (-1 or 1)
        """
        # tau = self.args["tau"]
        state = self.game_manager.reset()
        current_player = 1
        game_history = []
        results = {"1": 0, "-1": 0, "D": 0}
        while True:
            pi, _ = self.search(network, state, current_player, device)
            move = self.game_manager.select_move(pi)
            # self.step_root([move])
            self.step_root(None)
            self.game_manager.play(current_player, self.game_manager.network_to_board(move))
            # pi = [x for x in pi.values()]
            game_history.append((state * current_player, pi, None, current_player))
            state = self.game_manager.get_board()
            r = self.game_manager.game_result(current_player, state)
            if r is not None:
                if r == current_player:
                    results["1"] += 1
                elif -1 < r < 1:
                    results["D"] += 1
                else:
                    results["-1"] += 1

                if -1 < r < 1:
                    game_history = [(x[0], x[1], r, x[3]) for x in game_history]
                else:
                    game_history = [(x[0], x[1], r * current_player * x[3], x[3]) for x in game_history]
                break
            current_player *= -1

        # game_history = make_channels(game_history)
        game_history = augment_experience_with_symmetries(game_history, self.game_manager.board_size)
        return game_history, results["1"], results["-1"], results["D"]

    def search(self, network, state, current_player, device, tau=None):
        """
        Perform a Monte Carlo Tree Search on the current state starting with the current player.
        :param tau:
        :param network:
        :param state:
        :param current_player:
        :param device:
        :return:
        """
        num_simulations = self.alpha_zero_config.num_simulations
        if tau is None:
            tau = self.alpha_zero_config.tau
        if self.root_node is None:
            self.root_node = AlphaZeroNode(current_player, times_visited_init=0)
        state_ = self.game_manager.get_canonical_form(state, current_player)
        # state_ = make_channels_from_single(state_)
        state_ = th.tensor(state_, dtype=th.float32, device=device).unsqueeze(0)
        probabilities, v = network.predict(state_, muzero=False)

        probabilities = mask_invalid_actions(probabilities, state.copy(), self.game_manager.board_size)
        probabilities = probabilities.flatten().tolist()
        self.root_node.expand(state, probabilities)
        for simulation in range(num_simulations):
            current_node = self.root_node
            path = [current_node]
            action = None
            while current_node.was_visited():
                current_node, action = current_node.get_best_child(c=self.alpha_zero_config.c)
                if current_node is None:  # This was for testing purposes
                    th.save(self.root_node, "root_node.pt")
                    th.save(network.state_dict(), f"network_none_checkpoint_{current_player}.pt")
                    raise ValueError("current_node is None")
                path.append(current_node)

            # leaf node reached
            next_state = self.game_manager.get_next_state(current_node.parent().state,
                                                          self.game_manager.network_to_board(action),
                                                          current_node.parent().current_player)
            next_state_ = self.game_manager.get_canonical_form(next_state, current_node.current_player)
            v = self.game_manager.game_result(current_node.current_player, next_state)
            if v is None:
                # next_state_ = make_channels_from_single(next_state_)
                next_state_ = th.tensor(next_state_, dtype=th.float32, device=device).unsqueeze(0)
                probabilities, v = network.predict(next_state_, muzero=False)
                probabilities = mask_invalid_actions(probabilities, next_state, self.game_manager.board_size)
                v = v.flatten().tolist()[0]
                probabilities = probabilities.flatten().tolist()
                current_node.expand(next_state, probabilities)

            self.backprop(v, path)

        return self.root_node.get_self_action_probabilities(tau=tau), None

    def backprop(self, v, path):
        """
        Backpropagates the value `v` through the search tree, updating the relevant nodes.

        Args:
            v (float): The value to be backpropagated.
            path (list): The path from the leaf node to the root node.

        Returns:
            None
        """
        for node in reversed(path):
            v *= -1
            node.total_value += v
            node.update_q(v)
            node.times_visited += 1

    def step_root(self, actions: list | None) -> None:
        if actions is not None:
            if self.root_node is not None:
                if not self.root_node.was_visited():
                    return
                for action in actions:
                    self.root_node = self.root_node.children[action]
                self.root_node.parent = None
        else:
            # reset root node
            self.root_node = None

    def make_fresh_instance(self):
        return McSearchTree(self.game_manager.make_fresh_instance(), self.alpha_zero_config)

    def self_play(self, net: GeneralNetwork, device: th.device, num_games: int, memory: GeneralMemoryBuffer) -> tuple[
        int, int, int]:
        wins_p1, wins_p2, draws = 0, 0, 0
        for game in range(num_games):
            game_results, wins_p1_, wins_p2_, draws_ = self.play_one_game(net, device)
            wins_p1 += wins_p1_
            wins_p2 += wins_p2_
            draws += draws_
            memory.add_list(game_results)

        return wins_p1, wins_p2, draws

    @staticmethod
    def parallel_self_play(nets: list, trees: list, memory: GeneralMemoryBuffer, device: th.device, num_games: int,
                           num_jobs: int):
        with Pool(num_jobs) as p:
            if not memory.is_disk:
                results = p.starmap(p_self_play,
                                    [(
                                        nets[i], trees[i], copy.deepcopy(device), num_games // num_jobs,
                                        None)
                                        for i in
                                        range(len(nets))])
            else:
                results = p.starmap(p_self_play,
                                    [(
                                        nets[i], trees[i], copy.deepcopy(device), num_games // num_jobs,
                                        copy.deepcopy(memory))
                                        for i in
                                        range(len(nets))])
        wins_p1, wins_p2, draws = 0, 0, 0
        for result in results:
            wins_p1 += result[0]
            wins_p2 += result[1]
            draws += result[2]
            if not memory.is_disk:
                memory.add_list(result[3])
        return wins_p1, wins_p2, draws


def p_self_play(net, tree, device, num_games, memory):
    wins_p1, wins_p2, draws = 0, 0, 0
    data = []
    for game in range(num_games):
        game_results, wp1, wp2, ds = tree.play_one_game(net, device)
        wins_p1 += wp1
        wins_p2 += wp2
        draws += ds
        if memory is not None:
            memory.add_list(game_results)
        else:
            data.extend(game_results)
    if memory is None:
        return wins_p1, wins_p2, draws, data
    return wins_p1, wins_p2, draws
