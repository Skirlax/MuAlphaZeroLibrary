from torch.multiprocessing import set_start_method

from mu_alpha_zero.General.utils import not_zero

set_start_method('spawn', force=True)
from copy import deepcopy

from typing import Type

import torch as th
from tqdm import tqdm
from mu_alpha_zero.AlphaZero.Arena.arena import Arena
from mu_alpha_zero.AlphaZero.Arena.players import RandomPlayer, Player,NetPlayer
from mu_alpha_zero.AlphaZero.MCTS.az_search_tree import McSearchTree
from mu_alpha_zero.AlphaZero.Network.nnet import AlphaZeroNet
from mu_alpha_zero.AlphaZero.checkpointer import CheckPointer
from mu_alpha_zero.AlphaZero.logger import LoggingMessageTemplates, Logger
from mu_alpha_zero.AlphaZero.utils import build_all_from_config
from mu_alpha_zero.General.arena import GeneralArena
from mu_alpha_zero.General.az_game import AlphaZeroGame
from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.network import GeneralNetwork
from mu_alpha_zero.General.search_tree import SearchTree
from mu_alpha_zero.MuZero.JavaGateway.java_manager import JavaManager
from mu_alpha_zero.config import Config
from mu_alpha_zero.mem_buffer import MemBuffer


# joblib.parallel.BACKENDS['multiprocessing'].use_dill = True


class Trainer:
    def __init__(self, network: GeneralNetwork, game: AlphaZeroGame,
                 optimizer: th.optim, memory: GeneralMemoryBuffer,
                 muzero_alphazero_config: Config, checkpointer: CheckPointer,
                 search_tree: SearchTree, net_player: Player,
                 device, headless: bool = True,
                 opponent_network_override: th.nn.Module or None = None,
                 arena_override: GeneralArena or None = None,
                 java_manager: JavaManager = None) -> None:
        self.muzero_alphazero_config = muzero_alphazero_config
        self.device = device
        self.headless = headless
        self.game_manager = game
        self.mcts = search_tree
        self.net_player = net_player
        self.network = network
        self.opponent_network = self.network.make_fresh_instance() if opponent_network_override is None else opponent_network_override
        self.optimizer = optimizer
        self.memory = memory
        self.java_manager = java_manager
        self.arena = Arena(self.game_manager, self.muzero_alphazero_config,
                           self.device) if arena_override is None else arena_override
        self.checkpointer = checkpointer
        self.logger = Logger(logdir=self.muzero_alphazero_config.log_dir,
                             token=self.muzero_alphazero_config.pushbullet_token)
        self.arena_win_frequencies = []
        self.losses = []

    @classmethod
    def from_checkpoint(cls, net_class: Type[GeneralNetwork], tree_class: Type[SearchTree],
                        net_player_class: Type[Player],
                        checkpoint_path: str, checkpoint_dir: str,
                        game: AlphaZeroGame, headless: bool = True,
                        checkpointer_verbose: bool = False,
                        arena_override: GeneralArena or None = None,
                        mem: GeneralMemoryBuffer or None = None):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        checkpointer = CheckPointer(checkpoint_dir, verbose=checkpointer_verbose)

        network_dict, optimizer_dict, memory, lr, args, opponent_dict = checkpointer.load_checkpoint_from_path(
            checkpoint_path)
        conf = Config.from_args(args)
        tree = tree_class(game.make_fresh_instance(), conf)
        if "fc1.weight" in network_dict:
            conf.az_net_linear_input_size = network_dict["fc1.weight"].shape[1]
        network = net_class.make_from_config(conf).to(device)
        opponent_network = network.make_fresh_instance().to(device)
        optimizer = th.optim.Adam(network.parameters(), lr=lr)
        # opponent_network = build_net_from_args(args, device)
        net_player = net_player_class(game.make_fresh_instance(),
                                      **{"network": network, "monte_carlo_tree_search": tree})
        network.load_state_dict(network_dict)
        opponent_network.load_state_dict(opponent_dict)

        try:
            optimizer.load_state_dict(optimizer_dict)
        except ValueError:
            print("Couldn't load optimizer dict.")
        if memory is None:
            memory = mem
        return cls(network, game, optimizer, memory, conf, checkpointer, tree, net_player, device, headless=headless,
                   arena_override=arena_override)

    @classmethod
    def create(cls, muzero_alphazero_config: Config, game: AlphaZeroGame, network: GeneralNetwork,
               search_tree: SearchTree,
               net_player: Player,
               headless: bool = True,
               checkpointer_verbose: bool = False,
               arena_override: GeneralArena or None = None,
               memory_override: GeneralMemoryBuffer or None = None,
               java_manager: JavaManager or None = None):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        _, optimizer, mem = build_all_from_config(muzero_alphazero_config, device)
        memory = mem if memory_override is None else memory_override
        checkpointer = CheckPointer(muzero_alphazero_config.checkpoint_dir, verbose=checkpointer_verbose)
        return cls(network, game, optimizer, memory, muzero_alphazero_config, checkpointer, search_tree, net_player,
                   device,
                   headless=headless, arena_override=arena_override, java_manager=java_manager)

    @classmethod
    def from_state_dict(cls, path: str, muzero_alphazero_config: Config, game: AlphaZeroGame, search_tree: SearchTree,
                        headless: bool = True,
                        checkpointer_verbose: bool = False, java_manager: JavaManager or None = None):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        net, optimizer, memory = build_all_from_config(muzero_alphazero_config, device)
        checkpointer = CheckPointer(muzero_alphazero_config.checkpoint_dir, verbose=checkpointer_verbose)
        net.load_state_dict(th.load(path))
        net_player = NetPlayer(game.make_fresh_instance(), **{"network": net, "monte_carlo_tree_search": search_tree})
        return cls(net, game, optimizer, memory, muzero_alphazero_config, checkpointer, search_tree, net_player, device,
                   headless=headless,
                   java_manager=java_manager)

    def train(self) -> AlphaZeroNet:
        self.opponent_network.to(self.device)
        self.logger.log(LoggingMessageTemplates.TRAINING_START(self.muzero_alphazero_config.num_iters))
        self.opponent_network.load_state_dict(self.network.state_dict())
        # self.checkpointer.save_state_dict_checkpoint(self.network, "h_search_network")
        num_iters = self.muzero_alphazero_config.num_iters
        epochs = self.muzero_alphazero_config.epochs
        num_simulations = self.muzero_alphazero_config.num_simulations
        self_play_games = self.muzero_alphazero_config.self_play_games
        self.network.eval()
        # self.logger.pushbullet_log(f"Cuda status: {th.cuda.is_available()}")
        for i in self.make_tqdm_bar(range(num_iters), "Training Progress", 0):
            if i >= self.muzero_alphazero_config.zero_tau_after:
                self.muzero_alphazero_config.arena_tau = 0
            with th.no_grad():
                n_jobs = self.muzero_alphazero_config.num_workers
                self.logger.log(LoggingMessageTemplates.SELF_PLAY_START(self_play_games))

                if self.muzero_alphazero_config.num_workers > 1:
                    wins_p1, wins_p2, game_draws = self.mcts.parallel_self_play(self.make_n_networks(n_jobs),
                                                                                self.make_n_trees(n_jobs),
                                                                                self.memory, self.device,
                                                                                self_play_games,
                                                                                n_jobs)
                    if isinstance(self.memory, MemBuffer) and self.memory.is_disk and self.memory.full_disk:
                        self.memory = self.memory.make_fresh_instance()
                else:
                    wins_p1, wins_p2, game_draws = self.mcts.self_play(self.network, self.device, self_play_games,
                                                                       self.memory)
                self.logger.log(LoggingMessageTemplates.SELF_PLAY_END(wins_p1, wins_p2, game_draws, not_zero))
                self.logger.pushbullet_log("Finished self-play.")

            self.checkpointer.save_temp_net_checkpoint(self.network)
            self.logger.log(LoggingMessageTemplates.SAVED("temp checkpoint", self.checkpointer.get_temp_path()))
            self.checkpointer.load_temp_net_checkpoint(self.opponent_network)
            self.logger.log(LoggingMessageTemplates.LOADED("opponent network", self.checkpointer.get_temp_path()))
            self.network.train()
            self.logger.log(LoggingMessageTemplates.NETWORK_TRAINING_START(epochs))
            mean_loss, losses = self.network.train_net(self.memory, self.muzero_alphazero_config)
            self.logger.log(LoggingMessageTemplates.NETWORK_TRAINING_END(mean_loss))
            self.logger.pushbullet_log(f"Mean loss: {mean_loss}, Max loss: {max(losses)}, Min loss: {min(losses)}")
            self.losses.extend(losses)
            self.checkpointer.save_losses(self.losses)
            self.checkpointer.save_checkpoint(self.network, self.opponent_network, self.optimizer,
                                              self.muzero_alphazero_config.lr, i, self.muzero_alphazero_config,
                                              name="latest_trained_net")

            num_games = self.muzero_alphazero_config.num_pit_games
            update_threshold = self.muzero_alphazero_config.update_threshold
            p1_wins, p2_wins, draws, wins_total = self.run_pitting(num_simulations, num_games)
            self.check_model(p1_wins, wins_total, update_threshold, i)
            self.run_pitting_random(num_simulations, num_games, i)

        important_args = {
            "numIters": self.muzero_alphazero_config.num_iters,
            "numSelfPlayGames": self.muzero_alphazero_config.self_play_games,
            "tau": self.muzero_alphazero_config.tau,
            "updateThreshold": self.muzero_alphazero_config.update_threshold,
            "mcSimulations": self.muzero_alphazero_config.num_simulations,
            "c": self.muzero_alphazero_config.c,
            "numPitGames": self.muzero_alphazero_config.num_pit_games
        }

        self.logger.log(LoggingMessageTemplates.TRAINING_END(important_args))
        self.logger.pushbullet_log(LoggingMessageTemplates.TRAINING_END_PSB())
        return self.network

    def make_n_networks(self, n: int) -> list[AlphaZeroNet]:
        """
        Make n identical copies of self.network using deepcopy.

        :param n: The number of copies to make.
        :return: A list of n identical networks.
        """
        return [deepcopy(self.network) for _ in range(n)]

    def make_n_trees(self, n: int) -> list[McSearchTree]:
        """
        Make n new search trees.
        :param n: The number of trees to create.
        :return: A list of n new search trees.
        """
        trees = []
        for i in range(n):
            # manager = self.game_manager.make_fresh_instance()
            tree = self.mcts.make_fresh_instance()
            trees.append(tree)
        return trees

    def get_arena_win_frequencies_mean(self):
        return sum(self.arena_win_frequencies) / not_zero(len(self.arena_win_frequencies))

    def save_latest(self, path):
        state_dict = self.network.state_dict()
        opponent_state_dict = self.opponent_network.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        th.save({
            'optimizer': optimizer_state_dict,
            'memory': self.memory,
            'lr': self.muzero_alphazero_config.lr,
            'net': state_dict,
            'opponent_state_dict': opponent_state_dict,
            'args': self.muzero_alphazero_config.to_dict()
        }, path)
        print("Saved latest model data to {}".format(path))

    def make_tqdm_bar(self, iterable, desc, position, leave=True):
        if self.muzero_alphazero_config.show_tqdm:
            return tqdm(iterable, desc=desc, position=position, leave=leave)
        else:
            return iterable

    def get_network(self):
        return self.network

    def get_tree(self):
        return self.mcts

    def get_args(self):
        return self.muzero_alphazero_config

    def run_pitting(self, num_simulations: int, num_games: int):
        self.network.eval()
        self.opponent_network.eval()
        p1 = self.net_player.make_fresh_instance()
        p1.set_network(self.network)
        p2 = self.net_player.make_fresh_instance()
        p2.set_network(self.opponent_network)
        self.logger.log(LoggingMessageTemplates.PITTING_START(p1.name, p2.name, num_games))
        p1_wins, p2_wins, draws = self.arena.pit(p1, p2, num_games, num_mc_simulations=num_simulations,
                                                 one_player=False)
        wins_total = not_zero(p1_wins + p2_wins)
        self.logger.log(LoggingMessageTemplates.PITTING_END(p1.name, p2.name, p1_wins,
                                                            p2_wins, wins_total, draws))
        self.arena_win_frequencies.append(p1_wins / num_games)
        return p1_wins, p2_wins, draws, wins_total

    def run_pitting_random(self, num_simulations: int, num_games: int, i: int):
        if i % self.muzero_alphazero_config.random_pit_freq != 0:
            return
        self.network.eval()
        with th.no_grad():
            random_player = RandomPlayer(self.game_manager.make_fresh_instance(), **{})
            # self.network, self.mcts,
            p1 = self.net_player.make_fresh_instance()
            p1.set_network(self.network)
            self.logger.log(LoggingMessageTemplates.PITTING_START(p1.name, random_player.name, num_games))
            p1_wins_random, p2_wins_random, draws_random = self.arena.pit(p1, random_player, num_games,
                                                                          num_mc_simulations=num_simulations)
            wins_total = not_zero(p1_wins_random + p2_wins_random)
            self.logger.log(
                LoggingMessageTemplates.PITTING_END(p1.name, random_player.name, p1_wins_random,
                                                    p2_wins_random, wins_total, draws_random))

    def check_model(self, p1_wins: int, wins_total: int, update_threshold: float, i: int):
        if p1_wins / wins_total > update_threshold:
            self.logger.log(LoggingMessageTemplates.MODEL_ACCEPT(p1_wins / wins_total,
                                                                 update_threshold))
            self.checkpointer.save_checkpoint(self.network, self.opponent_network, self.optimizer,
                                              self.muzero_alphazero_config.lr, i,
                                              self.muzero_alphazero_config)
            self.logger.log(LoggingMessageTemplates.SAVED("accepted model checkpoint",
                                                          self.checkpointer.get_checkpoint_dir()))
        else:
            self.logger.log(LoggingMessageTemplates.MODEL_REJECT(p1_wins / wins_total,
                                                                 update_threshold))
            self.checkpointer.load_temp_net_checkpoint(self.network)
            self.logger.log(LoggingMessageTemplates.LOADED("previous version checkpoint",
                                                           self.checkpointer.get_temp_path()))
        self.logger.pushbullet_log(LoggingMessageTemplates.ITER_FINISHED_PSB(i))

    def self_play_get_r_mean(self):
        self.network.eval()
        self.mcts.parallel_self_play(self.make_n_networks(self.muzero_alphazero_config.num_workers),
                                     self.make_n_trees(self.muzero_alphazero_config.num_workers),
                                     self.memory, self.device,
                                     self.muzero_alphazero_config.self_play_games,
                                     self.muzero_alphazero_config.num_workers)
        return sum([x[2][0] for x in self.memory.buffer]) / not_zero(len(self.memory.buffer))
