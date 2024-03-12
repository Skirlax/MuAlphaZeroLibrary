import os
import shutil
import numpy as np
import optuna
# import pygraphviz
import torch as th
from IPython import get_ipython


from mu_alpha_zero.AlphaZero.Network.nnet import AlphaZeroNet
from mu_alpha_zero.AlphaZero.constants import SAMPLE_AZ_ARGS as test_args
from mu_alpha_zero.mem_buffer import MemBuffer
from mu_alpha_zero.config import MuZeroConfig, Config, AlphaZeroConfig



class DotDict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def augment_experience_with_symmetries(game_experience: list, board_size) -> list:
    game_experience_ = []
    for state, pi, v, _ in game_experience:
        pi = np.array([x for x in pi.values()])
        game_experience_.append((state, pi, v))
        for axis, k in zip([0, 1], [1, 3]):
            state_ = np.rot90(state.copy(), k=k)
            pi_ = np.rot90(pi.copy().reshape(board_size, board_size), k=k).flatten()
            game_experience_.append((state_, pi_, v))
            del state_, pi_
            state_ = np.flip(state.copy(), axis=axis)
            pi_ = np.flip(pi.copy().reshape(board_size, board_size), axis=axis).flatten()
            game_experience_.append((state_, pi_, v))

    return game_experience_


def rotate_stack(state: np.ndarray, k: int):
    for dim in range(state.shape[0]):
        state[dim] = np.rot90(state[dim], k=k)
    return state


def flip_stack(state: np.ndarray, axis: int):
    for dim in range(state.shape[0]):
        state[dim] = np.flip(state[dim], axis=axis)
    return state


def make_channels(game_experience: list):
    experience = []
    for state, pi, v, current_player, move in game_experience:
        state = make_channels_from_single(state)
        experience.append((state, pi, v, current_player, move))

    return experience


def make_channels_from_single(state: np.ndarray):
    player_one_state = np.where(state == 1, 1, 0)  # fill with 1 where player 1 has a piece else 0
    player_minus_one_state = np.where(state == -1, 1, 0)  # fill with 1 where player -1 has a piece else 0
    empty_state = np.where(state == 0, 1, 0)  # fill with 1 where empty spaces else 0
    return np.stack([state, player_one_state, player_minus_one_state, empty_state], axis=0)


def mask_invalid_actions(probabilities: np.ndarray, observations: np.ndarray, board_size) -> np.ndarray:
    to_print = ""  # for debugging
    mask = np.where(observations != 0, -5, observations)
    mask = np.where(mask == 0, 1, mask)
    mask = np.where(mask == -5, 0, mask)
    valids = probabilities.reshape(-1, board_size ** 2) * mask.reshape(-1, board_size ** 2)
    valids_sum = valids.sum()
    if valids_sum == 0:
        # When no valid moves are available (shouldn't happen) sum of valids is 0, making the returned valids an array
        # of nan's (result of division by zero). In this case, we create a uniform probability distribution.
        to_print += f"Sum of valid probabilities is 0. Creating a uniform probability...\nMask:\n{mask}"
        valids = np.full(valids.shape, 1.0 / np.prod(valids.shape))
    else:
        valids = valids / valids_sum  # normalize

    if len(to_print) > 0:
        print(to_print, file=open("masking_message.txt", "w"))
    return valids


def mask_invalid_actions_batch(states: th.tensor) -> th.tensor:
    masks = []
    for state in states:
        np_state = state.detach().cpu().numpy()
        mask = np.where(np_state != 0, -5, np_state)
        mask = np.where(mask == 0, 1, mask)
        mask = np.where(mask == -5, 0, mask)
        masks.append(mask)

    return th.tensor(np.array(masks), dtype=th.float32).squeeze(1)


def check_args(args: dict):
    required_keys = ["num_net_channels", "num_net_in_channels", "net_dropout", "net_action_size", "num_simulations",
                     "self_play_games", "num_iters", "epochs", "lr", "max_buffer_size", "num_pit_games",
                     "random_pit_freq", "board_size", "batch_size", "tau", "c", "checkpoint_dir", "update_threshold"]

    for key in required_keys:
        if key not in args:
            raise KeyError(f"Missing key {key} in args dict. Please supply all required keys.\n"
                           f"Required keys: {required_keys}.")


def calculate_board_win_positions(n: int, k: int):
    return get_num_horizontal_conv_slides(n, k) * (n * 2 + 2) + 4 * sum(
        [get_num_horizontal_conv_slides(x, k) for x in range(k, n)])


def get_num_horizontal_conv_slides(board_size: int, kernel_size: int) -> int:
    return (board_size - kernel_size) + 1


def az_optuna_parameter_search(n_trials: int, init_net_path: str, storage: str, study_name: str, game, config: AlphaZeroConfig):
    """
    Performs a hyperparameter search using optuna. This method is meant to be called using the start_jobs.py script.
    For this method to work, a mysql database must be running on the storage address and an optuna study with the
    given name and the 'maximize' direction must exist.

    :param n_trials: num of trials to run the search for.
    :param init_net_path: The path to the initial network to use for all trials.
    :param storage: The mysql storage string. Specifies what database to use.
    :param study_name: Name of the study to use.
    :param game: The game instance to use.
    :param config: The config to use for the search.
    :return:
    """

    def objective(trial):
        num_mc_simulations = trial.suggest_int("num_mc_simulations", 60, 1600)
        num_self_play_games = trial.suggest_int("num_self_play_games", 50, 200)
        num_epochs = trial.suggest_int("num_epochs", 100, 400)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        temp = trial.suggest_float("temp", 0.5, 1.5)
        arena_temp = trial.suggest_float("arena_temp", 1e-2, 0.5)
        cpuct = trial.suggest_float("cpuct", 0.5, 5)
        log_epsilon = trial.suggest_float("log_epsilon", 1e-10, 1e-7, log=True)

        trial_config.num_simulations = num_mc_simulations
        trial_config.self_play_games = num_self_play_games
        trial_config.epochs = num_epochs
        trial_config.lr = lr
        trial_config.tau = temp
        trial_config.c = cpuct
        trial_config.arena_tau = arena_temp
        trial_config.num_iters = 5
        trial_config.log_epsilon = log_epsilon
        search_tree = McSearchTree(game.make_fresh_instance(), trial_config)
        trainer = Trainer.from_state_dict(init_net_path, trial_config, game, search_tree)
        print(f"Trial {trial.number} started.")
        trainer.train()
        win_freq = trainer.get_arena_win_frequencies_mean()
        trial.report(win_freq, trial.number)
        print(f"Trial {trial.number} finished with win freq {win_freq}.")
        del trainer
        return win_freq

    from mu_alpha_zero.AlphaZero.Network.trainer import Trainer  # import here to avoid circular imports
    from mu_alpha_zero.AlphaZero.MCTS.az_search_tree import McSearchTree
    trial_config = config
    trial_config.show_tqdm = False
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials)


def build_net_from_config(muzero_config: Config, device) -> AlphaZeroNet:
    network = AlphaZeroNet(muzero_config.num_net_in_channels, muzero_config.num_net_channels,
                           muzero_config.net_dropout, muzero_config.net_action_size,
                           muzero_config.az_net_linear_input_size)
    return network.to(device)


def build_all_from_config(muzero_alphazero_config: Config, device, lr=None, buffer_size=None) -> tuple[
    AlphaZeroNet, th.optim.Optimizer, MemBuffer]:
    if lr is None:
        lr = muzero_alphazero_config.lr
    if buffer_size is None:
        buffer_size = muzero_alphazero_config.max_buffer_size
    network = build_net_from_config(muzero_alphazero_config, device)
    optimizer = th.optim.Adam(network.parameters(), lr=lr)
    memory = MemBuffer(max_size=buffer_size)
    return network, optimizer, memory


def make_net_from_checkpoint(checkpoint_path: str, args: DotDict | None):
    if args is None:
        args = DotDict(test_args)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    net = build_net_from_config(args, device)
    data = th.load(checkpoint_path)
    net.load_state_dict(data["net"])
    return net


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'Shell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False


def upload_checkpoint_to_gdrive(files: list, not_notebook_ok: bool = False):
    is_nbt = is_notebook()
    if not is_nbt and not_notebook_ok:
        return
    if not is_nbt:
        raise RuntimeError("This method should only be called from a notebook.")

    for file in files:
        if not os.path.exists("/content/drive/MyDrive/Checkpoints") and os.path.exists("/content/drive/MyDrive"):
            os.mkdir("/content/drive/MyDrive/Checkpoints")
        shutil.copy(file, "/content/drive/MyDrive/Checkpoints")


def visualize_tree(root_node, output_file_name: str, depth_limit: int | None = None):
    graph = pygraphviz.AGraph()
    graph.graph_attr["label"] = "MCTS visualization"
    graph.node_attr["shape"] = "circle"
    graph.edge_attr["color"] = "blue"
    graph.node_attr["color"] = "gold"
    if depth_limit is None:
        depth_limit = float("inf")

    def make_graph(node, parent, g: pygraphviz.AGraph, d_limit: int):

        state_ = None
        if node.state is None:
            state_ = str(np.random.randint(low=0, high=5, size=parent.state.shape))
        else:
            state_ = str(node.state)
        g.add_node(state_)
        if parent != node:
            g.add_edge(str(parent.state), state_)
        if not node.was_visited() or d_limit <= 0:
            return
        # queue_ = deque(root_node.children.values())
        # depth = 1
        # num_children = 25
        # children_iterated = 0
        # parent = root_node

        for child in node.children.values():
            make_graph(child, node, g, d_limit=d_limit - 1 if depth_limit != float("inf") else depth_limit)

    make_graph(root_node, root_node, graph, d_limit=depth_limit)
    graph.layout(prog="dot")
    graph.draw(f"{output_file_name}.png")


def cpp_data_to_memory(data: list, memory: MemBuffer, board_size: int):
    # import pickle
    # test_data = pickle.load(open(f"{find_project_root()}/history.pkl","rb"))
    for game_data in data:
        for state, pi, v in game_data:
            state = th.tensor(state, dtype=th.float32).reshape(board_size, board_size)
            pi = th.tensor(pi, dtype=th.float32)
            memory.add((state, pi, v))

#
# if __name__ == "__main__":
#     cpp_data_to_memory(None,MemBuffer(max_size=10_000),test_args)
