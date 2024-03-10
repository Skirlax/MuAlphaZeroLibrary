# --- Linear input sizes for different board and atari sizes ---
# 4608 for (5x5) board
# 512 for (3x3) board
# 32768 for (10x10) board
# 18432 for (8x8) board
# 8192 for (6x6) atari

from dataclasses import dataclass

@dataclass
class Config:
    num_net_channels: int = 512
    num_net_in_channels: int = 1
    net_dropout: float = 0.3
    net_action_size: int = 14
    num_simulations: int = 800
    self_play_games: int = 100
    num_iters: int = 50
    epochs: int = 1600
    lr: float = 1.2166489157239912e-05
    max_buffer_size: int = 70_000
    num_pit_games: int = 40
    random_pit_freq: int = 2
    batch_size: int = 255
    tau: float = 1
    arena_tau: float = 1e-2
    c: float = 1
    checkpoint_dir: str = None
    update_threshold: float = 0.6
    num_workers: int = 5
    log_epsilon: float = 1e-9
    zero_tau_after: int = 5
    az_net_linear_input_size: int = 8192
    log_dir: str = "Logs"
    pushbullet_token: str = None

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_args(args: dict):
        missing = []
        config = MuZeroConfig()
        for key, val in args.items():
            if key not in config.to_dict().keys():
                missing.append(key)
            setattr(config, key, val)

        if len(missing) > 0:
            print(f"The following keys were missing from the default config and were added: {missing}.")
        return config


@dataclass
class MuZeroConfig(Config):
    num_net_channels: int = 512
    num_net_out_channels: int = 256
    num_net_in_channels: int = 1
    net_dropout: float = 0.3
    net_action_size: int = 14
    net_latent_size: int = 36
    num_simulations: int = 800
    self_play_games: int = 100
    K: int = 5
    gamma: float = 0.997
    frame_buffer_size: int = 32
    frame_skip: int = 4
    num_steps: int = 400
    num_iters: int = 50
    epochs: int = 1600
    lr: float = 1.2166489157239912e-05
    max_buffer_size: int = 70_000
    num_pit_games: int = 40
    random_pit_freq: int = 2
    batch_size: int = 255
    tau: float = 1
    arena_tau: float = 1e-2
    c: float = 1
    c2: int = 19652
    alpha: float = 0.8
    checkpoint_dir: str = None
    update_threshold: float = 0.6
    num_workers: int = 5
    log_epsilon: float = 1e-9
    zero_tau_after: int = 5
    beta: int = 1
    pickle_dir: str = "Pickles/Data"
    target_resolution: tuple[int, int] = (96, 96)
    az_net_linear_input_size: int = 8192
    log_dir: str = None
    pushbullet_token: str = None
    show_tqdm: bool = False


@dataclass
class AlphaZeroConfig(Config):
    board_size: int = 8
    num_net_channels: int = 512
    num_net_in_channels: int = 1
    net_dropout: float = 0.3
    net_action_size: int = board_size ** 2
    num_simulations: int = 1317
    self_play_games: int = 300
    num_iters: int = 50
    epochs: int = 500
    lr: float = 0.0032485504583772953
    max_buffer_size: int = 100_000
    num_pit_games: int = 40
    random_pit_freq: int = 3
    batch_size: int = 256
    tau: float = 1
    arena_tau: float = 0.04139160592420218
    c: float = 1
    checkpoint_dir: str = None
    update_threshold: float = 0.6
    minimax_depth: int = 4
    show_tqdm: bool = True
    num_workers: int = 5
    num_to_win: int = 5
    log_epsilon: float = 1e-9
    zero_tau_after: int = 5
    az_net_linear_input_size: int = 18432
    log_dir: str = None
    pushbullet_token: str = None
