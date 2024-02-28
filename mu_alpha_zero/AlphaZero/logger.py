import atexit
import datetime
import logging
import os

from pushbullet import API

from mu_alpha_zero.General.utils import find_project_root


class Logger:
    def __init__(self, logdir: str or None, token: str or None = None) -> None:
        self.logdir = self.init_logdir(logdir)
        os.makedirs(self.logdir, exist_ok=True)
        self.logger = logging.getLogger("AlphaZeroLogger")
        self.logger.setLevel(logging.DEBUG)
        self.file_handler = logging.FileHandler(
            f"{self.logdir}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        self.file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.file_handler)
        self.is_token_set = False
        self.api = API()
        self.init_api_token(token)
        formatter = logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s")
        self.file_handler.setFormatter(formatter)
        atexit.register(self.cleanup)

    def log(self, msg: str, level: str = "debug") -> None:
        getattr(self.logger, level)(msg)

    def init_logdir(self,logdir: str or None):
        if logdir is None:
            return f"{find_project_root()}/Logs/ProgramLogs"
        else:
            return logdir

    def init_api_token(self, token: str or None) -> None:
        if token is None:
            return
        self.api.set_token(token)
        self.is_token_set = True

    def pushbullet_log(self, msg: str, algorithm: str = "MuZero") -> None:
        if not self.is_token_set:
            return
        try:
            self.api.send_note(f"{algorithm} training notification.", msg)
        except Exception as e:
            print(e)

    def clear_logdir(self):
        for file_name in os.listdir(self.logdir):
            os.remove(f"{self.logdir}/{file_name}")

    def cleanup(self) -> None:
        self.file_handler.close()
        self.logger.removeHandler(self.file_handler)


class LoggingMessageTemplates:

    @staticmethod
    def PITTING_START(name1: str, name2: str, num_games: int):
        return f"Starting pitting between {name1} and {name2} for {num_games} games."

    @staticmethod
    def PITTING_END(name1: str, name2: str, wins1: int, wins2: int, total: int, draws: int):
        return (f"Pitting ended between {name1} and {name2}. "
                f"Player 1 win frequency: {wins1 / total}. "
                f"Player 2 win frequency: {wins2 / total}. Draws: {draws}.")

    @staticmethod
    def SELF_PLAY_START(num_games: int):
        return f"Starting self play for {num_games} games."

    @staticmethod
    def SELF_PLAY_END(wins1: int, wins2: int, draws: int, not_zero_fn: callable):
        if wins1 is None or wins2 is None or draws is None:
            return "Self play ended. Results not available (This is expected if you are running MuZero)."
        return (f"Self play ended. Player 1 win frequency: {wins1 / (not_zero_fn(wins1 + wins2 + draws))}. "
                f"Player 2 win frequency: {wins2 / (not_zero_fn(wins1 + wins2 + draws))}. Draws: {draws}.")

    @staticmethod
    def NETWORK_TRAINING_START(num_epchs: int):
        return f"Starting network training for {num_epchs} epochs."

    @staticmethod
    def NETWORK_TRAINING_END(mean_loss: float):
        return f"Network training ended. Mean loss: {mean_loss}"

    @staticmethod
    def MODEL_REJECT(num_wins: float, update_threshold: float):
        return (f"!!! Model rejected, restoring previous version. Win rate: {num_wins}. "
                f"Update threshold: {update_threshold} !!!")

    @staticmethod
    def MODEL_ACCEPT(num_wins: float, update_threshold: float):
        return (
            f"!!! Model accepted, keeping current version. Win rate: {num_wins}. Update threshold: {update_threshold}"
            f" !!!")

    @staticmethod
    def TRAINING_START(num_iters: int):
        return f"Starting training for {num_iters} iterations."

    @staticmethod
    def TRAINING_END(args_used: dict):
        args_used_str = ""
        for key, value in args_used.items():
            args_used_str += f"{key}: {value}, "
        return f"Training ended. Args used: {args_used_str[:-2]}"

    @staticmethod
    def SAVED(type_: str, path: str):
        return f"Saved {type_} to {path}"

    @staticmethod
    def LOADED(type_: str, path: str):
        return f"Restored {type_} from {path}"

    @staticmethod
    def ITER_FINISHED_PSB(iter: int):
        return f"Iteration {iter} of the algorithm training finished!"

    @staticmethod
    def TRAINING_END_PSB():
        return "Algorithm Training finished, you can collect the results :)"
