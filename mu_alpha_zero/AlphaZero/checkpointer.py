import atexit
import os
import pickle
import sys

import torch as th

from mu_alpha_zero.AlphaZero.utils import DotDict
from mu_alpha_zero.General.utils import find_project_root
from mu_alpha_zero.config import Config


class CheckPointer:
    def __init__(self, checkpoint_dir: str | None, verbose: bool = True) -> None:
        self.__checkpoint_dir = checkpoint_dir
        self.make_dir()
        self.__checkpoint_num = self.initialize_checkpoint_num()
        self.__name_prefix = "improved_net_"
        self.verbose = verbose
        atexit.register(self.cleanup)

    def make_dir(self) -> None:
        if self.__checkpoint_dir is not None:
            os.makedirs(self.__checkpoint_dir, exist_ok=True)
            return
        root_dir = find_project_root()
        checkpoint_dir = f"{root_dir}/Checkpoints/NetVersions"
        self.__checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, net: th.nn.Module, opponent: th.nn.Module, optimizer: th.optim,
                        lr: float,
                        iteration: int, mu_alpha_zero_config: Config, name: str = None) -> None:
        if name is None:
            name = self.__name_prefix + str(self.__checkpoint_num)

        checkpoint_path = f"{self.__checkpoint_dir}/{name}.pth"
        th.save({
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr": lr,
            "iteration": iteration,
            "args": mu_alpha_zero_config.to_dict(),
            'opponent_state_dict': opponent.state_dict()
        }, checkpoint_path)
        self.print_verbose(f"Saved checkpoint to {checkpoint_path} at iteration {iteration}.")
        self.__checkpoint_num += 1

    def save_state_dict_checkpoint(self, net: th.nn.Module, name: str) -> None:
        checkpoint_path = f"{self.__checkpoint_dir}/{name}.pth"
        th.save(net.state_dict(), checkpoint_path)
        self.print_verbose(f"Saved state dict checkpoint.")

    def load_state_dict_checkpoint(self, net: th.nn.Module, name: str) -> None:
        checkpoint_path = f"{self.__checkpoint_dir}/{name}.pth"
        net.load_state_dict(th.load(checkpoint_path))
        self.print_verbose(f"Loaded state dict checkpoint.")

    def load_checkpoint_from_path(self, checkpoint_path: str) -> tuple:
        sys.modules['mem_buffer'] = sys.modules['mu_alpha_zero.mem_buffer']
        checkpoint = th.load(checkpoint_path)
        self.print_verbose(f"Restoring checkpoint {checkpoint_path} made at iteration {checkpoint['iteration']}.")
        if "memory" in checkpoint:
            memory = checkpoint["memory"]
        else:
            memory = None
        return checkpoint["net"], checkpoint["optimizer"], memory, checkpoint["lr"], \
            DotDict(checkpoint["args"]), checkpoint["opponent_state_dict"]

    def load_checkpoint_from_num(self, checkpoint_num: int) -> tuple:
        checkpoint_path = f"{self.__checkpoint_dir}/{self.__name_prefix}{checkpoint_num}.pth"
        return self.load_checkpoint_from_path(checkpoint_path)

    def clear_checkpoints(self) -> None:
        # This method doesn't obey the verbose flag as it's a destructive operation.

        print("Clearing all checkpoints.")
        answer = input("Are you sure?? (y/n): ")
        if answer != "y":
            print("Aborted.")
            return
        for file_name in os.listdir(self.__checkpoint_dir):
            os.remove(f"{self.__checkpoint_dir}/{file_name}")
        print(f"Cleared {len(os.listdir(self.__checkpoint_dir))} saved checkpoints (all).")

    def save_temp_net_checkpoint(self, net) -> None:
        process_pid = os.getpid()
        os.makedirs(f"{self.__checkpoint_dir}/Temp", exist_ok=True)
        checkpoint_path = f"{self.__checkpoint_dir}/Temp/temp_net_{process_pid}.pth"
        th.save(net.state_dict(), checkpoint_path)

    def load_temp_net_checkpoint(self, net) -> None:
        process_pid = os.getpid()
        checkpoint_path = f"{self.__checkpoint_dir}/Temp/temp_net_{process_pid}.pth"
        net.load_state_dict(th.load(checkpoint_path))

    def initialize_checkpoint_num(self) -> int:
        return len([x for x in os.listdir(self.__checkpoint_dir) if x.endswith(".pth")])

    def get_highest_checkpoint_num(self) -> int:
        return max([int(file_name.split("_")[2].split(".")[0]) for file_name in os.listdir(self.__checkpoint_dir)])

    def get_temp_path(self) -> str:
        return f"{self.__checkpoint_dir}/Temp/temp_net.pth"

    def get_checkpoint_dir(self) -> str:
        return self.__checkpoint_dir

    def get_latest_name_match(self, name: str):
        name_matches = [os.path.join(self.__checkpoint_dir, x) for x in os.listdir(self.__checkpoint_dir) if name in x]
        name_matches.sort(key=lambda x: os.path.getctime(x))
        return name_matches[-1]

    def get_name_prefix(self):
        return self.__name_prefix

    def print_verbose(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def cleanup(self):
        import shutil
        shutil.rmtree(f"{self.__checkpoint_dir}/Temp/", ignore_errors=True)

    def save_losses(self, losses: list[float]):
        with open(f"{self.__checkpoint_dir}/training_losses.pkl", "wb") as f:
            pickle.dump(losses, f)
