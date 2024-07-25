import os

import torch as th

from mu_alpha_zero import MuZero, MuZeroNet
from mu_alpha_zero.Game.asteroids import Asteroids
from mu_alpha_zero.Hooks.hook_callables import HookMethodCallable
from mu_alpha_zero.Hooks.hook_manager import HookManager
from mu_alpha_zero.Hooks.hook_point import HookPoint, HookAt

from mu_alpha_zero.config import MuZeroConfig
from mu_alpha_zero.mem_buffer import MemBuffer
import matplotlib.pyplot as plt


def main():
    game = Asteroids()
    config = MuZeroConfig()
    hook_manager = HookManager()
    hook_cls = CollectInfo()
    hook_manager.register_method_hook(HookPoint(HookAt.MIDDLE, "networks.py", "train_net"),
                                      HookMethodCallable(hook_cls.training_hook, ()))
    # print(torch.cuda.is_available())
    config.num_steps = 500
    config.epochs = 3600
    config.num_iters = 20
    config.num_workers = 8
    config.num_simulations = 200
    config.update_threshold = 0.55
    config.c = 1.25
    config.tau = 1
    config.zero_tau_after = 100
    print(th.cuda.is_available())
    # os.makedirs("/workspace/Data")
    # print(os.listdir("/workspace/Data"))
    scratch_dir = "/home/skyr/Downloads"
    config.pickle_dir = f"{scratch_dir}/Data".replace("//", "/")
    # config.log_dir = "/auto/vestec1-elixir/home/vvlcek/Logs"
    config.log_dir = f"{scratch_dir}/Logs".replace("//", "/")
    config.checkpoint_dir = f"{scratch_dir}/Checkpoints".replace("//", "/")
    mz = MuZero(game)
    memory = MemBuffer(config.max_buffer_size, disk=True, full_disk=False, dir_path=config.pickle_dir)
    mz.create_new(config, MuZeroNet, memory)
    mz.train()
    hook_cls.save_plot()
    hook_cls.save_error_experiences()


class CollectInfo:
    def __init__(self):
        self.losses = []
        self.error_experiences = []

    def training_hook(self, cls: object, *args):
        self.losses.append(args[1])
        if args[-1] % 500 != 0:
            return
        exp_batch = args[0]
        loss_r = args[4]
        max_error = th.argmax(loss_r).item()
        self.error_experiences.append(exp_batch[max_error])
        exp_batch[max_error][-1].make_persistent()

    def save_plot(self):
        plt.plot(self.losses)
        plt.savefig("/auto/vestec1-elixir/home/vvlcek/Logs/losses.png")

    def save_error_experiences(self):
        with open("/auto/vestec1-elixir/home/vvlcek/Logs/error_experiences.pkl", "wb") as file:
            th.save(self.error_experiences, file)

    def plot_random_img(self):
        data = th.load("/auto/vestec1-elixir/home/vvlcek/Logs/error_experiences.pkl")
        random_idx = th.randint(0, len(data), (1,)).item()
        img = data[random_idx][-1]
        plt.imshow(img)


if __name__ == "__main__":
    main()
