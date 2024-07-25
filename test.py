import pickle

from mu_alpha_zero import MuZero
from mu_alpha_zero import MuZeroNet
from mu_alpha_zero.Game.asteroids import Asteroids
from mu_alpha_zero.Hooks.hook_callables import HookMethodCallable
from mu_alpha_zero.Hooks.hook_manager import HookManager
from mu_alpha_zero.Hooks.hook_point import HookPoint, HookAt
from mu_alpha_zero.config import MuZeroConfig
from mu_alpha_zero.mem_buffer import MemBuffer
import uvicorn
import torch as th
from mu_alpha_zero.MuZero.JavaGateway.java_networks import JavaNetworks
from mu_alpha_zero.Game.tictactoe_game import TicTacToeGameManager
from mu_alpha_zero.config import AlphaZeroConfig
from mu_alpha_zero.AlphaZero.alpha_zero import AlphaZero
from mu_alpha_zero.AlphaZero.Network.nnet import OriginalAlphaZerNetwork
import wandb
import torch
def run():
    # print(get_site_packages_path())
    # set api key
    wandb.login(key="7cc858b676e39daf0f2fe73b253a00d1abbe113b")
    game = Asteroids()
    mz = MuZero(game)
    conf = MuZeroConfig()
    hook_manager = HookManager()
    hook_cls = HookClass()
    hook_manager.register_method_hook(HookPoint(HookAt.MIDDLE, "networks.py", "train_net"),
                                      HookMethodCallable(hook_cls.hook_method, ()))
    conf.num_workers = 1
    conf.self_play_games = 1
    conf.num_steps = 10
    conf.num_iters = 1
    # conf.use_original = False
    conf.az_net_linear_input_size = [36,72,144]
    conf.log_dir = r"C:\Users\Skyr\PycharmProjects\testMuAlphaZeroLib\Logs"
    conf.checkpoint_dir = r"C:\Users\Skyr\PycharmProjects\testMuAlphaZeroLib\Checkpoints"
    conf.pickle_dir = "/home/skyr/PycharmProjects/testMuAlphaZeroLib/Data"
    wandb.init(project="MuZero", config=conf.to_dict())
    memory = MemBuffer(conf.max_buffer_size, disk=True, full_disk=False, dir_path=conf.pickle_dir,
                       hook_manager=hook_manager)
    mz.create_new(conf, MuZeroNet, memory, headless=True, checkpointer_verbose=False, hook_manager=hook_manager)
    # mz.from_checkpoint(MuZeroNet, memory, "/home/skyr/Downloads/latest_trained_net.pth", conf.checkpoint_dir)
    # th.save(mz.net.state_dict(), "/home/skyr/PycharmProjects/testMuAlphaZeroLib/Checkpoints/h_search_network.pth")
    # net = mz.net.to("cuda")
    # net.eval()
    # net_scripted = th.jit.script(net)
    # th.jit.save(net_scripted, r"C:\Users\Skyr\PycharmProjects\testMuAlphaZeroLib\Checkpoints\script_exported_net.pth")
    # net_scripted.prediction_forward(th.rand(1, 256, 6, 6).to("cuda"), True)
    # start = time.time()
    # result = CppSelfPlay.runParallelSelfPlay(
    #     "C:\\Users\\Skyr\\PycharmProjects\\testMuAlphaZeroLib\\Checkpoints\\script_exported_net.pth",
    #     game, conf.to_dict(), conf.self_play_games, 2, game.get_noop())
    # print(time.time() - start)
    # print(result)
    # state = game.reset()
    mz.train()
    print(hook_cls.losses)


def run_alpha_zero():
    conf = AlphaZeroConfig()
    game = TicTacToeGameManager(3, headless=True, num_to_win=3)
    az = AlphaZero(game)
    conf.board_size = 3
    conf.num_workers = 1
    conf.net_action_size = 9
    # conf.az_net_linear_input_size = 512
    conf.self_play_games = 10
    conf.az_net_linear_input_size = [9,18]
    memory = MemBuffer(10_000)
    az.create_new(conf,OriginalAlphaZerNetwork,memory,checkpointer_verbose=False)
    az.train()


class HookClass:
    def __init__(self):
        self.losses = []

    def hook_method(self, cls: object, *args):
        self.losses.append(args[1])


def test():
    import torch

    tensor = torch.zeros(3, 5,dtype=th.int64)
    index = torch.tensor([[0, 1, 2, 0, 1],
                          [1, 2, 0, 1, 2],
                          [2, 0, 1, 2, 0]])
    src = torch.tensor([[10, 20, 30, 40, 50],
                        [50, 40, 30, 20, 10],
                        [60, 70, 80, 90, 100]])

    tensor.scatter_(1, index, src)
    print(tensor)

def scalar_to_support_2(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    single_support = logits[0,0,:].tolist()
    non_zero_idxs = [idx for idx,x in enumerate(single_support) if x != 0]
    print(x[0,0].item())
    print(non_zero_idxs)
    print([x for x in single_support if x != 0])
    return logits

def run_server():
    game = Asteroids()
    game_api = GymEnvWrapper(game)
    uvicorn.run(game_api.app, host="0.0.0.0", port=8000)


def just_trace():
    net = MuZeroNet.make_from_config(MuZeroConfig())
    net = net.to("cuda")
    net.eval()
    # th.jit.trace(net.representation_forward, th.rand(1, 128, 96, 96))
    rep_traced, dyn_traced, pred_traced = net.trace_all()
    rep_traced.save("rep_traced.pt")
    dyn_traced.save("dyn_traced.pt")
    pred_traced.save("pred_traced.pt")


if __name__ == "__main__":
    # run_alpha_zero()
    # run()
    # test()
    from mu_alpha_zero.MuZero.utils import scalar_to_support
    scalar_to_support(th.full((32,1),(1e9)),300)
    # scalar_to_support(th.full((256,10),10.7),300)
    # run_server()
    # just_trace()
