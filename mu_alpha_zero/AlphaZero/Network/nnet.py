import atexit
import glob
import os
import time
from itertools import chain

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.nn.functional import mse_loss
from torch.nn.modules.module import T

from mu_alpha_zero.AlphaZero.checkpointer import CheckPointer
from mu_alpha_zero.AlphaZero.logger import Logger
from mu_alpha_zero.General.network import GeneralAlphZeroNetwork
from mu_alpha_zero.Hooks.hook_manager import HookManager
from mu_alpha_zero.Hooks.hook_point import HookAt
from mu_alpha_zero.config import AlphaZeroConfig, Config
from mu_alpha_zero.mem_buffer import MemBuffer
from mu_alpha_zero.shared_storage_manager import SharedStorage


class AlphaZeroNet(nn.Module, GeneralAlphZeroNetwork):
    def __init__(self, in_channels: int, num_channels: int, dropout: float, action_size: int, linear_input_size: int,
                 hook_manager: HookManager or None = None):
        super(AlphaZeroNet, self).__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.dropout_p = dropout
        self.action_size = action_size
        self.linear_input_size = linear_input_size
        self.hook_manager = hook_manager if hook_manager is not None else HookManager()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3)
        self.bn4 = nn.BatchNorm2d(num_channels)

        # Fully connected layers
        # 4608 (5x5) or 512 (3x3) or 32768 (10x10) or 18432 (8x8) # or 8192 for atari (6x6)
        self.fc1 = nn.Linear(self.linear_input_size, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.pi = nn.Linear(512, action_size)  # probability head
        self.v = nn.Linear(512, 1)  # value head
        atexit.register(self.clear_traces)

    def forward(self, x, muzero: bool = True):
        if not muzero:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.reshape(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)

        pi = F.log_softmax(self.pi(x), dim=1)
        v = F.tanh(self.v(x))

        return pi, v

    @th.no_grad()
    def predict(self, x, muzero=True):
        pi, v = self.forward(x, muzero=muzero)
        pi = th.exp(pi)
        return pi.detach().cpu().numpy(), v.detach().cpu().numpy()

    def to_traced_script(self, board_size: int = 10):
        return th.jit.trace(self, th.rand(1, 256, board_size, board_size).cuda())

    def trace(self, board_size: int) -> str:
        traced = self.to_traced_script(board_size=board_size)
        path = "traced.pt"
        traced.save(path)
        return path

    def clear_traces(self) -> None:
        from mu_alpha_zero.General.utils import find_project_root
        for trace_file in glob.glob(f"{find_project_root()}/Checkpoints/Traces/*.pt"):
            os.remove(trace_file)

    def make_fresh_instance(self):
        return AlphaZeroNet(self.in_channels, self.num_channels, self.dropout_p, self.action_size,
                            self.linear_input_size)

    @staticmethod
    def make_from_config(config: AlphaZeroConfig, hook_manager: HookManager or None = None):
        return AlphaZeroNet(config.num_net_in_channels, config.num_net_channels, config.net_dropout,
                            config.net_action_size, config.az_net_linear_input_size, hook_manager=hook_manager)

    def train_net(self, memory_buffer: MemBuffer, alpha_zero_config: AlphaZeroConfig):
        from mu_alpha_zero.AlphaZero.utils import mask_invalid_actions_batch
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        losses = []
        optimizer = th.optim.Adam(self.parameters(), lr=alpha_zero_config.lr)
        memory_buffer.shuffle()
        for epoch in range(alpha_zero_config.epochs):
            for experience_batch in memory_buffer(alpha_zero_config.batch_size):
                if len(experience_batch) <= 1:
                    continue
                states, pi, v = zip(*experience_batch)
                states = th.tensor(np.array(states), dtype=th.float32, device=device)
                pi = th.tensor(np.array(pi), dtype=th.float32, device=device)
                v = th.tensor(v, dtype=th.float32, device=device).unsqueeze(1)
                pi_pred, v_pred = self.forward(states, muzero=False)
                masks = mask_invalid_actions_batch(states)
                loss = mse_loss(v_pred, v) + self.pi_loss(pi_pred, pi, masks, device)
                losses.append(loss.item())
                # self.summary_writer.add_scalar("Loss", loss.item(), i * epochs + epoch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.hook_manager.process_hook_executes(self, self.train_net.__name__, __file__, HookAt.MIDDLE,
                                                        args=(experience_batch, loss.item(), epoch))

        self.hook_manager.process_hook_executes(self, self.train_net.__name__, __file__, HookAt.TAIL, args=(losses,))
        return sum(losses) / len(losses), losses

    def pi_loss(self, y_hat, y, masks, device: th.device):
        masks = masks.reshape(y_hat.shape).to(device)
        masked_y_hat = masks * y_hat
        return -th.sum(y * masked_y_hat) / y.size()[0]

    def to_shared_memory(self):
        for param in self.parameters():
            param.share_memory_()

    def run_at_training_end(self):
        self.hook_manager.process_hook_executes(self, self.run_at_training_end.__name__, __file__, HookAt.ALL)


class OriginalAlphaZeroNetwork(nn.Module, GeneralAlphZeroNetwork):

    def __init__(self, in_channels: int, num_channels: int, dropout: float, action_size: int,
                 linear_input_size: list[int], support_size: int,
                 state_linear_layers: int, pi_linear_layers: int, v_linear_layers: int, linear_head_hidden_size: int,
                 is_atari: bool,
                 latent_size: list[int] = [6, 6],
                 hook_manager: HookManager or None = None, num_blocks: int = 8, muzero: bool = False,
                 is_dynamics: bool = False, is_representation: bool = False):
        super(OriginalAlphaZeroNetwork, self).__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.dropout_p = dropout
        self.action_size = action_size
        self.linear_input_size = linear_input_size
        self.support_size = support_size
        self.num_blocks = num_blocks
        self.muzero = muzero
        self.latent_size = latent_size
        self.is_dynamics = is_dynamics
        self.is_representation = is_representation
        self.state_linear_layers = state_linear_layers
        self.pi_linear_layers = pi_linear_layers
        self.v_linear_layers = v_linear_layers
        self.linear_head_hidden_size = linear_head_hidden_size
        self.is_atari = is_atari
        self.optimizer = None
        self.hook_manager = hook_manager if hook_manager is not None else HookManager()

        self.conv1 = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([OriginalAlphaZeroBlock(num_channels, num_channels) for _ in range(num_blocks)])
        if not is_representation:
            self.value_head = ValueHead(muzero, linear_input_size[0], support_size, num_channels, v_linear_layers,
                                        linear_head_hidden_size)
        else:
            self.value_head = th.nn.Identity()
        if is_dynamics or is_representation:
            self.policy_state_head = StateHead(linear_input_size[2], num_channels, latent_size, state_linear_layers,
                                               linear_head_hidden_size)
        else:
            self.policy_state_head = PolicyHead(action_size, linear_input_size[1], num_channels, pi_linear_layers,
                                                linear_head_hidden_size)

    def forward(self, x, muzero: bool = False, return_support: bool = False):
        if not muzero:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        # x = self.dropout(x)
        val_h_output = self.value_head(x)
        pol_h_output = self.policy_state_head(x)
        if self.is_representation:
            return pol_h_output
        if not return_support and self.muzero:
            from mu_alpha_zero.MuZero.utils import invert_scale_reward_value
            # multiply arange by softmax probabilities
            val_h_output = th.exp(val_h_output)
            support_range = th.arange(-self.support_size, self.support_size + 1, 1, dtype=th.float32,
                                      device=x.device).unsqueeze(0)
            output = th.sum(val_h_output * support_range, dim=1)
            if self.is_atari:
                output = invert_scale_reward_value(output)
            return pol_h_output, output.unsqueeze(1)
        return pol_h_output, val_h_output

    @th.no_grad()
    def predict(self, x, muzero=True):
        pi, v = self.forward(x, muzero=muzero)
        pi = th.exp(pi)
        return pi.detach().cpu().numpy(), v.detach().cpu().numpy()

    def pi_loss(self, y_hat, y, masks, device: th.device):
        masks = masks.reshape(y_hat.shape).to(device)
        masked_y_hat = masks * y_hat
        return -th.sum(y * masked_y_hat) / y.size()[0]

    def make_fresh_instance(self):
        return OriginalAlphaZeroNetwork(self.in_channels, self.num_channels, self.dropout_p, self.action_size,
                                        self.linear_input_size, self.support_size,
                                        self.state_linear_layers, self.pi_linear_layers, self.v_linear_layers,
                                        self.linear_head_hidden_size,
                                        self.is_atari,
                                        self.latent_size,
                                        hook_manager=self.hook_manager,
                                        num_blocks=self.num_blocks, muzero=self.muzero, is_dynamics=self.is_dynamics)

    @classmethod
    def make_from_config(cls, config: Config, hook_manager: HookManager or None = None):
        return OriginalAlphaZeroNetwork(config.num_net_in_channels, config.num_net_channels, config.net_dropout,
                                        config.net_action_size, config.az_net_linear_input_size,
                                        hook_manager=hook_manager,
                                        state_linear_layers=config.state_linear_layers,
                                        pi_linear_layers=config.pi_linear_layers,
                                        v_linear_layers=config.v_linear_layers,
                                        linear_head_hidden_size=config.linear_head_hidden_size,
                                        num_blocks=config.num_blocks, muzero=config.muzero,
                                        is_atari=config.is_atari,
                                        support_size=config.support_size, latent_size=config.net_latent_size)

    def train_net(self, memory_buffer, muzero_alphazero_config: Config) -> tuple[float, list[float]]:
        if memory_buffer.train_length() <= 1:
            return 0, []
        losses = []
        if self.optimizer is None:
            self.optimizer = th.optim.Adam(self.parameters(), lr=muzero_alphazero_config.lr,
                                           weight_decay=muzero_alphazero_config.l2)
        # memory_buffer.shuffle()
        for epoch in range(muzero_alphazero_config.epochs):
            for experience_batch in memory_buffer.batch(muzero_alphazero_config.batch_size):
                loss, v_loss, pi_loss = self.calculate_loss(experience_batch, muzero_alphazero_config)
                losses.append(loss.item())
                wandb.log({"combined_loss": loss.item(), "loss_v": v_loss.item(), "loss_pi": pi_loss.item()})
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.hook_manager.process_hook_executes(self, self.train_net.__name__, __file__, HookAt.MIDDLE,
                                                        args=(experience_batch, loss.item(), epoch))

        self.hook_manager.process_hook_executes(self, self.train_net.__name__, __file__, HookAt.TAIL, args=(losses,))
        return sum(losses) / len(losses), losses

    def eval_net(self, memory_buffer, muzero_alphazero_config: Config) -> None:
        if memory_buffer.eval_length() <= 1:
            return
        if self.optimizer is None:
            self.optimizer = th.optim.Adam(self.parameters(), lr=muzero_alphazero_config.lr,
                                           weight_decay=muzero_alphazero_config.l2)
        # memory_buffer.shuffle(is_eval=True)
        for epoch in range(muzero_alphazero_config.eval_epochs):
            for experience_batch in memory_buffer.batch(muzero_alphazero_config.batch_size, is_eval=True):
                loss, v_loss, pi_loss = self.calculate_loss(experience_batch, muzero_alphazero_config)
                wandb.log(
                    {"eval_combined_loss": loss.item(), "eval_loss_v": v_loss.item(), "eval_loss_pi": pi_loss.item()})

    def calculate_loss(self, experience_batch, muzero_alphazero_config):
        from mu_alpha_zero.AlphaZero.utils import mask_invalid_actions_batch
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        states, pi, v, _, masks = experience_batch[0], experience_batch[1], experience_batch[2], experience_batch[3], \
            experience_batch[4]
        pi = [[y for y in x.values()] for x in pi]
        # game = [[y.frame,y.pi,y.v,y.action_mask] for y in experience_batch.datapoints]
        # states, pi, v, masks = zip(*game)
        states = th.tensor(np.array(states), dtype=th.float32, device=device)
        pi = th.tensor(np.array(pi), dtype=th.float32, device=device)
        v = th.tensor(v, dtype=th.float32, device=device).unsqueeze(1)
        masks = th.tensor(np.array(masks), dtype=th.float32, device=device)
        pi_pred, v_pred = self.forward(states, muzero=muzero_alphazero_config.muzero)
        v_loss = mse_loss(v_pred, v)
        pi_loss = self.pi_loss(pi_pred, pi, masks, device)
        loss = v_loss + pi_loss
        return loss, v_loss, pi_loss

    def continuous_weight_update(self, shared_storage: SharedStorage, alpha_zero_config: AlphaZeroConfig,
                                 checkpointer: CheckPointer or None,
                                 logger: Logger or None):
        wandb.init(project=alpha_zero_config.wandbd_project_name, name="Continuous Weight Update")
        alpha_zero_config.epochs = 1
        alpha_zero_config.eval_epochs = 50
        self.train()
        while shared_storage.train_length() < 200:
            time.sleep(5)
        for iter_ in range(alpha_zero_config.num_worker_iters):
            # print(iter_)
            # if not shared_storage.get_was_pitted():
            #     print("Waiting for pitting to finish")
            #     time.sleep(5)
            #     continue
            if shared_storage.get_experimental_network_params() is None:
                params = shared_storage.get_stable_network_params()
            else:
                params = shared_storage.get_experimental_network_params()
            self.load_state_dict(params)
            avg_iter_losses = self.train_net(shared_storage, alpha_zero_config)
            shared_storage.set_experimental_network_params(self.state_dict())
            # shared_storage.set_was_pitted(False)
            if iter_ % alpha_zero_config.eval_interval == 0 and iter_ != 0:
                self.eval_net(shared_storage, alpha_zero_config)


class OriginalAlphaZeroBlock(th.nn.Module):
    def __init__(self, in_channels: int, num_channels: int):
        super(OriginalAlphaZeroBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x_skip = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += x_skip
        return F.relu(x)


class ValueHead(th.nn.Module):
    def __init__(self, muzero: bool, linear_input_size: int, support_size: int, num_channels: int, num_layers: int,
                 linear_hidden_size: int):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(num_channels, linear_hidden_size, 1)
        self.bn = nn.BatchNorm2d(linear_hidden_size)
        self.fc1 = nn.Linear(linear_input_size, 256)
        # self.fc = HeadLinear(256, 256, num_layers, linear_hidden_size)
        if muzero:
            self.fc2 = nn.Linear(256, 2 * support_size + 1)
            self.act = nn.LogSoftmax(dim=1)
        else:
            self.fc2 = nn.Linear(256, 1)
            self.act = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc(x))
        x = self.fc2(x)
        return self.act(x)


class PolicyHead(th.nn.Module):
    def __init__(self, action_size: int, linear_input_size_policy: int, num_channels: int, num_layers: int,
                 linear_hidden_size: int):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(num_channels, linear_hidden_size, 1)
        self.bn = nn.BatchNorm2d(linear_hidden_size)
        self.fc1 = nn.Linear(linear_input_size_policy, 256)
        # self.fc = HeadLinear(256, 256, num_layers, linear_hidden_size)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class StateHead(th.nn.Module):
    def __init__(self, linear_input_size: int, out_channels: int, latent_size: list[int], num_layers: int,
                 linear_hidden_size: int):
        super(StateHead, self).__init__()
        self.conv = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.fc = HeadLinear(linear_input_size, out_channels, num_layers, linear_hidden_size)
        self.fc3 = nn.Linear(linear_input_size, latent_size[0] * latent_size[1] * out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        # x = F.relu(self.fc(x))
        x = F.relu(self.fc3(x))
        return x


class HeadLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, hidden_size: int):
        super(HeadLinear, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.fc = nn.Sequential(*[nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc(x))
        return self.fc2(x)
