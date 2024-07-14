import time

import numpy as np
import torch as th
import torch.nn.functional as F
import wandb
from torch import nn
from torch.nn.functional import mse_loss

from mu_alpha_zero.AlphaZero.Network.nnet import AlphaZeroNet as PredictionNet, OriginalAlphaZerNetwork
from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.mz_game import MuZeroGame
from mu_alpha_zero.General.network import GeneralMuZeroNetwork
from mu_alpha_zero.Hooks.hook_manager import HookManager
from mu_alpha_zero.Hooks.hook_point import HookAt
from mu_alpha_zero.MuZero.utils import match_action_with_obs_batch
from mu_alpha_zero.config import MuZeroConfig
from mu_alpha_zero.shared_storage_manager import SharedStorage


class MuZeroNet(th.nn.Module, GeneralMuZeroNetwork):
    def __init__(self, input_channels: int, dropout: float, action_size: int, num_channels: int, latent_size: list[int],
                 num_out_channels: int, linear_input_size: int or list[int], rep_input_channels: int,
                 use_original: bool, support_size: int, num_blocks: int,
                 hook_manager: HookManager or None = None, use_pooling: bool = True):
        super(MuZeroNet, self).__init__()
        self.input_channels = input_channels
        self.rep_input_channels = rep_input_channels
        self.dropout = dropout
        self.use_original = use_original
        self.use_pooling = use_pooling
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.num_channels = num_channels
        self.latent_size = latent_size
        self.optimizer = None
        self.num_out_channels = num_out_channels
        self.linear_input_size = linear_input_size
        self.support_size = support_size
        self.num_blocks = num_blocks
        self.hook_manager = hook_manager if hook_manager is not None else HookManager()
        # self.action_embedding = th.nn.Embedding(action_size, 256)
        self.representation_network = RepresentationNet(rep_input_channels, use_pooling=use_pooling)
        if use_original:
            self.dynamics_network = OriginalAlphaZerNetwork(in_channels=257, num_channels=num_out_channels,
                                                            dropout=dropout,
                                                            action_size=action_size,
                                                            linear_input_size=linear_input_size,
                                                            support_size=support_size, latent_size=latent_size,
                                                            num_blocks=num_blocks, muzero=True, is_dynamics=True)
            self.prediction_network = OriginalAlphaZerNetwork(in_channels=256, num_channels=num_out_channels,
                                                              dropout=dropout,
                                                              action_size=action_size,
                                                              linear_input_size=linear_input_size,
                                                              support_size=support_size, latent_size=latent_size,
                                                              num_blocks=num_blocks, muzero=True, is_dynamics=False)
        else:
            self.dynamics_network = DynamicsNet(in_channels=257, num_channels=num_channels, dropout=dropout,
                                                latent_size=latent_size, out_channels=num_out_channels)
            # prediction outputs 6x6 latent state
            self.prediction_network = PredictionNet(in_channels=256, num_channels=num_channels, dropout=dropout,
                                                    action_size=action_size, linear_input_size=linear_input_size)

    @classmethod
    def make_from_config(cls, config: MuZeroConfig, hook_manager: HookManager or None = None):
        return cls(config.num_net_in_channels, config.net_dropout, config.net_action_size, config.num_net_channels,
                   config.net_latent_size, config.num_net_out_channels, config.az_net_linear_input_size,
                   config.rep_input_channels, config.use_original, config.support_size, config.num_blocks,
                   hook_manager=hook_manager, use_pooling=config.use_pooling)

    def dynamics_forward(self, x: th.Tensor, predict: bool = False):
        if predict:
            state, r = self.dynamics_network.forward(x, muzero=True)
            state = state.view(self.num_out_channels, self.latent_size[0], self.latent_size[1])
            r = r.detach().cpu().numpy()
            return state, r
        state, reward = self.dynamics_network(x, muzero=True)
        return state, reward

    def prediction_forward(self, x: th.Tensor, predict: bool = False):
        if predict:
            pi, v = self.prediction_network.predict(x, muzero=True)
            return pi, v
        pi, v = self.prediction_network(x, muzero=True)
        return pi, v

    def representation_forward(self, x: th.Tensor):
        x = self.representation_network(x)
        return x

    def make_fresh_instance(self):
        return MuZeroNet(self.input_channels, self.dropout, self.action_size, self.num_channels, self.latent_size,
                         self.num_out_channels, self.linear_input_size, self.rep_input_channels,
                         hook_manager=self.hook_manager, use_original=self.use_original, support_size=self.support_size,
                         num_blocks=self.num_blocks, use_pooling=self.use_pooling)

    def train_net(self, memory_buffer: GeneralMemoryBuffer, muzero_config: MuZeroConfig) -> tuple[float, list[float]]:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        if self.optimizer is None:
            self.optimizer = th.optim.Adam(self.parameters(), lr=muzero_config.lr,
                                           weight_decay=muzero_config.l2)
        losses = []
        K = muzero_config.K
        iteration = 0
        memory_buffer.reset_priorities()
        loader = lambda: memory_buffer.batch_with_priorities(muzero_config.enable_per,
                                                             muzero_config.batch_size, K,
                                                             alpha=muzero_config.alpha)
        for epoch in range(muzero_config.epochs):
            experience_batch, priorities = loader()
            if len(experience_batch) <= 1:
                continue
            loss, loss_v, loss_pi, loss_r = self.calculate_losses(experience_batch, priorities, device, muzero_config)
            wandb.log({"combined_loss": loss.item(), "loss_v": loss_v.item(), "loss_pi": loss_pi.item(),
                       "loss_r": loss_r.item()})
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.hook_manager.process_hook_executes(self, self.train_net.__name__, __file__, HookAt.MIDDLE, args=(
                experience_batch, loss.item(), loss_v, loss_pi, loss_r,
                iteration))
            iteration += 1
        self.hook_manager.process_hook_executes(self, self.train_net.__name__, __file__, HookAt.TAIL,
                                                args=(memory_buffer, losses))
        return sum(losses) / len(losses), losses

    def eval_net(self, memory_buffer: GeneralMemoryBuffer, muzero_config: MuZeroConfig) -> None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        if self.optimizer is None:
            self.optimizer = th.optim.Adam(self.parameters(), lr=muzero_config.lr,
                                           weight_decay=muzero_config.l2)

        K = muzero_config.K
        memory_buffer.reset_priorities()
        loader = lambda: memory_buffer.batch_with_priorities(muzero_config.enable_per,
                                                             muzero_config.batch_size, K,
                                                             alpha=muzero_config.alpha,
                                                             is_eval=True)
        for epoch in range(muzero_config.eval_epochs):
            experience_batch, priorities = loader()
            if len(experience_batch) <= 1:
                continue
            loss, loss_v, loss_pi, loss_r = self.calculate_losses(experience_batch, priorities, device, muzero_config)
            wandb.log({"eval_combined_loss": loss.item(), "eval_loss_v": loss_v.item(), "eval_loss_pi": loss_pi.item(),
                       "eval_loss_r": loss_r.item()})

    def calculate_losses(self, experience_batch, priorities, device, muzero_config):
        pis, vs, rews_moves_players, states = zip(*experience_batch)
        rews = [x[0] for x in rews_moves_players]
        moves = [x[1] for x in rews_moves_players]
        states = [np.array(x) for x in states]
        states = th.tensor(np.array(states), dtype=th.float32, device=device).permute(0, 3, 1, 2)
        pis = [list(x.values()) for x in pis]
        pis = th.tensor(np.array(pis), dtype=th.float32, device=device)
        vs = th.tensor(np.array(vs), dtype=th.float32, device=device).unsqueeze(0)
        rews = th.tensor(np.array(rews), dtype=th.float32, device=device).unsqueeze(0)
        latent = self.representation_forward(states)
        pred_pis, pred_vs = self.prediction_forward(latent)
        # masks = mask_invalid_actions_batch(self.game_manager.get_invalid_actions, pis, players)
        latent = match_action_with_obs_batch(latent, moves)
        _, pred_rews = self.dynamics_forward(latent)
        priorities = priorities.to(device)
        balance_term = muzero_config.balance_term
        if muzero_config.enable_per:
            w = (1 / (len(priorities) * priorities)) ** muzero_config.beta
            w /= w.sum()
            w = w.reshape(pred_vs.shape)
        else:
            w = 1
        loss_v = mse_loss(pred_vs, vs) * balance_term * w
        loss_pi = self.muzero_pi_loss(pred_pis, pis) * balance_term * w
        loss_r = mse_loss(pred_rews, rews) * balance_term * w
        loss = loss_v.sum() + loss_pi.sum() + loss_r.sum()
        return loss, loss_v.sum(), loss_pi.sum(), loss_r.sum()

    def muzero_pi_loss(self, y_hat, y, masks: th.Tensor or None = None):
        if masks is not None:
            masks = masks.reshape(y_hat.shape).to(self.device)
            y_hat = masks * y_hat
        return -th.sum(y * y_hat) / y.size()[0]

    def continuous_weight_update(self, shared_storage: SharedStorage, muzero_config: MuZeroConfig):
        wandb.init(project="MZ",name="Continuous Weight Update")
        self.train()
        losses = []
        loss_avgs = []
        while len(shared_storage.get_mem_buffer().get_buffer()) < muzero_config.batch_size * 3: # await reasonable buffer size
            time.sleep(5)
        for iter_ in range(muzero_config.num_worker_iters):
            shared_storage.set_experimental_network_params(self.state_dict())
            avg, iter_losses = self.train_net(shared_storage.get_mem_buffer(), muzero_config)
            loss_avgs.append(avg)
            losses.extend(iter_losses)
            wandb.log({"loss_avg": avg})

    def to_pickle(self, path: str):
        th.save(self, path)

    def run_on_training_end(self):
        self.hook_manager.process_hook_executes(self, self.run_on_training_end.__name__, __file__, HookAt.ALL)


class RepresentationNet(th.nn.Module):
    def __init__(self, rep_input_channels: int, use_pooling: bool = True):
        super(RepresentationNet, self).__init__()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.conv1 = th.nn.Conv2d(in_channels=rep_input_channels, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.residuals1 = th.nn.ModuleList([ResidualBlock(128) for _ in range(2)])
        self.conv2 = th.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.residuals2 = th.nn.ModuleList([ResidualBlock(256) for _ in range(3)])

        self.residuals3 = th.nn.ModuleList([ResidualBlock(256) for _ in range(3)])
        if use_pooling:
            self.pool1 = th.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.pool2 = th.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool1 = th.nn.Identity()
            self.pool2 = th.nn.Identity()
        self.relu = th.nn.ReLU()

    def forward(self, x: th.Tensor):
        # x.unsqueeze(0)
        x = x.to(self.device)
        x = self.relu(self.conv1(x))
        for residual in self.residuals1:
            x = residual(x)
        x = self.relu(self.conv2(x))
        for residual in self.residuals2:
            x = residual(x)
        x = self.pool1(x)
        for residual in self.residuals3:
            x = residual(x)
        x = self.pool2(x)
        return x

    def trace(self):
        data = th.rand((1, 128, 8, 8))
        traced_script_module = th.jit.trace(self, data)
        return traced_script_module


class DynamicsNet(nn.Module):
    def __init__(self, in_channels, num_channels, dropout, latent_size, out_channels):
        super(DynamicsNet, self).__init__()
        self.out_channels = out_channels
        self.latent_size = latent_size

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
        # 4608 (5x5) or 512 (3x3) or 32768 (10x10) or 18432 (8x8)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.state_head = nn.Linear(512, latent_size[0] * latent_size[1] * out_channels)  # state head
        self.reward_head = nn.Linear(512, 1)  # reward head

    def forward(self, x, muzero=False):
        # x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)

        state = self.state_head(x)
        r = self.reward_head(x)

        return state, r

    def trace(self) -> th.jit.ScriptFunction:
        data = th.rand((1, 257, 6, 6)).to("cuda:0")
        traced_script_module = th.jit.trace(self, data)
        return traced_script_module

    @th.no_grad()
    def predict(self, x):
        state, r = self.forward(x)
        state = state.view(self.out_channels, self.latent_size[0], self.latent_size[1])
        return state, r.detach().cpu().numpy()


class ResidualBlock(th.nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.convolution1 = th.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bnorm1 = th.nn.BatchNorm2d(channels)
        self.convolution2 = th.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bnorm2 = th.nn.BatchNorm2d(channels)
        self.relu = th.nn.ReLU()

    def forward(self, x):
        # x = x.unsqueeze(0)
        x_res = x
        convolved = self.convolution1(x)
        x = self.relu(self.bnorm1(convolved))
        x = self.bnorm2(self.convolution2(x))
        x += x_res
        x = self.relu(x)
        return x
