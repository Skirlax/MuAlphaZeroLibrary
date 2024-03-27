import math

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import mse_loss

from mu_alpha_zero.AlphaZero.Network.nnet import AlphaZeroNet as PredictionNet
from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.network import GeneralMuZeroNetwork
from mu_alpha_zero.MuZero.utils import match_action_with_obs_batch, scale_reward_value
from mu_alpha_zero.config import MuZeroConfig


class MuZeroNet(th.nn.Module, GeneralMuZeroNetwork):
    def __init__(self, input_channels: int, dropout: float, action_size: int, num_channels: int, latent_size: int,
                 num_out_channels: int, linear_input_size: int):
        super(MuZeroNet, self).__init__()
        self.input_channels = input_channels
        self.dropout = dropout
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.num_channels = num_channels
        self.latent_size = latent_size
        self.num_out_channels = num_out_channels
        self.linear_input_size = linear_input_size
        # self.action_embedding = th.nn.Embedding(action_size, 256)
        self.representation_network = RepresentationNet()
        self.dynamics_network = DynamicsNet(in_channels=257, num_channels=num_channels, dropout=dropout,
                                            latent_size=latent_size, out_channels=num_out_channels)
        # prediction outputs 6x6 latent state
        self.prediction_network = PredictionNet(in_channels=256, num_channels=num_channels, dropout=dropout,
                                                action_size=action_size, linear_input_size=linear_input_size)

    @classmethod
    def make_from_config(cls, config: MuZeroConfig):
        return cls(config.num_net_in_channels, config.net_dropout, config.net_action_size,
                   config.num_net_channels, config.net_latent_size, config.num_net_out_channels,
                   config.az_net_linear_input_size)

    def dynamics_forward(self, x: th.Tensor, predict: bool = False):
        if predict:
            return self.dynamics_network.predict(x)
        state, reward = self.dynamics_network(x)
        reward = scale_reward_value(reward)
        return state, reward

    def prediction_forward(self, x: th.Tensor, predict: bool = False):
        if predict:
            pi, v = self.prediction_network.predict(x, muzero=True)
            v = scale_reward_value(v[0][0])
            return pi, v
        pi, v = self.prediction_network(x, muzero=True)
        v = scale_reward_value(v)
        return pi, v

    def representation_forward(self, x: th.Tensor):
        x = self.representation_network(x)
        return x

    def make_fresh_instance(self):
        return MuZeroNet(self.input_channels, self.dropout, self.action_size, self.num_channels, self.latent_size,
                         self.num_out_channels, self.linear_input_size)

    def train_net(self, memory_buffer: GeneralMemoryBuffer, muzero_config: MuZeroConfig) -> tuple[float, list[float]]:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        losses = []
        K = muzero_config.K
        optimizer = th.optim.Adam(self.parameters(), lr=muzero_config.lr)
        for experience_batch, priorities in memory_buffer.batch_with_priorities(muzero_config.epochs,
                                                                                muzero_config.batch_size, K,
                                                                                alpha=muzero_config.alpha):
            if len(experience_batch) <= 1:
                continue
            pis, vs, rews_moves, states = zip(*experience_batch)
            rews = [x[0] for x in rews_moves]
            moves = [x[1] for x in rews_moves]
            states = [np.array(x) for x in states]
            states = th.tensor(np.array(states), dtype=th.float32, device=device).permute(0, 3, 1, 2)
            pis = [list(x.values()) for x in pis]
            pis = th.tensor(np.array(pis), dtype=th.float32, device=device)
            vs = th.tensor(np.array(vs), dtype=th.float32, device=device).unsqueeze(1)
            rews = th.tensor(np.array(rews), dtype=th.float32, device=device).unsqueeze(1)
            latent = self.representation_forward(states)
            pred_pis, pred_vs = self.prediction_forward(latent)
            latent = match_action_with_obs_batch(latent, moves)
            _, pred_rews = self.dynamics_forward(latent)
            priorities = priorities.to(device)
            balance_term = 1
            w = (1 / (len(priorities) * priorities)) ** muzero_config.beta
            w /= w.sum()
            w = w.reshape(pred_vs.shape)
            loss_v = mse_loss(pred_vs, vs) * balance_term * w
            loss_pi = self.muzero_pi_loss(pred_pis, pis) * balance_term * w
            loss_r = mse_loss(pred_rews, rews) * balance_term * w
            loss = loss_v.sum() + loss_pi.sum() + loss_r.sum()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return sum(losses) / len(losses), losses

    def muzero_pi_loss(self, y_hat, y):
        return -th.sum(y * y_hat) / y.size()[0]

    def to_pickle(self, path: str):
        th.save(self, path)


class RepresentationNet(th.nn.Module):
    def __init__(self):
        super(RepresentationNet, self).__init__()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.conv1 = th.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.residuals1 = th.nn.ModuleList([ResidualBlock(128) for _ in range(2)])
        self.conv2 = th.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.residuals2 = th.nn.ModuleList([ResidualBlock(256) for _ in range(3)])
        self.pool1 = th.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.residuals3 = th.nn.ModuleList([ResidualBlock(256) for _ in range(3)])
        self.pool2 = th.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
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
        self.latent_size = int(math.sqrt(latent_size))

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
        self.state_head = nn.Linear(512, latent_size * out_channels)  # state head
        self.reward_head = nn.Linear(512, 1)  # reward head

    def forward(self, x):
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
        state = state.view(self.out_channels, self.latent_size, self.latent_size)
        r = scale_reward_value(r)
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
