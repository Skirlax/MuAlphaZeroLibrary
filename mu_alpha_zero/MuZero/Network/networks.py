import time

import numpy as np
import torch as th
import torch.nn.functional as F
import wandb
from torch import nn
from torch.nn.functional import mse_loss

from mu_alpha_zero.AlphaZero.Network.nnet import AlphaZeroNet as PredictionNet, OriginalAlphaZeroNetwork
from mu_alpha_zero.AlphaZero.checkpointer import CheckPointer
from mu_alpha_zero.AlphaZero.logger import Logger
from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.mz_game import MuZeroGame
from mu_alpha_zero.General.network import GeneralMuZeroNetwork
from mu_alpha_zero.Hooks.hook_manager import HookManager
from mu_alpha_zero.Hooks.hook_point import HookAt
from mu_alpha_zero.MuZero.utils import match_action_with_obs_batch, scalar_to_support, support_to_scalar
from mu_alpha_zero.config import MuZeroConfig
from mu_alpha_zero.shared_storage_manager import SharedStorage


class MuZeroNet(th.nn.Module, GeneralMuZeroNetwork):
    def __init__(self, input_channels: int, dropout: float, action_size: int, num_channels: int, latent_size: list[int],
                 num_out_channels: int, linear_input_size: int or list[int], rep_input_channels: int,
                 use_original: bool, support_size: int, num_blocks: int,
                 state_linear_layers: int, pi_linear_layers: int, v_linear_layers: int, linear_head_hidden_size: int,
                 is_atari: bool,
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
        self.scheduler = None
        self.num_out_channels = num_out_channels
        self.linear_input_size = linear_input_size
        self.support_size = support_size
        self.is_atari = is_atari
        self.num_blocks = num_blocks
        self.state_linear_layers = state_linear_layers
        self.pi_linear_layers = pi_linear_layers
        self.v_linear_layers = v_linear_layers
        self.linear_head_hidden_size = linear_head_hidden_size
        self.hook_manager = hook_manager if hook_manager is not None else HookManager()
        # self.action_embedding = th.nn.Embedding(action_size, 256)
        if not is_atari:
            self.representation_network = OriginalAlphaZeroNetwork(in_channels=rep_input_channels,
                                                                   num_channels=num_out_channels,
                                                                   dropout=dropout,
                                                                   action_size=action_size,
                                                                   linear_input_size=linear_input_size,
                                                                   state_linear_layers=state_linear_layers,
                                                                   pi_linear_layers=pi_linear_layers,
                                                                   v_linear_layers=v_linear_layers,
                                                                   linear_head_hidden_size=linear_head_hidden_size,
                                                                   support_size=support_size, latent_size=latent_size,
                                                                   num_blocks=num_blocks, muzero=True, is_dynamics=False,
                                                                   is_representation=True)
        else:
            self.representation_network = RepresentationNet(rep_input_channels, use_pooling=use_pooling)
        if use_original:
            self.dynamics_network = OriginalAlphaZeroNetwork(in_channels=257, num_channels=num_out_channels,
                                                             dropout=dropout,
                                                             action_size=action_size,
                                                             linear_input_size=linear_input_size,
                                                             state_linear_layers=state_linear_layers,
                                                             pi_linear_layers=pi_linear_layers,
                                                             v_linear_layers=v_linear_layers,
                                                             linear_head_hidden_size=linear_head_hidden_size,
                                                             support_size=support_size, latent_size=latent_size,
                                                             num_blocks=num_blocks, muzero=True, is_dynamics=True)
            self.prediction_network = OriginalAlphaZeroNetwork(in_channels=256, num_channels=num_out_channels,
                                                               dropout=dropout,
                                                               action_size=action_size,
                                                               state_linear_layers=state_linear_layers,
                                                               pi_linear_layers=pi_linear_layers,
                                                               v_linear_layers=v_linear_layers,
                                                               linear_head_hidden_size=linear_head_hidden_size,
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
                   config.state_linear_layers, config.pi_linear_layers, config.v_linear_layers,
                   config.linear_head_hidden_size, config.is_atari,
                   hook_manager=hook_manager, use_pooling=config.use_pooling)

    def dynamics_forward(self, x: th.Tensor, predict: bool = False, return_support: bool = False,
                         convert_to_state: bool = True):
        if predict:
            state, r = self.dynamics_network.forward(x, muzero=True)
            reward = r.detach().cpu().numpy()
        else:
            state, reward = self.dynamics_network(x, muzero=True, return_support=return_support)
        if convert_to_state:
            try:
                state = state.view(self.num_out_channels, self.latent_size[0], self.latent_size[1])
            except RuntimeError:
                # The state is batched
                state = state.view(-1, self.num_out_channels, self.latent_size[0], self.latent_size[1])
        return state, reward

    def prediction_forward(self, x: th.Tensor, predict: bool = False, return_support: bool = False):
        if predict:
            pi, v = self.prediction_network.predict(x, muzero=True)
            return pi, v
        pi, v = self.prediction_network(x, muzero=True, return_support=return_support)
        return pi, v

    def representation_forward(self, x: th.Tensor):
        state = self.representation_network(x, muzero=True)
        try:
            state = state.view(self.num_out_channels, self.latent_size[0], self.latent_size[1])
        except RuntimeError:
            # The state is batched
            state = state.view(-1, self.num_out_channels, self.latent_size[0], self.latent_size[1])
        return state

    def forward_recurrent(self, hidden_state_with_action: th.Tensor, all_predict: bool, return_support: bool = False):
        next_state, reward = self.dynamics_forward(hidden_state_with_action, predict=all_predict,
                                                   return_support=return_support)
        pi, v = self.prediction_forward(next_state, predict=all_predict, return_support=return_support)
        return next_state, reward, pi, v

    def make_fresh_instance(self):
        return MuZeroNet(self.input_channels, self.dropout, self.action_size, self.num_channels, self.latent_size,
                         self.num_out_channels, self.linear_input_size, self.rep_input_channels,
                         hook_manager=self.hook_manager, use_original=self.use_original, support_size=self.support_size,
                         num_blocks=self.num_blocks, use_pooling=self.use_pooling,
                         state_linear_layers=self.state_linear_layers,
                         pi_linear_layers=self.pi_linear_layers, v_linear_layers=self.v_linear_layers,
                         linear_head_hidden_size=self.linear_head_hidden_size, is_atari=self.is_atari)

    def train_net(self, memory_buffer: GeneralMemoryBuffer, muzero_config: MuZeroConfig) -> tuple[float, list[float]]:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        if self.optimizer is None:
            self.optimizer = th.optim.Adam(self.parameters(), lr=muzero_config.lr,
                                           weight_decay=muzero_config.l2)
        if self.scheduler is None and muzero_config.lr_scheduler is not None:
            self.scheduler = muzero_config.lr_scheduler(self.optimizer, **muzero_config.lr_scheduler_kwargs)
        losses = []
        iteration = 0
        loader = lambda: memory_buffer.batch_with_priorities(muzero_config.enable_per,
                                                             muzero_config.batch_size, muzero_config)
        for epoch in range(muzero_config.epochs):
            sampled_game_data, priorities, weights = loader()
            if len(sampled_game_data) <= 1:
                continue
            loss, loss_v, loss_pi, loss_r = self.calculate_losses(sampled_game_data, weights, device, muzero_config)
            wandb.log({"combined_loss": loss.item(), "loss_v": loss_v.item(), "loss_pi": loss_pi.item(),
                       "loss_r": loss_r.item()})
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
                try:
                    wandb.log({"lr": self.scheduler.get_last_lr()[0]})
                except:
                    pass
            self.hook_manager.process_hook_executes(self, self.train_net.__name__, __file__, HookAt.MIDDLE, args=(
                sampled_game_data, loss.item(), loss_v, loss_pi, loss_r,
                iteration))
            iteration += 1
        self.hook_manager.process_hook_executes(self, self.train_net.__name__, __file__, HookAt.TAIL,
                                                args=(memory_buffer, losses))
        return sum(losses) / len(losses), losses

    def eval_net(self, memory_buffer: GeneralMemoryBuffer, muzero_config: MuZeroConfig) -> None:
        if memory_buffer.eval_length() == 0:
            return
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        if self.optimizer is None:
            self.optimizer = th.optim.Adam(self.parameters(), lr=muzero_config.lr,
                                           weight_decay=muzero_config.l2)

        loader = lambda: memory_buffer.batch_with_priorities(muzero_config.enable_per,
                                                             muzero_config.batch_size,
                                                             muzero_config,
                                                             is_eval=True)
        for epoch in range(muzero_config.eval_epochs):
            experience_batch, priorities, weights = loader()
            if len(experience_batch) <= 1:
                continue
            loss, loss_v, loss_pi, loss_r = self.calculate_losses(experience_batch, weights, device, muzero_config)
            wandb.log({"eval_combined_loss": loss.item(), "eval_loss_v": loss_v.item(), "eval_loss_pi": loss_pi.item(),
                       "eval_loss_r": loss_r.item()})

    def calculate_losses(self, experience_batch, weights, device, muzero_config):
        init_states, rewards, scalar_values, moves, pis = self.get_batch_for_unroll_index(0, experience_batch, device)
        # rewards = scalar_to_support(rewards, muzero_config.support_size)
        values = scalar_to_support(scalar_values, muzero_config.support_size)
        hidden_state = self.representation_forward(init_states)
        pred_pis, pred_vs = self.prediction_forward(hidden_state, return_support=True)
        pi_loss, v_loss, r_loss = 0, 0, 0
        pi_loss += self.muzero_loss(pred_pis, pis)
        v_loss += self.muzero_loss(pred_vs, values)
        new_priorities = [[] for x in range(pred_pis.size(0))]
        if muzero_config.enable_per:
            self.populate_priorities((th.abs(support_to_scalar(pred_vs,
                                                               muzero_config.support_size) - scalar_values) ** muzero_config.alpha).reshape(
                -1).tolist(), new_priorities)
        for i in range(1, muzero_config.K + 1):
            _, rewards, scalar_values, moves, pis = self.get_batch_for_unroll_index(i, experience_batch,
                                                                                    device)
            rewards = scalar_to_support(rewards, muzero_config.support_size)
            values = scalar_to_support(scalar_values, muzero_config.support_size)
            hidden_state, pred_rs, pred_pis, pred_vs = self.forward_recurrent(
                match_action_with_obs_batch(hidden_state, moves), False, return_support=True)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            current_pi_loss = self.muzero_loss(pred_pis, pis)
            current_v_loss = self.muzero_loss(pred_vs, values)
            current_r_loss = self.muzero_loss(pred_rs, rewards)
            current_r_loss.register_hook(lambda grad: grad * (1 / muzero_config.K))
            current_v_loss.register_hook(lambda grad: grad * (1 / muzero_config.K))
            current_pi_loss.register_hook(lambda grad: grad * (1 / muzero_config.K))
            pi_loss += current_pi_loss
            v_loss += current_v_loss
            r_loss += current_r_loss
            if muzero_config.enable_per:
                self.populate_priorities((th.abs(support_to_scalar(pred_vs,
                                                                   muzero_config.support_size) - scalar_values) ** muzero_config.alpha).reshape(
                    -1).tolist(), new_priorities)
        # TODO: Multiply v by 0.25 when reanalyze implemented.
        # v_loss *= 0.25
        loss = pi_loss + v_loss + r_loss
        if muzero_config.enable_per:
            loss *= th.tensor(weights, dtype=loss.dtype, device=loss.device)
        loss = loss.sum()
        if muzero_config.enable_per:
            self.update_priorities(new_priorities, experience_batch)
        return loss, v_loss.sum(), pi_loss.sum(), r_loss.sum()

    def get_batch_for_unroll_index(self, index: int, experience_batch, device) -> tuple[
        th.Tensor, th.Tensor, th.Tensor, list, th.Tensor]:
        tensor_from_x = lambda x: th.tensor(x, dtype=th.float32, device=device)
        init_states = None
        if index == 0:
            init_states = [np.array(x.datapoints[index].frame) for x in experience_batch]
            init_states = tensor_from_x(np.array(init_states)).permute(0, 3, 1, 2)
        rewards = np.array([x.datapoints[index].reward for x in experience_batch])
        rewards = tensor_from_x(rewards)
        values = np.array([x.datapoints[index].v for x in experience_batch])
        values = tensor_from_x(values)
        moves = [x.datapoints[index].move for x in experience_batch]
        # moves = np.array([x.datapoints[index].move for x in experience_batch])
        # moves = tensor_from_x(moves)
        pis = np.array([x.datapoints[index].pi for x in experience_batch])
        pis = tensor_from_x(pis)
        return init_states, rewards.unsqueeze(1), values.unsqueeze(1), moves, pis

    def populate_priorities(self, new_priorities: list, priorities_list: list):
        for idx, priority in enumerate(new_priorities):
            priorities_list[idx].append(priority)

    def update_priorities(self, new_priorities: list, experience_batch: list):
        for idx, game in enumerate(experience_batch):
            for i in range(len(game.datapoints)):
                game.datapoints[i].priority = new_priorities[idx][i]

    def muzero_loss(self, y_hat, y):
        return -th.sum(y * y_hat, dim=1).unsqueeze(1) / y.size()[0]

    def continuous_weight_update(self, shared_storage: SharedStorage, muzero_config: MuZeroConfig,
                                 checkpointer: CheckPointer, logger: Logger):
        wandb.init(project=muzero_config.wandbd_project_name, name="Continuous Weight Update")
        self.train()
        # losses = []
        # loss_avgs = []
        # num_epochs = muzero_config.epochs
        muzero_config.epochs = 100
        muzero_config.eval_epochs = 100
        while len(
                shared_storage.get_buffer()) < muzero_config.batch_size // 4:  # await reasonable buffer size
            time.sleep(5)
        logger.log("Finished waiting for target buffer size,starting training.")
        for iter_ in range(muzero_config.num_worker_iters // 100):
            # if not shared_storage.get_was_pitted():
            #     time.sleep(5)
            #     continue
            # self.load_state_dict(shared_storage.get_stable_network_params())
            # for epoch in range(num_epochs):
            avg, iter_losses = self.train_net(shared_storage, muzero_config)
            shared_storage.set_experimental_network_params(self.state_dict())
            # shared_storage.set_optimizer(self.optimizer.state_dict())
            # loss_avgs.append(avg)
            # losses.extend(iter_losses)
            # shared_storage.set_was_pitted(False)
            if iter_ % 100 == 0:
                self.eval_net(shared_storage, muzero_config)
            if iter_ % 5 == 0 and iter_ != 0:
                logger.log(f"Saving checkpoint at iteration {iter_}.")
                checkpointer.save_checkpoint(self, self, self.optimizer, muzero_config.lr, iter_, muzero_config)
            # wandb.log({"loss_avg": avg})

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

    def forward(self, x: th.Tensor,muzero:bool = False):
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
