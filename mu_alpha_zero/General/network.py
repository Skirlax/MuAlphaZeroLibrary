from abc import ABC, abstractmethod
from mu_alpha_zero.config import Config
import torch as th


class GeneralNetwork(ABC):
    @abstractmethod
    def make_fresh_instance(self):
        """
        Returns a fresh instance of the network
        """
        pass

    @classmethod
    @abstractmethod
    def make_from_config(cls, config: Config):
        """
        Builds the network from the given arguments dict.
        """
        pass

    @abstractmethod
    def train_net(self, memory_buffer, muzero_alphazero_config: Config) -> tuple[float, list[float]]:
        """
        Trains the network for given number of epochs
        """
        pass


class GeneralMuZeroNetwork(GeneralNetwork):

    @abstractmethod
    def dynamics_forward(self, x: th.Tensor, predict: bool = False) -> (th.Tensor, th.Tensor):
        """
        Forward pass for the dynamics network. Should operate only on the hidden state, not the actual game state.
        Call representation_forward first.
        :param x: The input hidden state.
        :param predict: Whether to predict or do pure forward pass.
        :return: A tuple of the next hidden state and the immediate reward.
        """
        pass

    @abstractmethod
    def prediction_forward(self, x: th.Tensor, predict: bool = False) -> (th.Tensor, th.Tensor):
        """
        Forward pass for the prediction network. As dynamics forward, this should operate only on the hidden state,
        not the actual game state.
        Call representation_forward first.
        :param x: The input hidden state.
        :param predict: Whether to predict or do pure forward pass.
        :return: A tuple of the action probability distribution and the value of the current state.
        """
        pass

    @abstractmethod
    def representation_forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass for the representation network. This should operate on the actual game state.
        :param x: The input game state.
        :return: The hidden state.
        """
        pass

    @abstractmethod
    def muzero_pi_loss(self, y_hat: th.Tensor, y: th.Tensor) -> th.Tensor:
        """
        Calculate the loss for the action probability distribution.
        :param y_hat: The predicted action probability distribution.
        :param y: The MCTS improved action probability distribution.
        :return: The loss.
        """
        pass


class GeneralAlphZeroNetwork(GeneralNetwork):
    @abstractmethod
    def predict(self, x: th.Tensor, muzero: bool = True) -> (th.Tensor, th.Tensor):
        """
        Predict the action probability distribution and the value of the current state.
        :param x: The input game state.
        :param muzero: Whether to predict for MuZero or AlphaZero.
        :return: A tuple of the action probability distribution and the value of the current state.
        """
        pass
