from abc import abstractmethod, ABC
from typing import Type

from mu_alpha_zero.AlphaZero.Arena.players import Player


class GeneralArena(ABC):
    @abstractmethod
    def pit(self, player1: Type[Player], player2: Type[Player],
            num_games_to_play: int, num_mc_simulations: int, one_player: bool = False,
            start_player: int = 1):
        pass
