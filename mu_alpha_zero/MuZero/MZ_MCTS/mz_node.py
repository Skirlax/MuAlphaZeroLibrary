import math
from mu_alpha_zero.AlphaZero.MCTS.az_node import AlphaZeroNode


class MzAlphaZeroNode(AlphaZeroNode):
    def __init__(self, select_probability=0, parent=None, times_visited_init=0, current_player=1):
        super().__init__(current_player, select_probability, parent, times_visited_init)
        self.reward = 0

    def get_best_child(self, min_q: float, max_q: float, c=1.5, c2=19652):
        best_utc = -float("inf")
        best_child = None
        best_action = None
        for action, child in self.children.items():
            child_utc = child.calculate_utc_score(min_q, max_q, c=c, c2=c2)
            if child_utc > best_utc:
                best_utc = child_utc
                best_child = child
                best_action = action
        return best_child, best_action

    def get_value_pred(self, prediction_forward: callable):
        return prediction_forward(self.state)

    def expand_node(self, state, action_probabilities, im_reward) -> None:

        self.state = state.clone()
        self.reward = im_reward
        for action, probability in enumerate(action_probabilities):
            node = MzAlphaZeroNode(select_probability=probability, parent=self,
                                   current_player=self.current_player * (-1))
            self.children[action] = node

    def get_immediate_reward(self, dynamics_forward: callable, action: int):
        return dynamics_forward(self.state, action)

    def calculate_utc_score(self, min_q: float, max_q: float, c=1.5, c2=19652):
        parent = self.parent()
        if self.q is None:
            # Inspiration taken from https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
            utc = c * self.select_probability * math.sqrt(parent.times_visited + 1e-8)
        else:
            q = self.scale_q(min_q, max_q)
            utc = q + self.select_probability * (
                    (math.sqrt(parent.times_visited)) / (1 + self.times_visited)) * (
                          c + math.log((parent.times_visited + c2 + 1) / c2))

        return utc

    def scale_q(self, min_q, max_q) -> float:
        if min_q == max_q or (min_q == float("inf") or max_q == float("-inf")):
            return self.q
        return (self.q - min_q) / (max_q - min_q)
