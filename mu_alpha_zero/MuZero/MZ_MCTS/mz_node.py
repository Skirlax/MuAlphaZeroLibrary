import math
from mu_alpha_zero.AlphaZero.MCTS.az_node import AlphaZeroNode


class MzAlphaZeroNode(AlphaZeroNode):
    def __init__(self, select_probability=0, parent=None, times_visited_init=0):
        super().__init__(0, select_probability, parent, times_visited_init)
        self.reward = 0

    def get_best_child(self, c=1.5, c2=19652):
        best_utc = -float("inf")
        best_child = None
        best_action = None
        for action, child in self.children.items():
            child_utc = child.calculate_utc_score(c=c, c2=c2)
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
            node = MzAlphaZeroNode(select_probability=probability, parent=self)
            self.children[action] = node

    def get_immediate_reward(self, dynamics_forward: callable, action: int):
        return dynamics_forward(self.state, action)

    def calculate_utc_score(self, c=1.5, c2=19652):
        parent = self.parent()
        if self.q is None:
            # Inspiration taken from https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
            utc = c * self.select_probability * math.sqrt(parent.times_visited + 1e-8)
        else:
            utc1 = self.q + self.select_probability * (
                    (math.sqrt(parent.times_visited)) / (1 + self.times_visited))
            utc2 = c + math.log((parent.times_visited + c2 + 1) / c2)
            utc = utc1 * utc2

        return utc

    def scale_q(self, min_q, max_q):
        if min_q == max_q:
            return
        self.q = (self.q - min_q) / (max_q - min_q)
