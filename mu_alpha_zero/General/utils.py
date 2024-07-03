import os
import subprocess
import sys


def find_project_root() -> str:
    dir_ = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dir_)
    while "root.root" not in os.listdir(dir_):
        if os.getcwd() == "/" or os.getcwd() == "C:\\":
            raise FileNotFoundError("Could not find project root.")
        os.chdir("..")
        dir_ = os.getcwd()
    return os.getcwd()


def not_zero(x):
    return x if x != 0 else 1


def get_python_home() -> str:
    return os.path.dirname(os.path.dirname(sys.executable))


def get_site_packages_path() -> str:
    output = subprocess.getoutput("pip show torch")
    return output.split("Location:")[1].split("Requires")[0].strip()


class OnlyUnique:
    def __init__(self, iterable: list = []):
        self.unique = []
        self.extend(iterable)

    def add(self, item):
        if item not in self.unique:
            self.unique.append(item)

    def extend(self, iterable):
        for item in iterable:
            self.add(item)

    def __len__(self):
        return len(self.unique)

    def get(self):
        return self.unique


def clear_disk_data():
    dir_ = f"{find_project_root()}/Pickles/Data"
    for file in os.listdir(dir_):
        os.remove(f"{dir_}/{file}")


def net_not_none(net):
    assert net is not None, (
        "Network is None, can't train/predict/pit. Make sure you initialize the network with either"
        "load_checkpoint or create_new method.")


def get_players() -> list[str]:
    path_prefix = find_project_root().replace("\\", "/").split("/")[-1]
    return [x for x in list(sys.modules[f"{path_prefix}.AlphaZero.Arena.players"].__dict__.keys()) if
            x.endswith("Player")]


def adjust_probabilities(action_probabilities: dict, tau=1.0) -> dict:
    """
    Selects a move from the action probabilities using either greedy or stochastic policy.
    The stochastic policy uses the tau parameter to adjust the probabilities. This is based on the
    temperature parameter in DeepMind's AlphaZero paper.

    :param action_probabilities: A dictionary containing the action probabilities in the form of {action_index: probability}.
    :param tau: The temperature parameter. 0 for greedy, >0 for stochastic.
    :return: The selected move as an integer (index).
    """
    if tau == 0:  # select greedy
        vals = [x for x in action_probabilities.values()]
        max_idx = vals.index(max(vals))
        probs = [0 for _ in range(len(vals))]
        probs[max_idx] = 1
        return dict(zip(action_probabilities.keys(), probs))
    # select stochastic
    moves, probabilities = zip(*action_probabilities.items())
    adjusted_probs = [prob ** (1 / tau) for prob in probabilities]
    adjusted_probs_sum = sum(adjusted_probs)
    normalized_probs = [prob / adjusted_probs_sum for prob in adjusted_probs]
    return dict(zip(moves, normalized_probs))
