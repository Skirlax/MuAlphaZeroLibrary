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
    return [x for x in list(sys.modules[f"{path_prefix}.AlphaZero.Arena.players"].__dict__.keys()) if x.endswith("Player")]

