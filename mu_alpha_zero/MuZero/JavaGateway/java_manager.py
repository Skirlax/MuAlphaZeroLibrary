import numbers
import os
import shutil
import subprocess

import numpy as np
from py4j.java_gateway import JavaGateway
from mu_alpha_zero.General.utils import get_python_home, find_project_root, get_site_packages_path
from mu_alpha_zero.MuZero.JavaGateway.java_networks import JavaNetworks
from mu_alpha_zero.MuZero.Network.networks import MuZeroNet
import atexit


class JavaManager:
    def __init__(self, network: MuZeroNet, args: dict):
        # jpype.addClassPath(f"{find_project_root()}/MuZero/JavaGateway/JSeflplayLinux-1.0-SNAPSHOT.jar")
        # jpype.addClassPath(f"/home/skyr/IdeaProjects/JSelfPlay/target/JSeflplayLinux-1.0-SNAPSHOT.jar")
        os.environ["PYTHONHOME"] = get_python_home()
        # jpype.startJVM(jpype.getDefaultJVMPath(),f"-Djava.library.path={get_site_packages_path()}/jep/")
        self.java_nets = JavaNetworks()
        self.net = network
        self.env_id = args["env_id"]
        self.args = self.prepare_args(args)
        self.java_process = self.spawn_java_p()
        self.java_gateway = JavaGateway()
        atexit.register(self.kill_java_process)

    def prepare_args(self, args: dict):
        return {k: v for k, v in args.items() if
                isinstance(v, numbers.Number) and not isinstance(v, bool) and v != float("inf") and v != float("-inf")}

    def spawn_java_p(self):
        return subprocess.Popen(
            ["java", "-jar", "/home/skyr/IdeaProjects/JSelfPlay/target/JSeflplayLinux-1.0-SNAPSHOT.jar",
             f"-Djava.library.path={get_site_packages_path()}/jep/"])

    def run_parallel_java_self_play(self, n_jobs: int, n_games: int):
        results = []
        path = self.java_nets.save(self.net)
        os.makedirs(f"{find_project_root()}/Arrays/", exist_ok=True)
        res = self.java_gateway.entry_point.runParallelSelfPlay(n_jobs, n_games, path, self.dict_to_java_map(self.args),
                                                                self.env_id)
        arr_paths = set([x.getValue3() for x in res])
        arrs = []
        for path in arr_paths:
            arrs.append(np.load(path))
        arr = np.concatenate(arrs, axis=0)
        for i,quarted in enumerate(res):
            hmap = quarted.getValue0()
            pd = {int(entry.getKey()): float(entry.getValue()) for entry in hmap.entrySet()}
            v = float(quarted.getValue1())
            rmpred_r = (float(quarted.getValue2().getValue0()), float(quarted.getValue2().getValue1()),
                        float(quarted.getValue2().getValue2()))
            results.append((pd, v, rmpred_r, arr[i]))
        shutil.rmtree(f"{find_project_root()}/Arrays/")
        return results

    def dict_to_java_map(self, dict: dict):
        java_map = self.java_gateway.jvm.java.util.HashMap()
        for k, v in dict.items():
            java_map.put(k, str(v))
        return java_map

    def kill_java_process(self):
        self.java_process.kill()
        self.java_process.wait()
        print("Java process ended.")
        return
