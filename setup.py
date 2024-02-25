from setuptools import setup
from setuptools.command.install import install
import subprocess
from mu_alpha_zero.General.utils import find_project_root


def get_requirements():
    with open(f"{find_project_root()}requirements.txt", "r") as file:
        return file.read().strip().split("\n")





setup(
    name="mu_alpha_zero_lib",
    version="1.0",
    description="Library for running and training MuZero and AlphaZero models.",
    author="Skyr",
    install_requires=get_requirements(),
    package_data={
        "mu_alpha_zero": ["*.txt", "*.root"]
    }

)
