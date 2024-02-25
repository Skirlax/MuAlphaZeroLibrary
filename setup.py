from setuptools import setup
from setuptools.command.install import install
import subprocess


def get_requirements():
    with open("requirements.txt", "r") as file:
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
