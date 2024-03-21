from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name="mu_alpha_zero_library", version="1.0.5",
      description="Library for running and training MuZero and AlphaZero models.", author="Skyr",
      packages=find_packages(),
      install_requires=open("requirements.txt").read().strip().split("\n"),
      long_description=open('README.md', encoding="utf-8").read(), long_description_content_type='text/markdown',
      package_data={"mu_alpha_zero": ["*.txt", "*.root"]}, url="https://github.com/Skirlax/MuAlphaZeroLibrary",

      )
