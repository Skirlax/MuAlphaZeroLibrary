from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [Extension("mz_search_tree",
                         ["mu_alpha_zero/MuZero/MZ_MCTS/mz_search_tree.pyx",
                          "mu_alpha_zero/MuZero/MZ_MCTS/mz_node.pyx",
                          "mu_alpha_zero/AlphaZero/MCTS/az_node.pyx"])]

setup(
    name="mu_alpha_zero_library",
    version="1.0.9.1",
    description="Library for running and training MuZero and AlphaZero models.",
    author="Skyr",
    ext_modules=cythonize(ext_modules),
    install_requires=open("requirements.txt").read().strip().split("\n"),
    long_description=open('README.md', encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    package_data={
        "mu_alpha_zero": ["*.txt", "*.root"]
    },
    url="https://github.com/Skirlax/MuAlphaZeroLibrary",

)
