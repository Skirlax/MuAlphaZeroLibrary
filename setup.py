from setuptools import setup

setup(
    name="mu_alpha_zero_library",
    version="1.0.4.1",
    description="Library for running and training MuZero and AlphaZero models.",
    author="Skyr",
    install_requires=open("requirements.txt").read().strip().split("\n"),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    package_data={
        "mu_alpha_zero": ["*.txt", "*.root"]
    },
    url="https://github.com/Skirlax/MuAlphaZeroLibrary",

)
