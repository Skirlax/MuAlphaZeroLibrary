from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name="mu_alpha_zero_library", version="1.0.5",
      description="Library for running and training MuZero and AlphaZero models.", author="Skyr", ext_modules=[
        cpp_extension.CppExtension('CppSelfPlay',
                                   ['mu_alpha_zero/CppSelfPlayDir/Buffers/MuZeroFrameBuffer.cpp', 'mu_alpha_zero/CppSelfPlayDir/MCTS/Node.cpp',
                                    'mu_alpha_zero/CppSelfPlayDir/MCTS/MuzeroSearchTree.cpp', 'mu_alpha_zero/CppSelfPlayDir/utils.cpp',
                                    'mu_alpha_zero/CppSelfPlayDir/Network/MuZeroDefaultNet.cpp', 'mu_alpha_zero/CppSelfPlayDir/CppSelfPlay.cpp'],
                                   include_dirs=['mu_alpha_zero/CppSelfPlayDir/', 'mu_alpha_zero/CppSelfPlayDir/MCTS/', 'mu_alpha_zero/CppSelfPlayDir/Buffers/',
                                                 'mu_alpha_zero/CppSelfPlayDir/Network/', 'mu_alpha_zero/CppSelfPlayDir/Network/'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages(),
      install_requires=open("requirements.txt").read().strip().split("\n"),
      long_description=open('README.md', encoding="utf-8").read(), long_description_content_type='text/markdown',
      package_data={"mu_alpha_zero": ["*.txt", "*.root"]}, url="https://github.com/Skirlax/MuAlphaZeroLibrary",

      )
