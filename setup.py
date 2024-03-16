from setuptools import setup
from torch.utils import cpp_extension

setup(name="mu_alpha_zero_library", version="1.0.5",
    description="Library for running and training MuZero and AlphaZero models.", author="Skyr", ext_modules=[
        cpp_extension.CppExtension('cpp_self_play', ['CppSelfPlay/MCTS/Node.cpp', 'CppSelfPlay/MCTS/Node.h',
                                                     'CppSelfPlay/MCTS/MuzeroSearchTree.cpp',
                                                     'CppSelfPlay/MCTS/MuzeroSearchTree.h',
                                                     'CppSelfPlay/Buffers/MuZeroFrameBuffer.cpp',
                                                     'CppSelfPlay/Buffers/MuZeroFrameBuffer.h', 'CppSelfPlay/utils.cpp',
                                                     'CppSelfPlay/utils.h', 'CppSelfPlay/Network/MuZeroDefaultNet.cpp',
                                                     'CppSelfPlay/Network/MuZeroDefaultNet.h'
                                                     'CppSelfPlay/CppSelfPlay.cpp'],
                                   include_dirs=['CppSelfPlay/', 'CppSelfPlay/MCTS/', 'CppSelfPlay/Buffers/',
                                                 'CppSelfPlay/Network/', 'CppSelfPlay/Network/'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=open("requirements.txt").read().strip().split("\n"),
    long_description=open('README.md', encoding="utf-8").read(), long_description_content_type='text/markdown',
    package_data={"mu_alpha_zero": ["*.txt", "*.root"]}, url="https://github.com/Skirlax/MuAlphaZeroLibrary",

)
