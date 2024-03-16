from setuptools import setup
from torch.utils import cpp_extension

setup(name="mu_alpha_zero_library", version="1.0.5",
    description="Library for running and training MuZero and AlphaZero models.", author="Skyr", ext_modules=[
        cpp_extension.CppExtension('CppSelfPlay', ['CppSelfPlayDir/MCTS/Node.cpp',
                                                     'CppSelfPlayDir/MCTS/MuzeroSearchTree.cpp',
                                                     'CppSelfPlayDir/Buffers/MuZeroFrameBuffer.cpp',
                                                     'CppSelfPlayDir/utils.cpp',
                                                     'CppSelfPlayDir/Network/MuZeroDefaultNet.cpp',
                                                     'CppSelfPlayDir/CppSelfPlayDir.cpp'],
                                   include_dirs=['CppSelfPlayDir/', 'CppSelfPlayDir/MCTS/', 'CppSelfPlayDir/Buffers/',
                                                 'CppSelfPlayDir/Network/', 'CppSelfPlayDir/Network/'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=open("requirements.txt").read().strip().split("\n"),
    long_description=open('README.md', encoding="utf-8").read(), long_description_content_type='text/markdown',
    package_data={"mu_alpha_zero": ["*.txt", "*.root"]}, url="https://github.com/Skirlax/MuAlphaZeroLibrary",

)
