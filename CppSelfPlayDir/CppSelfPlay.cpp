//
// Created by Skyr on 16.03.2024.
//

#include <pybind11/pybind11.h>
#include "MCTS/MuzeroSearchTree.h"
#include "Network/MuZeroDefaultNet.h"
#include "Buffers/MuZeroFrameBuffer.h"
#include "torch/torch.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "pybind11/functional.h"
#include "pybind11/iostream.h"
#include "pybind11/chrono.h"
#include "pybind11/embed.h"
#include "omp.h"
namespace py = pybind11;

vector<MuZeroDefaultNet> makeNetworks(int numNetworks, string networkPath) {
    vector<MuZeroDefaultNet> networks;
    for (int i = 0; i < numNetworks; ++i) {
        MuZeroDefaultNet net(networkPath);
        networks.push_back(net);
    }
    return networks;
}

vector<MuzeroSearchTree> makeTrees(int numTrees, py::object gameManager, map<string, py::object> configArgs,
                                   int noopAction) {
    vector<MuzeroSearchTree> trees;
    for (int i = 0; i < numTrees; ++i) {
        MuzeroSearchTree tree(gameManager, configArgs);
        trees.push_back(tree);
    }
    return trees;
}

std::vector<std::tuple<PlayeOneStepReturn> > runParallelSelfPlay(string netPath, py::object gameManager,
                                                                 map<string, py::object> configArgs, int numGames,
                                                                 int numProcesses, int noopAction) {
    int numGamesPerProcess = static_cast<int>(numGames / numProcesses);
    std::vector<std::tuple<PlayeOneStepReturn> > histories;
    omp_set_num_threads(numProcesses);
#pragma omp parallel default(none) shared(histories, numGamesPerProcess, numGames, std::cout)
    {
        MuZeroDefaultNet net = MuZeroDefaultNet(netPath);
        MuzeroSearchTree tree = MuzeroSearchTree(gameManager, configArgs);
        std::cout << "Thread " << omp_get_thread_num() << " started." << std::endl;

#pragma omp for nowait
        for (int i = 0; i < numGames; ++i) {
            try {
                std::tuple<PlayeOneStepReturn> gameReturn = tree.playOneGame(net);

#pragma omp critical
                {
                    histories.push_back(gameReturn);
                }
            } catch (const std::exception &e) {
                std::cerr << "Caught exception in thread " << omp_get_thread_num() << ": " << e.what() << std::endl;
            }
        }
    }
    std::cout << "Finished parallel self play." << std::endl;
    return histories;
}


PYBIND11_MODULE(CppSelfPlay, m) {
    m.def("runParallelSelfPlay", &runParallelSelfPlay, "Run parallel self play");
}


