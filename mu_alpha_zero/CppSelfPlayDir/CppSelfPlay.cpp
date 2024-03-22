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
using namespace std;
namespace py = pybind11;

vector<MuZeroDefaultNet> makeNetworks(int numNetworks, string networkPath) {
    vector<MuZeroDefaultNet> networks;
    for (int i = 0; i < numNetworks; ++i) {
        MuZeroDefaultNet net(networkPath);
        networks.push_back(net);
    }
    return networks;
}

std::vector<std::tuple<std::map<int, double>,double, std::tuple<double, int, double>, std::string>> unpackToPython(vector<vector<PlayeOneStepReturn>> histories)
{
	std::vector<std::tuple<std::map<int, double>,double, std::tuple<double, int, double>, std::string>> unpackedHistories;
	for (auto& history : histories)
	{
		for (auto& step : history){
			unpackedHistories.push_back(std::make_tuple(step.probabilities,step.v,step.info,step.framePath));
		}
	}
	return unpackedHistories;
}



std::vector < std::tuple<std::map<int, double>, double, std::tuple<double, int, double>, std::string>> runParallelSelfPlay(string netPath, py::object gameManager,
                                                                 map<string, py::object> configArgs, int numGames,
                                                                 int numProcesses, int noopAction) {
    std::cout << "Starting parallel self play." << std::endl;
    py::gil_scoped_release release;
    int numGamesPerProcess = static_cast<int>(numGames / numProcesses);
    unique_ptr<vector<std::vector<PlayeOneStepReturn>>> histories = make_unique<vector<vector<PlayeOneStepReturn>>>();
    omp_set_num_threads(numProcesses);
#pragma omp parallel default(none) shared(histories, numGamesPerProcess, numGames)
    {
        cout << "Thread " << omp_get_thread_num() << " starting." << endl;
        unique_ptr<MuZeroDefaultNet> net = std::make_unique<MuZeroDefaultNet>(netPath);
        unique_ptr<MuzeroSearchTree> tree = std::make_unique<MuzeroSearchTree>(gameManager, configArgs);

#pragma omp for nowait
        for (int i = 0; i < numGames; ++i) {
            try {
                unique_ptr<vector<PlayeOneStepReturn>> gameReturn = tree->playOneGame(net.get());

#pragma omp critical
                {
                    histories->push_back(*gameReturn);
                }
            } catch (const std::exception &e) {
                std::cerr << "Caught exception in thread " << omp_get_thread_num() << ": " << e.what() << std::endl;
            }
        }
    }
    std::cout << "Finished parallel self play." << std::endl;
    return unpackToPython(*histories);
}


PYBIND11_MODULE(CppSelfPlay, m) {
    m.def("runParallelSelfPlay", &runParallelSelfPlay, "Run parallel self play");
}


