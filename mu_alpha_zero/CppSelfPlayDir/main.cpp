#include <iostream>
#include "pybind11/pybind11.h"
#include "torch/torch.h"
#include "MCTS/MuzeroSearchTree.h"

namespace py = pybind11;

int main() {
    std::map<std::string, double> testArgs = { {"net_action_size",14},{"num_simulations",800},{"self_play_games",100},
        {"gamma",0.997},{"frame_buffer_size",32},{"frame_skip",4},{"num_steps",400},{"max_buffer_size",70000},{"tau",1},{"c",1},{"c2",19652}};

    py::object testArgsObj = py::cast(testArgs);
    //MuzeroSearchTree tree = new MuzeroSearchTree()

}
