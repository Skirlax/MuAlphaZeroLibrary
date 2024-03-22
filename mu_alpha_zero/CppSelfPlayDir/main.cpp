#include <iostream>
#include "pybind11/pybind11.h"
#include "torch/torch.h"
#include "MCTS/MuzeroSearchTree.h"
#include "pybind11/embed.h"
#include "MCTS/MuzeroSearchTree.h"
#include "Network/MuZeroDefaultNet.h"

namespace py = pybind11;

int main() {
    std::cout << "Hello" << std::endl;
    std::map<std::string, int> testArgs = { {"net_action_size",14},{"num_simulations",800},{"self_play_games",100},
        {"gamma",0.997},{"frame_buffer_size",32},{"frame_skip",4},{"num_steps",400},{"max_buffer_size",70000},
        {"tau",1},{"c",1},{"c2",19652}};
    py::scoped_interpreter guard{};

    py::module_::import("sys").attr("path").attr("append")("C\\Users\\Skyr\\PycharmProjects\\MuAlphaZeroBuild\\mu_alpha_zero\\Game\\asteroids");
    py::exec(R"(
import sys
sys.path.append(r"C:\Users\Skyr\CLionProjects\CppSelfPlay")
)");
    // convert teestArgs to string,py::object
    std::map<std::string, py::object> testArgsPy;
    for (auto const [key, value] : testArgs) {
        std::string keyStr = key;
        py::object valueObj = py::cast(value);
        testArgsPy[keyStr] = valueObj;
    }
    testArgsPy["target_resolution"] = py::make_tuple(96,96);
    testArgsPy["pickle_dir"] = py::cast("C:\\Users\\Skyr\\PycharmProjects\\testMuAlphaZeroLib\\Data");
    // start the interpreter
    auto asteroid = py::module_::import("asteroids");
    py::object obj = asteroid.attr("Asteroids")();
    std::unique_ptr<MuZeroDefaultNet> net = std::make_unique<MuZeroDefaultNet>(R"(C:\Users\Skyr\PycharmProjects\testMuAlphaZeroLib\Checkpoints\script_exported_net.pth)");
    std::unique_ptr<MuzeroSearchTree> tree = std::make_unique<MuzeroSearchTree>(obj, testArgsPy);
    tree->playOneGame(net.get());

}
